"""Bounded local supervision of agent tool trajectories."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
import json
import re
import time
from typing import Any


class OverseerState(StrEnum):
    HEALTHY_PROGRESS = "HEALTHY_PROGRESS"
    LOOPING_DETECTED = "LOOPING_DETECTED"
    HALLUCINATION_DETECTED = "HALLUCINATION_DETECTED"
    PROGRESS_STALLED = "PROGRESS_STALLED"
    SEMANTIC_DRIFT = "SEMANTIC_DRIFT"


@dataclass(frozen=True)
class OverseerDecision:
    state: OverseerState
    confidence: float
    directive: str = ""
    latency_ms: float = 0.0
    tier: str = "gate"


@dataclass(frozen=True)
class _Observation:
    name: str
    signature: str
    target: str
    success: bool
    error: str
    output: str


class RuntimeOverseer:
    """Fast evidence gate; optional local semantic resolver for uncertain drift."""

    def __init__(self, task: str, semantic_backend: Any = None):
        self.task = task
        self.semantic_backend = semantic_backend
        self.recent: deque[_Observation] = deque(maxlen=8)
        self.last_decision = OverseerDecision(OverseerState.HEALTHY_PROGRESS, 1.0)
        self.consecutive_interventions = 0
        self.intervention_states: list[OverseerState] = []

    def observe(self, name: str, arguments: dict[str, Any], *, success: bool,
                error: str = "", output: str = "", model_text: str = "") -> OverseerDecision:
        start = time.perf_counter()
        signature = name + ":" + json.dumps(arguments, sort_keys=True, default=str)
        target = str(arguments.get("path") or arguments.get("command") or "")[:160]
        self.recent.append(_Observation(name, signature, target, success, error[:600], output[:600]))
        state = OverseerState.HEALTHY_PROGRESS
        confidence = 0.96 if success else 0.65
        tier = "gate"
        tail = list(self.recent)
        failure_counts = [self._failure_count(item.error + " " + item.output) for item in tail[-3:]]
        named_files = set(re.findall(r"[\w./-]+\.(?:py|js|ts|md|json|toml|yaml|yml)", self.task, re.I))
        off_target_edit = (name in {"edit_file", "write_file"} and bool(named_files) and
                           target not in named_files and
                           not any(target.endswith("/" + file) for file in named_files))
        if (len(tail) >= 3 and len({item.signature for item in tail[-3:]}) == 1 and
                (tail[-1].success or name in {"edit_file", "write_file"})):
            state, confidence = OverseerState.LOOPING_DETECTED, 0.99
        elif len(tail) >= 4 and all(item.name in {"edit_file", "write_file"} for item in tail[-4:]) and \
                tail[-4].signature == tail[-2].signature and tail[-3].signature == tail[-1].signature:
            state, confidence = OverseerState.LOOPING_DETECTED, 0.95
        elif (not success and re.search(r"(?:not found|no such file|does not exist|unknown symbol)",
                                        error + " " + output, re.I) and
              re.search(r"\b(?:exists|found|confirmed|verified)\b", model_text, re.I)):
            state, confidence = OverseerState.HALLUCINATION_DETECTED, 0.94
        elif len(tail) >= 3 and all(not item.success for item in tail[-3:]) and (
                (all(count is not None for count in failure_counts) and
                 failure_counts[-1] >= failure_counts[-2] >= failure_counts[-3]) or
                (all(count is None for count in failure_counts) and
                 len({re.sub(r"\d+", "#", item.error.lower())[:100] for item in tail[-3:]}) <= 2)):
            state, confidence = OverseerState.PROGRESS_STALLED, 0.93
        elif off_target_edit and re.search(r"\b(?:only|solely|strictly|just)\b", self.task, re.I):
            state, confidence = OverseerState.SEMANTIC_DRIFT, 0.93
        elif (not success or off_target_edit) and self.semantic_backend is not None:
            # Semantic inference is paid only for ambiguous failure trajectories.
            try:
                labels = [item.value for item in OverseerState]
                result = self.semantic_backend.classify(
                    f"Task: {self.task[:300]}\nRecent tool: {name} {target}\nError: {error[:300]}", labels)
                if result.label in OverseerState._value2member_map_ and \
                        result.probabilities.get(result.label, 0) >= 0.7:
                    state = OverseerState(result.label)
                    confidence = result.probabilities[result.label]
                    tier = f"semantic:{getattr(self.semantic_backend, 'name', 'local')}"
            except Exception:
                pass  # Overseer failure must never break the agent.
        directive = self._directive(state, target, error or output)
        if state == OverseerState.HEALTHY_PROGRESS:
            self.consecutive_interventions = 0
        elif directive:
            self.consecutive_interventions += 1
            self.intervention_states.append(state)
        self.last_decision = OverseerDecision(state, confidence, directive,
            (time.perf_counter() - start) * 1000, tier)
        return self.last_decision

    @staticmethod
    def _failure_count(text: str) -> int | None:
        match = re.search(r"\b(\d+)\s+(?:tests?\s+)?failed\b", text, re.I)
        return int(match.group(1)) if match else None

    def check_claim(self, content: str) -> OverseerDecision | None:
        if not content or not self.recent:
            return None
        last = self.recent[-1]
        claim = bool(re.search(r"\b(?:tests? (?:pass|passed)|verified|successfully fixed|file exists)\b", content, re.I))
        if claim and not last.success:
            directive = self._directive(OverseerState.HALLUCINATION_DETECTED, last.target,
                                        last.error or last.output)
            self.consecutive_interventions += 1
            self.intervention_states.append(OverseerState.HALLUCINATION_DETECTED)
            self.last_decision = OverseerDecision(OverseerState.HALLUCINATION_DETECTED, 0.96,
                                                  directive)
            return self.last_decision
        return None

    def termination_verdict(self) -> OverseerDecision | None:
        """Classifier verdict that further attempts are pointless after failed interventions.

        Returns a verdict only when at least two interventions have been injected and
        the trajectory is still looping or stalled, i.e. corrective prompts did not
        get the model back on a productive path.
        """
        looping = [state for state in self.intervention_states
                   if state in {OverseerState.LOOPING_DETECTED, OverseerState.PROGRESS_STALLED}]
        if self.consecutive_interventions < 2 or not looping:
            return None
        state = looping[-1]
        directive = ("OVERSEER TERMINATION VERDICT: The classifier detects no path to progress after "
                     "repeated interventions (" + state.value + "). Further tool calls are unnecessary. "
                     "The run is stopped here; report the verified results obtained so far and the "
                     "concrete blocker that prevented completion.")
        self.last_decision = OverseerDecision(state, 0.99, directive,
            self.last_decision.latency_ms, "verdict")
        return self.last_decision

    def _directive(self, state: OverseerState, target: str, evidence: str) -> str:
        if state == OverseerState.LOOPING_DETECTED:
            return ("OVERSEER INTERVENTION: Repeated tool actions are looping on " + repr(target) +
                    ". Stop repeating the same edit or command. Read the relevant module afresh, "
                    "identify why prior attempts failed, and use a different strategy before editing again.")
        if state == OverseerState.HALLUCINATION_DETECTED:
            return ("OVERSEER INTERVENTION: Reality check failed. The available tool evidence says " +
                    repr(evidence[:250]) + ". Do not claim success or assume file contents. "
                    "Use read_file or search_files to ground the next step.")
        if state == OverseerState.PROGRESS_STALLED:
            return ("OVERSEER INTERVENTION: Three attempts made no verified progress. "
                    "Isolate the smallest failing case, inspect its error, and change strategy. "
                    "Use run_python_repl only when numerical or symbolic reproduction is appropriate.")
        if state == OverseerState.SEMANTIC_DRIFT:
            return ("OVERSEER INTERVENTION: Scope may have drifted. Re-anchor the next action to the "
                    f"user's original task: {self.task[:400]!r}. Explain how any proposed edit serves it.")
        return ""
