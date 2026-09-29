"""The non-interactive contract: exit codes, a JSON event stream, and ceilings.

This is what makes the harness usable from CI rather than only from a terminal.
Three pieces, and each exists because a script cannot otherwise tell what
happened:

- **Exit codes.** A script's only signal is the exit status, so it has to carry
  a meaning. `0` succeeded, `1` the agent failed, `2` a gate refused, `3` a
  budget ran out, `4` a required classifier was unavailable. Without this every
  failure looks the same to a caller.
- **A JSON event stream.** `--json` emits one JSON object per event, so a
  harness-of-record can diff two runs, assert on tool usage, and count cost
  without scraping prose. The format is deliberately the *events*, not a
  summary: the summary is a lossy view of something already emitted.
- **Ceilings.** `--max-cost` and `--max-turns` halt deterministically with an
  attributable stop reason. A run that spends without bound is worse than one
  that fails, because the failure is the visible signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

# --- exit codes ------------------------------------------------------------

EXIT_OK = 0
#: The agent ran and did not complete the task.
EXIT_AGENT_FAILED = 1
#: A gate refused: a rule, a safety profile, a verification check, a plan the
#: operator rejected.
EXIT_GATE_REFUSED = 2
#: A budget ceiling was reached. Distinct from failure because the work may
#: have been fine and the ceiling simply too low.
EXIT_BUDGET_EXHAUSTED = 3
#: Something the run needed was not available: a classifier backend, a model
#: credential, a required toolchain. Distinct because it is usually fixable by
#: configuration rather than by changing the task.
EXIT_UNAVAILABLE = 4
#: The harness itself is broken. Reserved so a caller can tell "your run failed"
#: from "this program is misconfigured".
EXIT_INTERNAL_ERROR = 70

#: What a human reads. The numbers are the contract; these are the reason.
EXIT_MEANINGS = {
    EXIT_OK: "completed",
    EXIT_AGENT_FAILED: "the agent did not complete the task",
    EXIT_GATE_REFUSED: "a gate refused: a rule, a safety profile, or a check",
    EXIT_BUDGET_EXHAUSTED: "a budget ceiling was reached",
    EXIT_UNAVAILABLE: "something the run needed was not available",
    EXIT_INTERNAL_ERROR: "the harness failed",
}

#: Stop reasons that mean a gate refused rather than the agent failing.
#: `provider_error` is deliberately absent: a model that ran and declined the
#: work is the agent failing, and putting it here would tell a CI job to
#: investigate a policy when it should retry or re-prompt.
_GATE_STOPS = frozenset({
    "verification_failed", "skill_verification_failed", "quality_gate_unsubstantiated",
    "cancelled", "classifier_stop", "overseer_impasse", "missing_file_changes",
})

#: Stop reasons that mean a ceiling was reached.
_BUDGET_STOPS = frozenset({"step_limit", "max_cost", "max_turns"})

#: Stop reasons that mean something the run needed was not available. Distinct
#: from a gate refusing and from the agent failing, because it is usually
#: fixable by configuration rather than by changing the task.
_UNAVAILABLE_STOPS = frozenset({"provider_unavailable", "classifier_unavailable"})


def exit_code_for(stop_reason: str, *, success: bool = False,
                  unavailable: bool = False) -> int:
    """Map a run's outcome to the exit code a caller will branch on.

    Ordered so the most specific condition wins: a budget that ran out on an
    otherwise-successful run is a budget code, not a success, because the
    answer is incomplete either way.
    """
    if unavailable or stop_reason in _UNAVAILABLE_STOPS:
        return EXIT_UNAVAILABLE
    if stop_reason in _BUDGET_STOPS:
        return EXIT_BUDGET_EXHAUSTED
    if stop_reason in _GATE_STOPS:
        return EXIT_GATE_REFUSED
    return EXIT_OK if success else EXIT_AGENT_FAILED


# --- the JSON event stream --------------------------------------------------

#: Events that carry a payload a caller cannot reconstruct from the summary.
#: Everything is emitted by default; this is a hint for readers, not a filter.
NOTABLE_EVENTS = frozenset({
    "response", "tool_call", "tool_result", "tool_blocked", "hook_blocked",
    "plan_mode_blocked", "quality_gate", "context_budget", "requirements",
    "step_policy", "llm_error", "provider_failover", "operator_message",
})


class JsonStream:
    """Writes one JSON object per line.

    Line-delimited rather than a single array so a long run streams: a caller
    reading a pipe gets events as they happen and does not have to wait for the
    run to end to discover it failed.
    """

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self.events: List[Dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        """Emit one event. Never raises into the run.

        A stream that fails must not take the task down with it -- but a failure
        here means the caller is getting a partial transcript, so it is
        reported rather than swallowed.
        """
        payload = {
            "event": getattr(event, "event_type", "unknown"),
            "payload": _jsonable(getattr(event, "payload", {})),
            "timestamp": getattr(event, "timestamp", None),
        }
        self.events.append(payload)
        try:
            self._write(json.dumps(payload, ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001 - a broken sink is the caller's problem
            pass

    def consume(self, events: Iterable[Any]) -> List[Dict[str, Any]]:
        for event in events:
            self.emit(event)
        return self.events

    def summary(self) -> Dict[str, Any]:
        """The end-of-run block, so a caller gets one object rather than
        having to fold the stream itself."""
        last: Dict[str, Any] = {}
        tools: Dict[str, int] = {}
        tokens = 0
        for entry in self.events:
            payload = entry.get("payload") or {}
            if entry["event"] == "response":
                last = payload
                usage = payload.get("usage") or {}
                tokens = int(usage.get("total_tokens") or tokens)
            elif entry["event"] == "tool_call":
                name = str(payload.get("name", "?"))
                tools[name] = tools.get(name, 0) + 1
        code = exit_code_for(str(last.get("stop_reason", "")),
                             success=bool(last.get("success", False)))
        return {
            "event": "summary",
            "success": bool(last.get("success", False)),
            "stop_reason": last.get("stop_reason"),
            "exit_code": code,
            "exit_meaning": EXIT_MEANINGS.get(code, "unknown"),
            "total_time_ms": last.get("total_time_ms"),
            "steps": last.get("steps"),
            "total_tokens": tokens,
            "cost_usd": last.get("cost_usd"),
            "tool_calls": tools,
        }


def _jsonable(value: Any) -> Any:
    """Make an event payload safe for json.dumps without losing information."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --- ceilings ---------------------------------------------------------------

@dataclass
class Ceilings:
    """What a run is allowed to spend before it stops.

    A ceiling that only reports after the fact is a report, not a ceiling. Both
    are checked at every step, and hitting one stops the run with an
    attributable reason rather than an unexplained truncation.
    """

    max_cost_usd: Optional[float] = None
    max_turns: Optional[int] = None
    #: Where the cost came from, when the provider reported it.
    reported_cost_usd: float = 0.0
    cost_reported: bool = False
    turns: int = 0
    #: Set when a ceiling was hit, with the reason.
    breached: Optional[str] = None
    detail: str = ""

    def observe_turn(self) -> Optional[str]:
        """Called once per model turn."""
        self.turns += 1
        if self.max_turns is not None and self.turns > self.max_turns:
            self.breached = "max_turns"
            self.detail = f"reached the ceiling of {self.max_turns} turn(s)"
            return self.detail
        return None

    def observe_usage(self, usage: Any) -> Optional[str]:
        """Called when a response reports usage. The cost is cumulative."""
        if not isinstance(usage, dict):
            return None
        cost = usage.get("cost_usd")
        if isinstance(cost, (int, float)):
            # Add, not take a max: this is called once per response with that
            # response's cost, and a ceiling is about the run's total. Taking
            # the larger would mean a run spending 0.9 five times reports 0.9.
            self.reported_cost_usd += float(cost)
            self.cost_reported = True
        elif self.max_cost_usd:
            # The provider did not price the call. Rather than pretend the
            # ceiling was met, estimate from tokens at a deliberately
            # conservative rate and say so. Several providers report only the
            # prompt and completion counts, so the total is summed rather than
            # read -- reading a key that is often absent means a cost ceiling
            # silently never fires.
            total = (usage.get("total_tokens")
                     or int(usage.get("prompt_tokens", 0) or 0)
                     + int(usage.get("completion_tokens", 0) or 0))
            if total:
                self.reported_cost_usd += (
                    int(total) / 1_000_000 * ESTIMATED_USD_PER_MTOK)
        if self.max_cost_usd is not None and self.reported_cost_usd >= self.max_cost_usd:
            self.breached = "max_cost"
            basis = "reported" if self.cost_reported else "estimated from tokens"
            self.detail = (f"reached the ceiling of ${self.max_cost_usd:.4f} "
                           f"(${self.reported_cost_usd:.4f} {basis})")
            return self.detail
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"max_cost_usd": self.max_cost_usd, "max_turns": self.max_turns,
                "turns": self.turns, "reported_cost_usd": self.reported_cost_usd,
                "cost_reported": self.cost_reported, "breached": self.breached,
                "detail": self.detail}


#: Used only when the provider does not price a call. Deliberately high: a
#: ceiling that stops a run early is a bug report, and under-estimating would
#: do that.
ESTIMATED_USD_PER_MTOK = 15.0
