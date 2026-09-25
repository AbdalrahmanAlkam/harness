"""Autonomous Developer Agent with pervasive ML classification, clarification dialogs, and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Dict, Generator, List, Optional
from pathlib import Path

from adaptive_harness.classifiers.ambiguity_classifier import AmbiguityAssessment, AmbiguityClassifier
from adaptive_harness.classifiers.complexity_router import ComplexityRouter, ComplexityRoutingResult
from adaptive_harness.classifiers.skill_classifier import SkillClassificationResult, SkillClassifier
from adaptive_harness.classifiers.verification_classifier import VerificationAssessment, VerificationClassifier
from adaptive_harness.classifiers.risk_classifier import ToolRiskClassifier
from adaptive_harness.classifiers.engine import (BaseClassifierBackend, SklearnBackend, DOMAIN_LABELS,
    THINKING_LABELS, TIER_LABELS, VERIFICATION_LABELS, Classification)
from adaptive_harness.classifiers.domain_classifier import (DomainClassifier, DomainAssessment,
    DomainMode, parse_domain_mode, audit_command_is_read_only)
from adaptive_harness.classifiers.thinking_classifier import (ThinkingClassifier, ThinkingAssessment,
    ThinkingLevel, BUDGET_TOKENS, parse_thinking_level)
from adaptive_harness.classifiers.skill_classifier import SKILL_CLASSES
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.llm.client import LLMClient, MODEL_TIERS
from adaptive_harness.llm.mock_client import ToolCall, LLMResponse
from adaptive_harness.models.domain import ExecutionAttempt, ExecutionTrace, VerificationResult
from adaptive_harness.tools.base import Tool, ToolResult
from adaptive_harness.tools.bash import RunBashTool
from adaptive_harness.tools.clarification import AskUserTool
from adaptive_harness.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from adaptive_harness.tools.testing import RunPytestTool
from adaptive_harness.tools.science import CalculateTool, CheckConvergenceTool
from adaptive_harness.tools.python_repl import RunPythonReplTool, VerifyEquationTool
from adaptive_harness.tools.plotting import PlotTerminalTool
from adaptive_harness.agent.compaction import rank_search_results
from adaptive_harness.data.preferences import ClarificationMemory
from adaptive_harness.tools.research import WebSearchTool
from adaptive_harness.tools.workspace import ListDirectoryTool, SearchFilesTool
from adaptive_harness.tools.delegation import DelegateSubagentTool
from adaptive_harness.agent.skills import SkillCatalog
from adaptive_harness.skills.router import SkillRouter
from adaptive_harness.skills.verifier import SkillVerifier
from adaptive_harness.classifiers.runtime_overseer import RuntimeOverseer, OverseerState
from adaptive_harness.agent.context_window import prepare_context
from adaptive_harness.agent.code_fallback import extract_file_calls, requests_file_changes
from adaptive_harness.prompts import PromptRegistry, DEFAULT_PROMPTS


# Compatibility export; the editable source of truth is prompts.py ("system.default").
DEFAULT_SYSTEM_PROMPT = DEFAULT_PROMPTS["system.default"]

STEP_POLICIES = ("classifier", "fixed", "unbounded")

# Previous hardcoded tool-step budgets, selectable via step_policy="fixed".
FIXED_STEP_BUDGETS = {ThinkingLevel.NONE: 4, ThinkingLevel.LOW: 8, ThinkingLevel.MEDIUM: 12,
                      ThinkingLevel.DEEP: 16, ThinkingLevel.EXTREME: 20}




def _injection_event(source: str, content: str, step: int, kind: str = "system_message",
                     state: str = "") -> AgentEvent:
    """Make every classifier/harness prompt injection visible to the user."""
    return AgentEvent("prompt_injection", {"source": source, "kind": kind, "state": state,
                                          "content": content, "step": step})


_WORKSPACE_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".tox",
                        ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _workspace_signature(root: Path) -> dict[str, tuple[int, int]]:
    """Cheap observable file state (path -> mtime_ns, size) for mutation detection.

    Shell commands may legitimately create or modify files; evidence of that is
    file state, not which tool was used. Bounded by skip dirs and entry count.
    """
    signature: dict[str, tuple[int, int]] = {}
    try:
        for path in root.rglob("*"):
            if len(signature) >= 50_000:
                break
            try:
                if any(part in _WORKSPACE_SKIP_DIRS for part in path.parts):
                    continue
                if path.is_file():
                    stat = path.stat()
                    signature[str(path.relative_to(root))] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
    except OSError:
        return signature
    return signature


def _cacheable_read_request(task: str) -> bool:
    """Admit only self-contained read requests to the zero-token answer cache."""
    return bool(re.fullmatch(
        r"(?i)(?:please\s+)?(?:read|view|show|inspect)(?:\s+(?:file|the file))?\s+"
        r"[\w./-]+\.(?:py|md|txt|json|toml)[.!]?", task.strip()))


@dataclass
class AgentEvent:
    """Telemetry and message events emitted during agent execution."""

    event_type: str  # 'classification', 'clarification_needed', 'thought', 'tool_call', 'tool_result', 'verification', 'response'
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class DeveloperAgent:
    """Full-featured AI Coding Agent powered by OpenRouter LLMs and multi-tier ML classifiers."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tools: Optional[List[Tool]] = None,
        repository: Optional[ExperienceRepository] = None,
        clarification_callback: Optional[Callable[[str, Optional[List[str]], Optional[str]], str]] = None,
        system_prompt: str | None = None,
        workspace_root: Optional[str] = None,
        explicit_model: Optional[str] = None,
        classifier_backend: Optional[BaseClassifierBackend] = None,
        overseer_backend: Optional[BaseClassifierBackend] = None,
        forced_mode: str | DomainMode | None = None,
        forced_thinking: str | ThinkingLevel | None = None,
        safety_profile: str = "turbo",
        preferences_dir: str | Path | None = None,
        swarm_enabled: bool = False,
        require_file_changes: bool | None = None,
        enable_skill_routing: bool = True,
        step_policy: str = "classifier",
        max_steps: int | None = None,
        prompts: PromptRegistry | None = None,
    ):
        if safety_profile not in {"turbo", "balanced", "cautious", "strict"}:
            raise ValueError("Safety profile must be turbo, balanced, cautious, or strict")
        if step_policy not in STEP_POLICIES:
            raise ValueError("Step policy must be classifier, fixed, or unbounded")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.safety_profile = safety_profile
        self.step_policy = step_policy
        self.max_steps = max_steps
        self.require_file_changes = require_file_changes
        self.enable_skill_routing = enable_skill_routing
        self.clarification_memory = ClarificationMemory(preferences_dir)
        self.llm_client = llm_client or LLMClient()
        self.repository = repository
        self.clarification_callback = clarification_callback
        self.workspace_root = Path(workspace_root or Path.cwd()).expanduser().resolve()
        self.prompts = prompts or PromptRegistry.for_workspace(self.workspace_root)
        self.system_prompt = (system_prompt if system_prompt is not None
                              else self.prompts.get("system.default"))
        self.active_skills: dict[str, str] = {}
        self.skill_catalog = SkillCatalog(self.workspace_root)
        self.skill_router = SkillRouter()
        self.skill_verifier = SkillVerifier()
        self.explicit_model = (None if explicit_model == "auto" else
                               explicit_model or getattr(self.llm_client, "default_model", MODEL_TIERS["standard"]))
        self.forced_mode = parse_domain_mode(forced_mode)
        self.forced_thinking = parse_thinking_level(forced_thinking)
        self.classifier_backend = classifier_backend or SklearnBackend()
        self.overseer_backend = overseer_backend
        self.fallback_classifier = SklearnBackend()
        self.domain_classifier = DomainClassifier()
        self.thinking_classifier = ThinkingClassifier()

        # Classifiers everywhere
        self.skill_classifier = SkillClassifier()
        self.ambiguity_classifier = AmbiguityClassifier()
        self.complexity_router = ComplexityRouter()
        self.verification_classifier = VerificationClassifier()
        self.tool_risk_classifier = ToolRiskClassifier()

        # Tools setup
        default_tools = [
            RunBashTool(workspace_root=workspace_root),
            ReadFileTool(workspace_root=workspace_root),
            WriteFileTool(workspace_root=workspace_root),
            EditFileTool(workspace_root=workspace_root),
            ListDirectoryTool(workspace_root=workspace_root),
            SearchFilesTool(workspace_root=workspace_root),
            RunPytestTool(workspace_root=workspace_root),
            CalculateTool(),
            CheckConvergenceTool(),
            RunPythonReplTool(),
            PlotTerminalTool(),
            VerifyEquationTool(),
            AskUserTool(callback=self._handle_clarification),
        ]
        if WebSearchTool().api_key:
            default_tools.append(WebSearchTool())
        tools_list = tools if tools is not None else default_tools
        self.tools: Dict[str, Tool] = {t.name: t for t in tools_list}
        self.swarm_enabled = False
        if swarm_enabled:
            self.enable_swarm(True)
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def enable_swarm(self, enabled: bool) -> None:
        self.swarm_enabled = enabled
        if enabled:
            def client_factory() -> LLMClient:
                source = self.llm_client
                if not isinstance(source, LLMClient):
                    return source
                return LLMClient(api_key=source.api_key, base_url=source.base_url,
                    default_model=source.default_model, force_mock=source.force_mock,
                    provider=source.provider, provider_keys=source.provider_keys,
                    backup_providers=source.backup_providers)
            self.tools["delegate_subagent"] = DelegateSubagentTool(self.workspace_root,
                llm_client_factory=client_factory,
                repository=self.repository,
                safety_profile=self.safety_profile,
                step_policy=self.step_policy,
                max_steps=self.max_steps,
                on_event=getattr(self, "subagent_event_callback", None))
        else:
            self.tools.pop("delegate_subagent", None)

    def set_workspace(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise ValueError(f"Workspace directory does not exist: {target}")
        self.workspace_root = target
        self.prompts = PromptRegistry.for_workspace(target)
        self.skill_catalog = SkillCatalog(target)
        for tool in self.tools.values():
            if hasattr(tool, "workspace_root"):
                tool.workspace_root = target

    def _handle_clarification(self, question: str, options: Optional[List[str]], context: Optional[str],
                              remember: bool = True) -> str:
        remembered = self.clarification_memory.lookup(question, context) if remember else None
        if remembered:
            return remembered
        if self.clarification_callback is not None:
            answer = self.clarification_callback(question, options, context)
            if remember:
                try:
                    self.clarification_memory.remember(question, answer, context)
                except OSError:
                    pass
            return answer
        # Headless execution cannot obtain authorization for a destructive action.
        if not options or (context and "destructive" in context.lower()):
            return "Action cancelled by user"
        return options[0]

    def _request_messages(self) -> List[Dict[str, Any]]:
        """The saved conversation is verbatim; request compaction is conditional."""
        return self.messages

    def run_stream(self, user_input: str, max_steps: Optional[int] = None) -> Generator[AgentEvent, None, None]:
        """Executes a task through the agentic lifecycle, yielding real-time events for the TUI."""
        start_time = time.perf_counter()
        original_input = user_input
        local_skill = self.skill_classifier.classify(user_input)
        preference_guidance = self.clarification_memory.guidance()
        settings_key = json.dumps({"mode": self.forced_mode.value if self.forced_mode else "auto",
                                   "thinking": self.forced_thinking.value if self.forced_thinking else "auto",
                                   "model": self.explicit_model or "auto", "skills": sorted(self.active_skills),
                                   "prompt": hashlib.sha256(self.system_prompt.encode()).hexdigest(),
                                   "preferences": hashlib.sha256(preference_guidance.encode()).hexdigest()}, sort_keys=True)
        if self.repository is not None and _cacheable_read_request(user_input):
            try:
                cached = self.repository.lookup_solution(user_input, str(self.workspace_root), settings_key)
            except Exception as exc:
                cached = None
                yield AgentEvent("storage_error", {"error": f"Could not read solution cache: {exc}"})
            if cached:
                yield AgentEvent("skill_classification", {
                    "primary_skill": local_skill.primary_skill,
                    "confidence": local_skill.confidence,
                    "probabilities": local_skill.probabilities,
                    "backend": "sklearn", "classifier_model": "TF-IDF + Logistic Regression",
                    "latency_ms": 0.0,
                })
                values = sorted(local_skill.probabilities.values(), reverse=True)
                yield AgentEvent("ambiguity_assessment", {
                    "entropy": -sum(value * math.log2(value) for value in values if value > 0),
                    "confidence_margin": values[0] - values[1] if len(values) > 1 else 1.0,
                    "risk_level": "low",
                })
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append({"role": "assistant", "content": cached["answer"]})
                yield AgentEvent("memory_hit", {"original_tokens": cached["original_tokens"]})
                yield AgentEvent("response", {"content": cached["answer"], "total_time_ms":
                    round((time.perf_counter() - start_time) * 1000, 2), "steps": 0,
                    "success": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
                return

        # 1. Skill classification
        semif_engine = getattr(self.classifier_backend, "engine", None)
        if (self.classifier_backend.name == "semif" and semif_engine is not None and
                not getattr(semif_engine, "loaded", False)):
            yield AgentEvent("classifier_loading", {"backend": "semif",
                "model": self.classifier_backend.model})
        try:
            classification = self.classifier_backend.classify(user_input, SKILL_CLASSES)
            classifier_name = self.classifier_backend.name
            classifier_model = self.classifier_backend.model
        except Exception as exc:
            yield AgentEvent("classifier_error", {"backend": self.classifier_backend.name, "error": str(exc)})
            fallback = self.fallback_classifier
            classification = fallback.classify(user_input, SKILL_CLASSES)
            self.classifier_backend = fallback
            classifier_name, classifier_model = fallback.name, fallback.model
            yield AgentEvent("classifier_fallback", {"backend": classifier_name, "model": classifier_model})
        probabilities = classification.probabilities
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        skill_res = SkillClassificationResult(classification.label, ranked[0][1], probabilities, ranked,
                                              classification.entropy, classification.margin)
        yield AgentEvent(
            event_type="skill_classification",
            payload={
                "primary_skill": skill_res.primary_skill,
                "confidence": skill_res.confidence,
                "probabilities": skill_res.probabilities,
                "backend": classifier_name,
                "classifier_model": classifier_model,
                "latency_ms": round(classification.latency_ms, 2),
            },
        )

        # 2. Ambiguity & Clarification assessment ("questions in the middle of development")
        ambiguity_res = self.ambiguity_classifier.evaluate(user_input, skill_res,
            semantic_decision=(self.safety_profile in {"balanced", "cautious", "strict"} and
                               classifier_name == "semif" and self.clarification_callback is not None),
            ask_on_ambiguity_phrase=(self.safety_profile in {"cautious", "strict"} and
                                     self.clarification_callback is not None))
        yield AgentEvent(
            event_type="ambiguity_assessment",
            payload=ambiguity_res.to_dict(),
        )

        # If high ambiguity or risk is detected, ask the user before calling LLM!
        authorized_destructive = False
        catastrophic_request = self.tool_risk_classifier.catastrophic_intent(user_input)
        if (catastrophic_request or
                self.safety_profile in {"cautious", "strict"} and
                ambiguity_res.should_ask_question and ambiguity_res.suggested_question):
            if catastrophic_request:
                ambiguity_res.should_ask_question = True
                ambiguity_res.risk_level = "high"
                ambiguity_res.reason = "Destructive catastrophic action detected"
                ambiguity_res.suggested_question = "This task may destroy disks or external data. Proceed?"
                ambiguity_res.suggested_options = ["Proceed with this command", "Abort operation"]
            remembered = (None if catastrophic_request else
                          self.clarification_memory.lookup(ambiguity_res.suggested_question, ambiguity_res.reason))
            if remembered:
                yield AgentEvent("clarification_memory_hit", {"question": ambiguity_res.suggested_question})
                user_input = f"{user_input}\n[Known User Preference]: {remembered}"
            else:
                yield AgentEvent("clarification_needed", {
                    "question": ambiguity_res.suggested_question,
                    "options": ambiguity_res.suggested_options,
                    "reason": ambiguity_res.reason,
                    "risk_level": ambiguity_res.risk_level})
                answer = self._handle_clarification(ambiguity_res.suggested_question,
                                                    ambiguity_res.suggested_options, ambiguity_res.reason,
                                                    remember=not catastrophic_request)
                if answer.lower().startswith(("abort", "action cancelled")):
                    yield AgentEvent("response", {"content": "Action cancelled by user.",
                        "total_time_ms": round((time.perf_counter()-start_time)*1000, 2),
                        "steps": 0, "success": False})
                    return
                authorized_destructive = ambiguity_res.risk_level == "high" and answer.lower().startswith("proceed")
                yield AgentEvent("clarification_answered", {"answer": answer})
                user_input = f"{user_input}\n[User Clarification]: {answer}"

        # 3. Model tier routing
        try:
            tier_prediction = self.classifier_backend.classify(user_input, TIER_LABELS)
        except Exception as exc:
            yield AgentEvent("classifier_error", {"backend": self.classifier_backend.name,
                                                    "error": f"Complexity: {exc}"})
            self.classifier_backend = self.fallback_classifier
            tier_prediction = self.fallback_classifier.classify(user_input, TIER_LABELS)
            classifier_name, classifier_model = self.fallback_classifier.name, self.fallback_classifier.model
            yield AgentEvent("classifier_fallback", {"backend": classifier_name, "model": classifier_model})
        complexity_res = self.complexity_router.route(user_input, tier_prediction.probabilities)
        if self.forced_mode is not None:
            domain_prediction = Classification(self.forced_mode.value,
                {label: float(label == self.forced_mode.value) for label in DOMAIN_LABELS}, 0.0)
        else:
            try:
                domain_prediction = self.classifier_backend.classify(user_input, DOMAIN_LABELS)
            except Exception as exc:
                yield AgentEvent("classifier_error", {"backend": self.classifier_backend.name, "error": f"Domain: {exc}"})
                self.classifier_backend = self.fallback_classifier
                domain_prediction = self.fallback_classifier.classify(user_input, DOMAIN_LABELS)
                classifier_name, classifier_model = self.fallback_classifier.name, self.fallback_classifier.model
                yield AgentEvent("classifier_fallback", {"backend": classifier_name, "model": classifier_model})
        if self.forced_thinking is not None:
            thinking_prediction = Classification(self.forced_thinking.value,
                {label: float(label == self.forced_thinking.value) for label in THINKING_LABELS}, 0.0)
        else:
            try:
                thinking_prediction = self.classifier_backend.classify(user_input, THINKING_LABELS)
            except Exception as exc:
                yield AgentEvent("classifier_error", {"backend": self.classifier_backend.name, "error": f"Thinking: {exc}"})
                self.classifier_backend = self.fallback_classifier
                thinking_prediction = self.fallback_classifier.classify(user_input, THINKING_LABELS)
                classifier_name, classifier_model = self.fallback_classifier.name, self.fallback_classifier.model
                yield AgentEvent("classifier_fallback", {"backend": classifier_name, "model": classifier_model})
        domain_res = DomainAssessment(DomainMode(domain_prediction.label), max(domain_prediction.probabilities.values()))
        heuristic_domain = self.domain_classifier.classify(user_input)
        if self.forced_mode is None and classifier_name != "semif" and heuristic_domain.confidence >= 0.8:
            domain_res = heuristic_domain
        predicted_level = ThinkingLevel(thinking_prediction.label)
        heuristic_thinking = self.thinking_classifier.classify(user_input)
        order = list(ThinkingLevel)
        if self.forced_thinking is None and classifier_name != "semif" and order.index(heuristic_thinking.level) > order.index(predicted_level):
            predicted_level = heuristic_thinking.level
        thinking_res = ThinkingAssessment(predicted_level, BUDGET_TOKENS[predicted_level],
                                          None if predicted_level == ThinkingLevel.NONE else
                                          "high" if predicted_level in (ThinkingLevel.DEEP, ThinkingLevel.EXTREME) else predicted_level.value)
        if self.forced_thinking is not None and self.explicit_model is None:
            forced_tier = ("fast" if predicted_level == ThinkingLevel.NONE else
                           "reasoning" if predicted_level in (ThinkingLevel.DEEP, ThinkingLevel.EXTREME) else
                           "standard")
            complexity_res = ComplexityRoutingResult(forced_tier,
                self.complexity_router.tier_models[forced_tier], 1.0,
                "Manual thinking level selected this model tier.")
        elif self.explicit_model is None:
            aligned_tier = ("fast" if predicted_level == ThinkingLevel.NONE else
                            "reasoning" if predicted_level in (ThinkingLevel.DEEP, ThinkingLevel.EXTREME) else
                            "standard" if predicted_level == ThinkingLevel.MEDIUM and complexity_res.tier == "fast" else
                            complexity_res.tier)
            if aligned_tier != complexity_res.tier:
                complexity_res = ComplexityRoutingResult(aligned_tier,
                    self.complexity_router.tier_models[aligned_tier],
                    thinking_prediction.probabilities.get(predicted_level.value, 0.0),
                    "Thinking classification adjusted the model tier.")
        # Tool-step limits: explicit max_steps always wins, otherwise the policy decides.
        # "classifier" leaves the budget open and stops looping via classifier interventions,
        # "fixed" restores the previous hardcoded per-thinking-level budgets (capped at 32),
        # "unbounded" runs without any cap or classifier intervention.
        explicit_limit = max_steps if max_steps is not None else self.max_steps
        if explicit_limit is not None:
            max_steps = max(1, int(explicit_limit))
            limit_source = "explicit"
        elif self.step_policy == "fixed":
            max_steps = max(1, min(FIXED_STEP_BUDGETS[thinking_res.level], 32))
            limit_source = "fixed-budget"
        else:
            max_steps = math.inf
            limit_source = self.step_policy
        yield AgentEvent("step_policy", {"policy": self.step_policy, "limit_source": limit_source,
            "max_steps": None if math.isinf(max_steps) else max_steps,
            "classifier_supervision": self.step_policy == "classifier"})
        yield AgentEvent("domain_mode", {"mode": domain_res.mode.value, "confidence": domain_res.confidence,
                                          "selection": "forced" if self.forced_mode else "auto",
                                          "latency_ms": round(domain_prediction.latency_ms, 2)})
        yield AgentEvent("thinking_budget", {"level": thinking_res.level.value, "tokens": thinking_res.budget_tokens,
                                               "selection": "forced" if self.forced_thinking else "auto",
                                               "effort": thinking_res.effort,
                                               "latency_ms": round(thinking_prediction.latency_ms, 2),
                                               "classifier_total_ms": round(classification.latency_ms + tier_prediction.latency_ms
                                                                             + domain_prediction.latency_ms
                                                                             + thinking_prediction.latency_ms, 2)})
        selected_model = self.explicit_model or (self.llm_client.get_model_for_tier(complexity_res.tier)
            if hasattr(self.llm_client, "get_model_for_tier") else complexity_res.recommended_model)
        selected_tier = (next((tier for tier, model in self.complexity_router.tier_models.items()
                               if model == selected_model), "manual") if self.explicit_model else complexity_res.tier)
        yield AgentEvent(
            event_type="model_routing",
            payload={
                "tier": selected_tier,
                "model": selected_model,
                "selection": "manual" if self.explicit_model else "auto",
                "reasoning": complexity_res.reasoning,
            },
        )

        catalog = self.skill_catalog.all()
        for missing_skill in set(self.active_skills) - set(catalog):
            self.active_skills.pop(missing_skill)
            yield AgentEvent("storage_error", {"error": f"Skill {missing_skill} is no longer installed; resuming automatic routing."})
        skill_selection = self.skill_router.route(original_input, catalog,
            self.classifier_backend, tuple(self.active_skills))
        selected_skills = skill_selection.skills if self.enable_skill_routing else ()
        security_skill_active = any(skill.name == "security_audit_scanner"
                                    for skill in selected_skills)
        selected_tool_names = set().union(*(set(skill.tools) for skill in selected_skills)) if selected_skills else None


        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})
        mutation_required = (requests_file_changes(original_input) if self.require_file_changes is None
                             else self.require_file_changes)
        successful_mutations = 0
        skill_guidance = "\n".join(
            f"Active skill: {skill.title}. {skill.instructions} Completion checks: {', '.join(skill.invariants) or 'none'}."
            for skill in selected_skills)
        self.messages[0] = {"role": "system", "content": self.system_prompt
                            + self.prompts.get("system.workspace_line") + str(self.workspace_root)
                            + ("\n" + self.prompts.get("system.swarm_coordinator") if self.swarm_enabled else "")
                            + self.prompts.get("system.mode_line") + domain_res.mode.value + ". "
                            + self.prompts.get(f"domain.guidance.{domain_res.mode.value}")
                            + "\n" + self.prompts.get("system.tool_guidance")
                            + (self.prompts.get("system.preferences_line") + preference_guidance if preference_guidance else "")
                            + ("\n" + skill_guidance if skill_guidance else "")}
        exemplar_ingested = False
        if self.repository is not None:
            try:
                exemplar = self.repository.lookup_verified_exemplar(str(self.workspace_root), original_input)
                if exemplar:
                    compact = {"task": exemplar["user_prompt"][:300],
                               "steps": exemplar["trajectory_steps"][:8],
                               "outcome": exemplar["final_solution"][:600]}
                    self.messages[0]["content"] += (self.prompts.get("system.exemplar_notice")
                        + json.dumps(compact, ensure_ascii=False)[:2200])
                    exemplar_ingested = True
                    yield AgentEvent("memory_exemplar", {"similarity": exemplar["similarity"]})
            except Exception as exc:
                yield AgentEvent("storage_error", {"error": f"Could not retrieve verified example: {exc}"})
        # Ingested system prompts are surfaced verbatim so the user can audit them.
        yield AgentEvent("system_prompt", {"content": self.messages[0]["content"],
            "ingested": {"workspace": str(self.workspace_root),
                         "mode": domain_res.mode.value,
                         "preferences": preference_guidance or "",
                         "skills": [skill.title for skill in selected_skills],
                         "memory_exemplar": exemplar_ingested},
            "step": 0})
        domain_tool_names = {
            DomainMode.CODING: set(self.tools) - {"check_convergence"},
            DomainMode.RESEARCH: {"read_file", "write_file", "list_directory", "search_files", "web_search", "run_bash", "calculate", "plot_terminal", "ask_user"},
            DomainMode.SCIENCE: {"read_file", "write_file", "edit_file", "list_directory", "search_files", "run_bash", "run_pytest", "calculate", "check_convergence", "run_python_repl", "verify_equation", "plot_terminal", "ask_user"},
            DomainMode.AUDIT: {"read_file", "list_directory", "search_files", "run_bash", "run_pytest", "ask_user"},
        }[domain_res.mode]
        if self.safety_profile == "turbo":
            domain_tool_names.discard("ask_user")
        if selected_tool_names is not None:
            domain_tool_names.intersection_update(selected_tool_names)
        if domain_res.mode == DomainMode.CODING or (mutation_required and domain_res.mode != DomainMode.AUDIT):
            # Skills add guidance; they must never remove the core coding tools.
            domain_tool_names.update({"read_file", "list_directory", "search_files", "write_file",
                                      "edit_file", "run_bash", "run_pytest"} & set(self.tools))
        if self.swarm_enabled and domain_res.mode != DomainMode.AUDIT:
            domain_tool_names.add("delegate_subagent")
        tool_schemas = [t.to_openai_schema() for t in self.tools.values() if t.name in domain_tool_names]
        yield AgentEvent("specialized_skill", {
            "skills": [{"name": skill.name, "title": skill.title, "category": skill.category,
                        "icon": skill.icon} for skill in selected_skills],
            "confidence": skill_selection.confidence, "selection": skill_selection.selection,
            "backend": skill_selection.backend, "latency_ms": round(skill_selection.latency_ms, 2),
            "tools": sorted(domain_tool_names & set(self.tools)),
        })

        # Agent execution loop (LLM call -> tool execution -> verification -> repeat if needed)
        step = 0
        final_answer = ""
        completed = False
        execution_attempts: List[ExecutionAttempt] = []
        trajectory_steps: list[dict[str, Any]] = []
        skill_observations: list[dict[str, Any]] = []
        cache_dependencies: dict[str, str] = {}
        cache_eligible = _cacheable_read_request(original_input)
        unresolved_failures: set[str] = set()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                 "reasoning_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0}
        reported_cost_usd = 0.0
        cost_reported = False
        cached_tokens = 0
        answer_parts: list[str] = []
        stop_reason = ""
        verification_followups = 0
        overseer_backend = self.overseer_backend or (self.classifier_backend
            if self.classifier_backend.name in {"onnx", "ollama", "local-slm"} else None)
        overseer = RuntimeOverseer(original_input, overseer_backend)
        context_limit = getattr(self, "context_window_override", None)

        while step < max_steps:
            step += 1
            request_messages, context_info = prepare_context(self._request_messages(), selected_model,
                tool_schemas, provider=getattr(self.llm_client, "provider", "openrouter"), limit=context_limit)
            yield AgentEvent("context_status", context_info)
            yield AgentEvent("agent_stage", {"stage": "thinking" if thinking_res.budget_tokens else "generating",
                                             "step": step, "model": selected_model,
                                             "thinking_tokens": thinking_res.budget_tokens})
            try:
                llm_resp = self.llm_client.complete(
                    messages=request_messages, tools=tool_schemas,
                    model=selected_model, tier=complexity_res.tier,
                    reasoning_effort=thinking_res.effort,
                    reasoning_budget_tokens=thinking_res.budget_tokens,
                )
            except Exception as exc:
                llm_resp = LLMResponse(content=f"Model request failed ({type(exc).__name__}). Retry the task or check the connection.",
                    finish_reason="error", usage={"prompt_tokens": 0, "completion_tokens": 0})
            response_usage = llm_resp.usage or {}
            if (llm_resp.metadata or {}).get("failed_over_from"):
                selected_model = llm_resp.model or self.llm_client.default_model
                yield AgentEvent("provider_failover", {"from": llm_resp.metadata["failed_over_from"],
                    "to": self.llm_client.provider, "model": selected_model})
            for token_type in usage:
                usage[token_type] += int(response_usage.get(token_type, 0) or 0)
            if not response_usage.get("total_tokens"):
                usage["total_tokens"] += int(response_usage.get("prompt_tokens", 0) or 0) + int(
                    response_usage.get("completion_tokens", 0) or 0)
            if response_usage.get("cost_usd") is not None:
                reported_cost_usd += float(response_usage["cost_usd"])
                cost_reported = True
            cached_tokens += int((llm_resp.usage or {}).get("cached_tokens", 0) or 0)
            if (llm_resp.metadata or {}).get("thinking_fallback"):
                reason = (llm_resp.metadata or {}).get("thinking_fallback_reason")
                notice = ("Older assistant tool history has no replayable thinking blocks; using provider default reasoning for this session. Start a new session to re-enable the selected thinking budget."
                          if reason == "legacy_tool_history" else
                          "The provider rejected the requested thinking settings; this request was retried with its default reasoning behavior.")
                yield AgentEvent("llm_notice", {"message": notice})
            if llm_resp.finish_reason == "error":
                yield AgentEvent("llm_error", {"message": llm_resp.content or "Unknown model error", "model": selected_model})
                final_answer = llm_resp.content or "Model request failed"
                stop_reason = "provider_error"
                break

            if (not llm_resp.tool_calls and llm_resp.content and
                    "write_file" in domain_tool_names):
                recovered = extract_file_calls(llm_resp.content, original_input)
                if recovered:
                    llm_resp.tool_calls = recovered
                    yield AgentEvent("llm_notice", {"message":
                        f"Recovered {len(recovered)} explicitly named file(s) from model output."})

            if llm_resp.content or llm_resp.tool_calls:
                # Expose the phase change to the TUI once the provider returns
                # from any hidden reasoning work and produces a visible result.
                yield AgentEvent("agent_stage", {"stage": "generating", "step": step,
                    "model": selected_model, "thinking_tokens": thinking_res.budget_tokens})

            # Emit thought / content if present
            if llm_resp.content:
                yield AgentEvent(
                    event_type="thought",
                    payload={"content": llm_resp.content, "model": llm_resp.model},
                )
                final_answer = llm_resp.content

            # A plain-text answer is not completion when a tool failed or the
            # selected skill still lacks required execution evidence.
            if not llm_resp.tool_calls:
                partial = llm_resp.content or ""
                answer_parts.append(partial)
                final_answer = "\n\n".join(part for part in answer_parts if part)
                assistant_message = {"role": "assistant", "content": llm_resp.content or ""}
                if (llm_resp.metadata or {}).get("reasoning_details"):
                    assistant_message["reasoning_details"] = llm_resp.metadata["reasoning_details"]
                self.messages.append(assistant_message)
                if mutation_required and successful_mutations == 0:
                    if step < max_steps and verification_followups < 2:
                        verification_followups += 1
                        yield AgentEvent("llm_notice", {"message": "The requested files have not been written; asking the model to use file tools."})
                        nudge = self.prompts.get("harness.missing_file_changes")
                        self.messages.append({"role": "user", "content": nudge})
                        yield _injection_event("harness", nudge, step, kind="user_nudge")
                        answer_parts.clear()
                        final_answer = ""
                        continue
                    completed = False
                    stop_reason = "missing_file_changes"
                    break
                if str(llm_resp.finish_reason).lower() in {"length", "max_tokens", "max_output_tokens"}:
                    if step < max_steps:
                        yield AgentEvent("llm_notice", {"message": "The model hit its response-length limit; continuing from the cutoff."})
                        nudge = self.prompts.get("harness.continuation")
                        self.messages.append({"role": "user", "content": nudge})
                        yield _injection_event("harness", nudge, step, kind="user_nudge")
                        continue
                    stop_reason = "response_length_limit"
                    break
                claim_problem = overseer.check_claim(final_answer) if self.step_policy != "unbounded" else None
                if claim_problem and verification_followups < 2 and step < max_steps:
                    yield AgentEvent("overseer", {"state": claim_problem.state.value,
                        "confidence": claim_problem.confidence, "directive": claim_problem.directive,
                        "latency_ms": claim_problem.latency_ms, "tier": claim_problem.tier})
                    self.messages.append({"role": "system", "content": claim_problem.directive})
                    yield _injection_event("claim_check", claim_problem.directive, step,
                                           state=claim_problem.state.value)
                    verification_followups += 1
                    answer_parts.clear()
                    final_answer = ""
                    continue
                pending_checks = self.skill_verifier.verify(selected_skills, skill_observations)
                missing_checks = [f"{check.skill}: {', '.join(check.missing)}"
                                  for check in pending_checks if not check.verified]
                if (unresolved_failures or missing_checks or not final_answer.strip()) and step < max_steps and verification_followups < 2:
                    verification_followups += 1
                    yield AgentEvent("llm_notice", {"message":
                        "The response lacks a final answer or required verification; continuing to repair and check it."})
                    nudge = self.prompts.get("harness.verification_repair",
                        failures=', '.join(sorted(unresolved_failures)) or 'none',
                        checks='; '.join(missing_checks) or 'none')
                    self.messages.append({"role": "user", "content": nudge})
                    yield _injection_event("harness", nudge, step, kind="user_nudge")
                    answer_parts.clear()
                    final_answer = ""
                    continue
                completed = bool(final_answer.strip()) and not unresolved_failures and not missing_checks
                stop_reason = ("verification_failed" if unresolved_failures or missing_checks else
                               "no_answer" if not final_answer.strip() else "completed")
                break

            # Obtain authorization before adding tool calls to conversation history.
            for tc in llm_resp.tool_calls:
                risky_command = self.tool_risk_classifier.evaluate(tc.name, tc.arguments,
                    catastrophic_only=self.safety_profile == "turbo")
                if self.safety_profile == "strict" and tc.name in {
                        "run_bash", "write_file", "edit_file", "run_pytest", "run_python_repl", "web_search"}:
                    risky_command = risky_command or f"strict profile approval for {tc.name}"
                if risky_command and not authorized_destructive:
                    strict_gate = self.safety_profile == "strict"
                    question = (f"Approve this action under the strict safety profile? {risky_command}"
                                if strict_gate else f"Allow this destructive command? {risky_command}")
                    yield AgentEvent("clarification_needed", {"question": question,
                        "options": ["Proceed with this command", "Abort operation"],
                        "reason": "Strict safety profile requires approval before tool execution" if strict_gate else
                                 "Destructive tool command detected", "risk_level": "high"})
                    answer = self._handle_clarification(question, ["Proceed with this command", "Abort operation"],
                        "Destructive tool command detected", remember=not strict_gate)
                    yield AgentEvent("clarification_answered", {"answer": answer})
                    if not answer.lower().startswith("proceed"):
                        yield AgentEvent("response", {"content": "Action cancelled by user.",
                            "total_time_ms": round((time.perf_counter()-start_time)*1000, 2),
                            "steps": step, "success": False, "usage": usage})
                        return

            # Handle tool calls
            assistant_tool_message = {
                "role": "assistant", "content": llm_resp.content or "",
                "tool_calls": [{"id": tc.id, "type": "function", "function": {
                    "name": tc.name, "arguments": json.dumps(tc.arguments)}} for tc in llm_resp.tool_calls],
            }
            if (llm_resp.metadata or {}).get("reasoning_details"):
                assistant_tool_message["reasoning_details"] = llm_resp.metadata["reasoning_details"]
            self.messages.append(assistant_tool_message)
            pending_directives: list[tuple[str, str]] = []
            for tc in llm_resp.tool_calls:
                yield AgentEvent("agent_stage", {"stage": "tool_running", "step": step, "tool": tc.name})
                yield AgentEvent(
                    event_type="tool_call",
                    payload={"name": tc.name, "arguments": tc.arguments, "call_id": tc.id},
                )

                tool = self.tools.get(tc.name) if tc.name in domain_tool_names else None
                t_start = time.perf_counter()
                # Shell and test tools may write files; mutation evidence is the
                # observable file state before/after, not the tool name.
                observes_writes = tool is not None and tc.name in {"run_bash", "run_pytest"}
                signature_before = _workspace_signature(self.workspace_root) if observes_writes else None
                if (tool and (domain_res.mode == DomainMode.AUDIT or security_skill_active) and tc.name == "run_bash" and
                        not audit_command_is_read_only(str(tc.arguments.get("command", "")))):
                    tool_res = ToolResult(success=False, output="",
                        error="Security review permits read-only git inspection and approved dependency scanners")
                elif tool:
                    try:
                        tool_res = tool.execute(**tc.arguments)
                    except Exception as exc:
                        tool_res = ToolResult(success=False, output="", error=f"Tool error: {type(exc).__name__}: {exc}")
                else:
                    tool_res = ToolResult(success=False, output="", error=f"Tool `{tc.name}` is unavailable in {domain_res.mode.value} mode")
                if tool_res.success and (tc.name in {"write_file", "edit_file"} or
                    tc.name == "delegate_subagent" and (tool_res.metadata or {}).get("role") == "coder"):
                    successful_mutations += 1
                elif tool_res.success and signature_before is not None and \
                        _workspace_signature(self.workspace_root) != signature_before:
                    successful_mutations += 1
                if tc.name != "read_file" or not tool_res.success:
                    cache_eligible = False
                elif tool_res.success:
                    try:
                        path = (self.workspace_root / str(tc.arguments["path"])).resolve()
                        if not path.is_relative_to(self.workspace_root) or not path.is_file():
                            cache_eligible = False
                        else:
                            cache_dependencies[str(path.relative_to(self.workspace_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
                    except (KeyError, OSError, ValueError):
                        cache_eligible = False
                if tool_res.success and domain_res.mode == DomainMode.RESEARCH and tc.name == "web_search":
                    urls = (tool_res.metadata.get("source_urls") or []) if tool_res.metadata else []
                    if not any(isinstance(url, str) and url.startswith(("https://", "http://")) for url in urls):
                        tool_res = ToolResult(success=False, output=tool_res.output,
                            error="Research search returned no cited sources; refine the query or use local documents")
                t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                shown_output = tool_res.output
                if len(shown_output) > 20_000:
                    shown_output = shown_output[:20_000] + "\n… output truncated at 20,000 characters; request a narrower read."
                model_output = tool_res.output
                if tc.name == "search_files" and tool_res.success:
                    model_output = rank_search_results(model_output, original_input,
                        getattr(self.classifier_backend, "engine", None)
                        if self.classifier_backend.name == "semif" else None)
                # Preserve tool evidence until the context gauge crosses its
                # threshold; prepare_context compacts only the request copy.
                model_error = (tool_res.error or "")[:600]
                estimated_saved = max(0, (len(tool_res.output) - len(model_output)) // 4)

                yield AgentEvent(
                    event_type="tool_result",
                    payload={
                        "name": tc.name,
                        "success": tool_res.success,
                        "output": shown_output,
                        "error": (tool_res.error or "")[:4000] or None,
                        "time_ms": round(t_elapsed_ms, 2),
                        "context_chars": len(model_output), "raw_chars": len(tool_res.output),
                        "saved_tokens_estimate": estimated_saved,
                        "chart": bool((tool_res.metadata or {}).get("chart")) or tc.name == "plot_terminal",
                    },
                )

                # 4. Verification Classifier on tool execution output
                yield AgentEvent("agent_stage", {"stage": "verifying", "step": step, "tool": tc.name})
                try:
                    verification_prediction = self.classifier_backend.classify(
                        f"Tool: {tc.name}\nSuccess: {tool_res.success}\nError: {model_error}\nOutput: {model_output}",
                        VERIFICATION_LABELS)
                except Exception as exc:
                    yield AgentEvent("classifier_error", {"backend": self.classifier_backend.name,
                                                            "error": f"Verification: {exc}"})
                    self.classifier_backend = self.fallback_classifier
                    verification_prediction = self.fallback_classifier.classify(
                        f"Tool: {tc.name}\nSuccess: {tool_res.success}\nError: {model_error}\nOutput: {model_output}",
                        VERIFICATION_LABELS)
                    yield AgentEvent("classifier_fallback", {"backend": self.fallback_classifier.name,
                                                               "model": self.fallback_classifier.model})
                verif_res = self.verification_classifier.evaluate(tc.name, tool_res,
                                                                   verification_prediction.label)
                if verif_res.needs_retry:
                    unresolved_failures.add(tc.name)
                elif tool_res.success:
                    unresolved_failures.discard(tc.name)
                yield AgentEvent(
                    event_type="verification",
                    payload={
                        "tool": tc.name,
                        "status": verif_res.status,
                        "needs_retry": verif_res.needs_retry,
                        "action": verif_res.recommended_action,
                        "diagnostic": verif_res.diagnostic_summary,
                    },
                )

                execution_attempts.append(
                    ExecutionAttempt(
                        strategy=tc.name,
                        success=tool_res.success,
                        time_ms=round(t_elapsed_ms, 2),
                        value=tool_res.output[:200] if tool_res.output else None,
                        error=tool_res.error,
                        verification=VerificationResult(
                            success=tool_res.success,
                            reason=verif_res.diagnostic_summary,
                        ),
                    )
                )
                recorded_args = ({key: str(tc.arguments[key])[:12_000] for key in
                                  ("path", "content", "target_text", "replacement_text")
                                  if key in tc.arguments} if tc.name in {"write_file", "edit_file"} else
                                 {"path": str(tc.arguments.get("path"))[:200]} if "path" in tc.arguments else {})
                trajectory_steps.append({"tool": tc.name, "arguments": recorded_args,
                                         "success": tool_res.success,
                                         "result": (tool_res.output if tool_res.success else tool_res.error or "")[:300]})
                skill_observations.append({"name": tc.name, "arguments": tc.arguments,
                                           "success": tool_res.success})

                # Append tool result to conversation history
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": model_output if tool_res.success else
                                   f"ERROR: {model_error or 'Tool failed'}\n{model_output}",
                    }
                )

                decision = overseer.observe(tc.name, tc.arguments, success=tool_res.success,
                    error=tool_res.error or "", output=tool_res.output, model_text=llm_resp.content or "")
                yield AgentEvent("overseer", {"state": decision.state.value,
                    "confidence": decision.confidence, "directive": decision.directive,
                    "latency_ms": round(decision.latency_ms, 2), "tier": decision.tier})
                if decision.directive and self.step_policy != "unbounded":
                    pending_directives.append((decision.state.value, decision.directive))
                    if self.step_policy == "classifier":
                        # The classifier judges whether corrective prompts can still
                        # restore productive work; if not, it stops the run outright.
                        verdict = overseer.termination_verdict()
                        if verdict is not None:
                            pending_directives.append((verdict.state.value, verdict.directive))
                            for state_name, text in pending_directives:
                                self.messages.append({"role": "system", "content": text})
                                yield _injection_event("runtime_overseer", text, step, state=state_name)
                            pending_directives.clear()
                            yield AgentEvent("overseer", {"state": verdict.state.value,
                                "confidence": verdict.confidence, "directive": verdict.directive,
                                "latency_ms": round(verdict.latency_ms, 2), "tier": verdict.tier})
                            final_answer = verdict.directive
                            stop_reason = "classifier_stop"
                            break
                        if decision.state in {OverseerState.LOOPING_DETECTED, OverseerState.PROGRESS_STALLED}:
                            # Unnecessary tool calls: inject a visible prompt to stop circling.
                            pending_directives.append((decision.state.value,
                                                       self.prompts.get("intervention.stop_circling")))
                    elif decision.state in {OverseerState.LOOPING_DETECTED, OverseerState.PROGRESS_STALLED} and \
                            overseer.consecutive_interventions >= 2:
                        question = (f"The agent remains {decision.state.value.lower().replace('_', ' ')} "
                                    f"after two interventions. Last action: {tc.name} {tc.arguments}.")
                        if self.safety_profile in {"turbo", "balanced"} or self.clarification_callback is None:
                            final_answer = question
                            stop_reason = "overseer_impasse"
                            break
                        yield AgentEvent("clarification_needed", {"question": question,
                            "options": ["Try a different approach", "Stop and report the impasse"],
                            "reason": "Runtime overseer detected repeated failure", "risk_level": "medium"})
                        answer = self._handle_clarification(question,
                            ["Try a different approach", "Stop and report the impasse"],
                            "Runtime overseer detected repeated failure", remember=False)
                        if answer.lower().startswith("stop"):
                            final_answer = question
                            stop_reason = "overseer_impasse"
                            break
                        overseer.consecutive_interventions = 0
            for state_name, text in pending_directives:
                self.messages.append({"role": "system", "content": text})
                yield _injection_event("runtime_overseer", text, step, state=state_name)
            if stop_reason in {"overseer_impasse", "classifier_stop"}:
                break

        skill_checks = self.skill_verifier.verify(selected_skills, skill_observations)
        for skill_check in skill_checks:
            yield AgentEvent("skill_verification", {"skill": skill_check.skill,
                "verified": skill_check.verified, "satisfied": skill_check.satisfied,
                "missing": skill_check.missing})
        if skill_checks and not all(check.verified for check in skill_checks):
            completed = False
            stop_reason = "skill_verification_failed"
        if not completed and not stop_reason and step >= max_steps:
            stop_reason = "step_limit"
        total_wall_ms = (time.perf_counter() - start_time) * 1000.0
        if self.repository is not None and completed and cache_eligible and cache_dependencies and final_answer:
            try:
                self.repository.save_solution(original_input, str(self.workspace_root), settings_key,
                                              final_answer, cache_dependencies,
                                              usage["prompt_tokens"] + usage["completion_tokens"])
            except Exception as exc:
                yield AgentEvent("storage_error", {"error": f"Could not save solution cache: {exc}"})
        if self.repository is not None:
            try:
                self.repository.save_agent_trace(str(self.workspace_root), original_input,
                    trajectory_steps, final_answer,
                    verified_success=bool(completed and execution_attempts and
                        any(item.success and item.verification and item.verification.success
                            for item in execution_attempts)))
            except Exception as exc:
                yield AgentEvent("storage_error", {"error": f"Could not save agent trace: {exc}"})

        # Emit completion
        yield AgentEvent(
            event_type="response",
            payload={
                "content": final_answer,
                "total_time_ms": round(total_wall_ms, 2),
                "steps": step,
                "success": completed,
                "usage": usage,
                "cached_tokens": cached_tokens,
                "cost_usd": reported_cost_usd,
                "cost_reported": cost_reported,
                "stop_reason": stop_reason or ("completed" if completed else "incomplete"),
            },
        )

        # Record trace in repository if available
        if self.repository is not None:
            trace = ExecutionTrace(
                task_id=f"dev-{int(time.time()*1000)}",
                task_text=user_input,
                classifier_probabilities=skill_res.probabilities,
                entropy=ambiguity_res.entropy,
                confidence=skill_res.confidence,
                margin=ambiguity_res.confidence_margin,
                policy=selected_tier,
                attempts=execution_attempts,
                final_strategy=skill_res.primary_skill,
                success=completed,
                classifier_top1=skill_res.primary_skill,
                total_time_ms=round(total_wall_ms, 2),
                metadata={
                    "model": selected_model,
                    "tier": selected_tier,
                    "ambiguity": ambiguity_res.to_dict(),
                },
            )
            try:
                self.repository.record_trace(trace)
            except Exception as exc:
                yield AgentEvent("storage_error", {"error": f"Could not save execution trace: {exc}"})
