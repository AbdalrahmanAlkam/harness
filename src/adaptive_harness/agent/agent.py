"""Autonomous Developer Agent with pervasive ML classification, clarification dialogs, and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
import json
import time
from typing import Any, Callable, Dict, Generator, List, Optional
from pathlib import Path

from adaptive_harness.classifiers.ambiguity_classifier import AmbiguityAssessment, AmbiguityClassifier
from adaptive_harness.classifiers.complexity_router import ComplexityRouter, ComplexityRoutingResult
from adaptive_harness.classifiers.skill_classifier import SkillClassificationResult, SkillClassifier
from adaptive_harness.classifiers.verification_classifier import VerificationAssessment, VerificationClassifier
from adaptive_harness.classifiers.risk_classifier import ToolRiskClassifier
from adaptive_harness.classifiers.engine import BaseClassifierBackend, SklearnBackend, DOMAIN_LABELS, THINKING_LABELS
from adaptive_harness.classifiers.domain_classifier import DomainClassifier, DomainAssessment, DOMAIN_GUIDANCE, DomainMode
from adaptive_harness.classifiers.thinking_classifier import ThinkingClassifier, ThinkingAssessment, ThinkingLevel, BUDGET_TOKENS
from adaptive_harness.classifiers.skill_classifier import SKILL_CLASSES
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.llm.mock_client import ToolCall
from adaptive_harness.models.domain import ExecutionAttempt, ExecutionTrace, VerificationResult
from adaptive_harness.tools.base import Tool, ToolResult
from adaptive_harness.tools.bash import RunBashTool
from adaptive_harness.tools.clarification import AskUserTool
from adaptive_harness.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from adaptive_harness.tools.testing import RunPytestTool
from adaptive_harness.tools.science import CalculateTool
from adaptive_harness.tools.research import WebSearchTool
from adaptive_harness.tools.workspace import ListDirectoryTool, SearchFilesTool


DEFAULT_SYSTEM_PROMPT = """You are Adaptive Agent, an expert AI software engineer.
You solve coding tasks by reading files, writing clean code, running tests, and executing bash commands.
Before modifying critical code, you think carefully. You verify your work using tests.
When asking clarifying questions, keep them targeted and actionable.
"""


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
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        workspace_root: Optional[str] = None,
        explicit_model: Optional[str] = None,
        classifier_backend: Optional[BaseClassifierBackend] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.repository = repository
        self.clarification_callback = clarification_callback
        self.system_prompt = system_prompt
        self.workspace_root = Path(workspace_root or ".").resolve()
        self.active_skills: dict[str, str] = {}
        self.explicit_model = explicit_model
        self.classifier_backend = classifier_backend or SklearnBackend()
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
            AskUserTool(callback=self._handle_clarification),
        ]
        if WebSearchTool().api_key:
            default_tools.append(WebSearchTool())
        tools_list = tools or default_tools
        self.tools: Dict[str, Tool] = {t.name: t for t in tools_list}
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def set_workspace(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise ValueError(f"Workspace directory does not exist: {target}")
        self.workspace_root = target
        for tool in self.tools.values():
            if hasattr(tool, "workspace_root"):
                tool.workspace_root = target

    def _handle_clarification(self, question: str, options: Optional[List[str]], context: Optional[str]) -> str:
        if self.clarification_callback is not None:
            return self.clarification_callback(question, options, context)
        # Headless execution cannot obtain authorization for a destructive action.
        if not options or (context and "destructive" in context.lower()):
            return "Action cancelled by user"
        return options[0]

    def run_stream(self, user_input: str, max_steps: int = 4) -> Generator[AgentEvent, None, None]:
        """Executes a task through the agentic lifecycle, yielding real-time events for the TUI."""
        start_time = time.perf_counter()

        # 1. Skill classification
        try:
            classification = self.classifier_backend.classify(user_input, SKILL_CLASSES)
            classifier_name = self.classifier_backend.name
            classifier_model = self.classifier_backend.model
        except Exception as exc:
            yield AgentEvent("classifier_error", {"backend": self.classifier_backend.name, "error": str(exc)})
            fallback = self.fallback_classifier
            classification = fallback.classify(user_input, SKILL_CLASSES)
            classifier_name, classifier_model = fallback.name, fallback.model
        probabilities = classification.probabilities
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        skill_res = SkillClassificationResult(classification.label, ranked[0][1], probabilities, ranked)
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
        ambiguity_res = self.ambiguity_classifier.evaluate(user_input, skill_res)
        yield AgentEvent(
            event_type="ambiguity_assessment",
            payload=ambiguity_res.to_dict(),
        )

        # If high ambiguity or risk is detected, ask the user before calling LLM!
        authorized_destructive = False
        if ambiguity_res.should_ask_question and ambiguity_res.suggested_question:
            yield AgentEvent(
                event_type="clarification_needed",
                payload={
                    "question": ambiguity_res.suggested_question,
                    "options": ambiguity_res.suggested_options,
                    "reason": ambiguity_res.reason,
                    "risk_level": ambiguity_res.risk_level,
                },
            )
            # Invoke clarification handler (pauses for user input in TUI modal)
            user_answer = self._handle_clarification(
                ambiguity_res.suggested_question,
                ambiguity_res.suggested_options,
                ambiguity_res.reason,
            )
            if user_answer.lower().startswith(("abort", "action cancelled")):
                yield AgentEvent("response", {"content": "Action cancelled by user.", "total_time_ms": round((time.perf_counter()-start_time)*1000, 2), "steps": 0, "success": False})
                return
            authorized_destructive = ambiguity_res.risk_level == "high" and user_answer.lower().startswith("proceed")
            yield AgentEvent(
                event_type="clarification_answered",
                payload={"answer": user_answer},
            )
            # Enrich context with clarification
            user_input = f"{user_input}\n[User Clarification]: {user_answer}"

        # 3. Model tier routing
        complexity_res = self.complexity_router.route(user_input)
        try:
            domain_prediction = self.classifier_backend.classify(user_input, DOMAIN_LABELS)
        except Exception as exc:
            yield AgentEvent("classifier_error", {"backend": self.classifier_backend.name, "error": f"Domain: {exc}"})
            domain_prediction = self.fallback_classifier.classify(user_input, DOMAIN_LABELS)
        try:
            thinking_prediction = self.classifier_backend.classify(user_input, THINKING_LABELS)
        except Exception as exc:
            yield AgentEvent("classifier_error", {"backend": self.classifier_backend.name, "error": f"Thinking: {exc}"})
            thinking_prediction = self.fallback_classifier.classify(user_input, THINKING_LABELS)
        domain_res = DomainAssessment(DomainMode(domain_prediction.label), max(domain_prediction.probabilities.values()))
        heuristic_domain = self.domain_classifier.classify(user_input)
        if heuristic_domain.confidence >= 0.8:
            domain_res = heuristic_domain
        predicted_level = ThinkingLevel(thinking_prediction.label)
        heuristic_thinking = self.thinking_classifier.classify(user_input)
        order = list(ThinkingLevel)
        if order.index(heuristic_thinking.level) > order.index(predicted_level):
            predicted_level = heuristic_thinking.level
        thinking_res = ThinkingAssessment(predicted_level, BUDGET_TOKENS[predicted_level],
                                          None if predicted_level == ThinkingLevel.NONE else
                                          "high" if predicted_level in (ThinkingLevel.DEEP, ThinkingLevel.EXTREME) else predicted_level.value)
        yield AgentEvent("domain_mode", {"mode": domain_res.mode.value, "confidence": domain_res.confidence,
                                          "latency_ms": round(domain_prediction.latency_ms, 2)})
        yield AgentEvent("thinking_budget", {"level": thinking_res.level.value, "tokens": thinking_res.budget_tokens,
                                               "latency_ms": round(thinking_prediction.latency_ms, 2),
                                               "classifier_total_ms": round(classification.latency_ms + domain_prediction.latency_ms + thinking_prediction.latency_ms, 2)})
        selected_model = self.explicit_model or complexity_res.recommended_model
        selected_tier = (next((tier for tier, model in self.complexity_router.tier_models.items()
                               if model == selected_model), "manual") if self.explicit_model else complexity_res.tier)
        yield AgentEvent(
            event_type="model_routing",
            payload={
                "tier": selected_tier,
                "model": selected_model,
                "selection": "forced" if self.explicit_model else "auto",
                "reasoning": complexity_res.reasoning,
            },
        )

        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})
        skill_guidance = "\n".join(f"Selected skill {name}:\n{content}" for name, content in self.active_skills.items())
        self.messages[0] = {"role": "system", "content": self.system_prompt + "\nWorkspace: " + str(self.workspace_root)
                            + "\nOperating mode: " + domain_res.mode.value + ". " + DOMAIN_GUIDANCE[domain_res.mode]
                            + ("\n" + skill_guidance if skill_guidance else "")}
        domain_tool_names = {
            DomainMode.CODING: set(self.tools),
            DomainMode.RESEARCH: {"read_file", "write_file", "list_directory", "search_files", "web_search", "ask_user"},
            DomainMode.SCIENCE: {"read_file", "write_file", "list_directory", "search_files", "run_bash", "run_pytest", "calculate", "ask_user"},
            DomainMode.AUDIT: {"read_file", "list_directory", "search_files", "run_bash", "run_pytest", "ask_user"},
        }[domain_res.mode]
        tool_schemas = [t.to_openai_schema() for t in self.tools.values() if t.name in domain_tool_names]

        # Agent execution loop (LLM call -> tool execution -> verification -> repeat if needed)
        step = 0
        final_answer = ""
        completed = False
        execution_attempts: List[ExecutionAttempt] = []
        unresolved_failures: set[str] = set()

        while step < max_steps:
            step += 1
            llm_resp = self.llm_client.complete(
                messages=self.messages,
                tools=tool_schemas,
                model=selected_model,
                tier=complexity_res.tier,
                reasoning_effort=thinking_res.effort,
            )
            if llm_resp.finish_reason == "error":
                yield AgentEvent("llm_error", {"message": llm_resp.content or "Unknown model error", "model": selected_model})
                final_answer = llm_resp.content or "Model request failed"
                break

            # Emit thought / content if present
            if llm_resp.content:
                yield AgentEvent(
                    event_type="thought",
                    payload={"content": llm_resp.content, "model": llm_resp.model},
                )
                final_answer = llm_resp.content

            # If no tool calls, task is finished
            if not llm_resp.tool_calls:
                self.messages.append({"role": "assistant", "content": llm_resp.content or ""})
                completed = not unresolved_failures
                break

            # Obtain authorization before adding tool calls to conversation history.
            for tc in llm_resp.tool_calls:
                risky_command = self.tool_risk_classifier.evaluate(tc.name, tc.arguments)
                if risky_command and not authorized_destructive:
                    question = f"Allow this destructive command? {risky_command}"
                    yield AgentEvent("clarification_needed", {"question": question,
                        "options": ["Proceed with this command", "Abort operation"],
                        "reason": "Destructive tool command detected", "risk_level": "high"})
                    answer = self._handle_clarification(question, ["Proceed with this command", "Abort operation"],
                                                        "Destructive tool command detected")
                    yield AgentEvent("clarification_answered", {"answer": answer})
                    if not answer.lower().startswith("proceed"):
                        yield AgentEvent("response", {"content": "Action cancelled by user.",
                            "total_time_ms": round((time.perf_counter()-start_time)*1000, 2),
                            "steps": step, "success": False})
                        return

            # Handle tool calls
            self.messages.append({
                "role": "assistant", "content": llm_resp.content or "",
                "tool_calls": [{"id": tc.id, "type": "function", "function": {
                    "name": tc.name, "arguments": json.dumps(tc.arguments)}} for tc in llm_resp.tool_calls],
            })
            for tc in llm_resp.tool_calls:
                yield AgentEvent(
                    event_type="tool_call",
                    payload={"name": tc.name, "arguments": tc.arguments, "call_id": tc.id},
                )

                tool = self.tools.get(tc.name)
                t_start = time.perf_counter()
                if tool:
                    try:
                        tool_res = tool.execute(**tc.arguments)
                    except Exception as exc:
                        tool_res = ToolResult(success=False, output="", error=f"Tool error: {type(exc).__name__}: {exc}")
                else:
                    tool_res = ToolResult(success=False, output="", error=f"Unknown tool `{tc.name}`")
                if tool_res.success and tc.name in ("write_file", "edit_file") and str(tc.arguments.get("path", "")).endswith(".py"):
                    try:
                        source_path = (tool.workspace_root / tc.arguments["path"]).resolve()
                        ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
                    except (SyntaxError, OSError) as exc:
                        tool_res = ToolResult(success=False, output=tool_res.output, error=f"SyntaxError: Python AST validation failed: {exc}")
                t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                yield AgentEvent(
                    event_type="tool_result",
                    payload={
                        "name": tc.name,
                        "success": tool_res.success,
                        "output": tool_res.output,
                        "error": tool_res.error,
                        "time_ms": round(t_elapsed_ms, 2),
                    },
                )

                # 4. Verification Classifier on tool execution output
                verif_res = self.verification_classifier.evaluate(tc.name, tool_res)
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

                # Append tool result to conversation history
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": tool_res.output if tool_res.success else f"ERROR: {tool_res.error}",
                    }
                )

        total_wall_ms = (time.perf_counter() - start_time) * 1000.0

        # Emit completion
        yield AgentEvent(
            event_type="response",
            payload={
                "content": final_answer,
                "total_time_ms": round(total_wall_ms, 2),
                "steps": step,
                "success": completed,
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
            except Exception:
                pass
