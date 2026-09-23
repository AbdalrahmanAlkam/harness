"""Autonomous Developer Agent with pervasive ML classification, clarification dialogs, and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Dict, Generator, List, Optional

from adaptive_harness.classifiers.ambiguity_classifier import AmbiguityAssessment, AmbiguityClassifier
from adaptive_harness.classifiers.complexity_router import ComplexityRouter, ComplexityRoutingResult
from adaptive_harness.classifiers.skill_classifier import SkillClassificationResult, SkillClassifier
from adaptive_harness.classifiers.verification_classifier import VerificationAssessment, VerificationClassifier
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.llm.mock_client import ToolCall
from adaptive_harness.models.domain import ExecutionAttempt, ExecutionTrace, VerificationResult
from adaptive_harness.tools.base import Tool, ToolResult
from adaptive_harness.tools.bash import RunBashTool
from adaptive_harness.tools.clarification import AskUserTool
from adaptive_harness.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from adaptive_harness.tools.testing import RunPytestTool
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
    ):
        self.llm_client = llm_client or LLMClient()
        self.repository = repository
        self.clarification_callback = clarification_callback
        self.system_prompt = system_prompt

        # Classifiers everywhere
        self.skill_classifier = SkillClassifier()
        self.ambiguity_classifier = AmbiguityClassifier()
        self.complexity_router = ComplexityRouter()
        self.verification_classifier = VerificationClassifier()

        # Tools setup
        default_tools = [
            RunBashTool(workspace_root=workspace_root),
            ReadFileTool(workspace_root=workspace_root),
            WriteFileTool(workspace_root=workspace_root),
            EditFileTool(workspace_root=workspace_root),
            ListDirectoryTool(workspace_root=workspace_root),
            SearchFilesTool(workspace_root=workspace_root),
            RunPytestTool(workspace_root=workspace_root),
            AskUserTool(callback=self._handle_clarification),
        ]
        tools_list = tools or default_tools
        self.tools: Dict[str, Tool] = {t.name: t for t in tools_list}
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def _handle_clarification(self, question: str, options: Optional[List[str]], context: Optional[str]) -> str:
        if self.clarification_callback is not None:
            return self.clarification_callback(question, options, context)
        # Default headless fallback
        return options[0] if options else "Approved to proceed"

    def run_stream(self, user_input: str, max_steps: int = 4) -> Generator[AgentEvent, None, None]:
        """Executes a task through the agentic lifecycle, yielding real-time events for the TUI."""
        start_time = time.perf_counter()

        # 1. Skill classification
        skill_res = self.skill_classifier.classify(user_input)
        yield AgentEvent(
            event_type="skill_classification",
            payload={
                "primary_skill": skill_res.primary_skill,
                "confidence": skill_res.confidence,
                "probabilities": skill_res.probabilities,
            },
        )

        # 2. Ambiguity & Clarification assessment ("questions in the middle of development")
        ambiguity_res = self.ambiguity_classifier.evaluate(user_input, skill_res)
        yield AgentEvent(
            event_type="ambiguity_assessment",
            payload=ambiguity_res.to_dict(),
        )

        # If high ambiguity or risk is detected, ask the user before calling LLM!
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
            yield AgentEvent(
                event_type="clarification_answered",
                payload={"answer": user_answer},
            )
            # Enrich context with clarification
            user_input = f"{user_input}\n[User Clarification]: {user_answer}"

        # 3. Model tier routing
        complexity_res = self.complexity_router.route(user_input)
        yield AgentEvent(
            event_type="model_routing",
            payload={
                "tier": complexity_res.tier,
                "model": complexity_res.recommended_model,
                "reasoning": complexity_res.reasoning,
            },
        )

        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})
        tool_schemas = [t.to_openai_schema() for t in self.tools.values()]

        # Agent execution loop (LLM call -> tool execution -> verification -> repeat if needed)
        step = 0
        final_answer = ""
        execution_attempts: List[ExecutionAttempt] = []

        while step < max_steps:
            step += 1
            llm_resp = self.llm_client.complete(
                messages=self.messages,
                tools=tool_schemas,
                model=complexity_res.recommended_model,
                tier=complexity_res.tier,
            )

            # Emit thought / content if present
            if llm_resp.content:
                yield AgentEvent(
                    event_type="thought",
                    payload={"content": llm_resp.content, "model": llm_resp.model},
                )
                final_answer = llm_resp.content

            # If no tool calls, task is finished
            if not llm_resp.tool_calls:
                break

            # Handle tool calls
            for tc in llm_resp.tool_calls:
                yield AgentEvent(
                    event_type="tool_call",
                    payload={"name": tc.name, "arguments": tc.arguments, "call_id": tc.id},
                )

                tool = self.tools.get(tc.name)
                t_start = time.perf_counter()
                if tool:
                    tool_res = tool.execute(**tc.arguments)
                else:
                    tool_res = ToolResult(success=False, output="", error=f"Unknown tool `{tc.name}`")
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
                policy=complexity_res.tier,
                attempts=execution_attempts,
                final_strategy=skill_res.primary_skill,
                success=True,
                classifier_top1=skill_res.primary_skill,
                total_time_ms=round(total_wall_ms, 2),
                metadata={
                    "model": complexity_res.recommended_model,
                    "tier": complexity_res.tier,
                    "ambiguity": ambiguity_res.to_dict(),
                },
            )
            try:
                self.repository.record_trace(trace)
            except Exception:
                pass
