"""Offline Mock LLM client providing intelligent simulated responses and tool calls."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]


class LLMResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    model: str = "mock-developer-agent"
    finish_reason: str = "stop"
    usage: Dict[str, int | float] = Field(default_factory=lambda: {"prompt_tokens": 100, "completion_tokens": 50})
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MockLLMClient:
    """Simulates an LLM for testing and offline development."""

    def __init__(self, default_model: str = "mock-llm"):
        self.default_model = default_model
        # Parameters of the most recent call, for assertions in tests.
        self.last_call: Dict[str, Any] = {}

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        # The agent passes routing and reasoning parameters that a real provider
        # uses. The mock has no notion of them, but it must accept them: without
        # them every offline call raises TypeError and the run degrades to
        # "Model request failed", which makes offline development useless.
        **kwargs: Any,
    ) -> LLMResponse:
        """Generates realistic responses or tool calls based on user message content."""
        # Record what the harness asked for so tests can assert the routing
        # decision reached the provider boundary even in mock mode.
        self.last_call = {"model": model, "tools": [t.get("function", {}).get("name")
                                                    for t in (tools or [])],
                          "tier": kwargs.get("tier"),
                          "reasoning_effort": kwargs.get("reasoning_effort"),
                          "reasoning_budget_tokens": kwargs.get("reasoning_budget_tokens")}
        # If last message is a tool result, synthesize completion
        if messages and messages[-1].get("role") == "tool":
            tool_content = str(messages[-1].get("content", ""))
            return LLMResponse(
                content=f"Operation completed. Here is the tool result summary:\n{tool_content[:400]}",
                tool_calls=[],
                model=model or self.default_model,
            )

        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                candidate = str(m.get("content", ""))
                if not candidate.startswith(("The task requires real file changes.",
                                             "The task is not complete.",
                                             "Continue your previous response")):
                    last_user_msg = candidate
                    break

        low = last_user_msg.lower()

        # The mock is a test fixture, not a code generator. A generic sample
        # file cannot satisfy an implementation request, so never claim it did.
        simple_write = re.fullmatch(r"\s*write file ([\w./-]+\.(?:py|md|toml|txt|json))\s*", last_user_msg, re.I)
        implementation_request = (re.search(r"\b(?:create|build|implement|make|edit|write|fix|refactor)\b", low)
            and re.search(r"\b(?:app|code|file|folder|project|module|function|website|script|tests?|bug)\b|"
                          r"\b[\w./-]+\.(?:py|js|ts|html|css|json|md|toml)\b", low))
        if implementation_request and not simple_write:
            return LLMResponse(
                content="Offline mock cannot implement this edit. Configure a live or local model and retry.",
                model=model or self.default_model,
                finish_reason="error",
            )

        # If user asks to run tests or pytest
        if "test" in low or "pytest" in low:
            return LLMResponse(
                content="I'll run the test suite to verify our code.",
                tool_calls=[
                    ToolCall(
                        id="call_mock_1",
                        name="run_bash",
                        arguments={"command": "pytest tests/ -q"},
                    )
                ],
                model=model or self.default_model,
            )

        # If user asks to read or view a file
        if "read" in low or "view" in low or "cat" in low or "inspect" in low:
            file_match = re.search(r"(\S+\.(?:py|md|toml|txt|json))", last_user_msg)
            path = file_match.group(1) if file_match else "README.md"
            return LLMResponse(
                content=f"Let me inspect the contents of `{path}`.",
                tool_calls=[
                    ToolCall(
                        id="call_mock_2",
                        name="read_file",
                        arguments={"path": path},
                    )
                ],
                model=model or self.default_model,
            )

        # If user asks to list files
        if "list" in low or "ls" in low or "directory" in low:
            return LLMResponse(
                content="I will list the current workspace files.",
                tool_calls=[
                    ToolCall(
                        id="call_mock_3",
                        name="list_directory",
                        arguments={"path": "."},
                    )
                ],
                model=model or self.default_model,
            )

        # If user asks to edit or write a file
        if simple_write:
            file_match = re.search(r"(\S+\.(?:py|md|toml|txt|json))", last_user_msg)
            path = file_match.group(1)
            return LLMResponse(
                content=f"Writing implementation to `{path}`.",
                tool_calls=[
                    ToolCall(
                        id="call_mock_4",
                        name="write_file",
                        arguments={
                            "path": path,
                            "content": "# Generated by Adaptive Agent\nprint('Agent execution successful!')\n",
                        },
                    )
                ],
                model=model or self.default_model,
            )

        # Default conversational response
        return LLMResponse(
            content=(
                f"I processed your request: '{last_user_msg}'. "
                "I am ready to edit files, execute tests, run bash commands, or answer clarifying questions."
            ),
            tool_calls=[],
            model=model or self.default_model,
        )
