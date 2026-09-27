"""The agent must actually filter, and must never lose evidence doing so."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.classifiers.domain_classifier import DomainMode
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.llm.mock_client import LLMResponse
from adaptive_harness.tools.base import Tool, ToolResult

NOISY = "\n".join(f"test_thing_{i} PASSED" for i in range(300))
USEFUL = "Traceback (most recent call last):\nNameError: name 'foo' is not defined"


class NoisyBash(Tool):
    name = "run_bash"
    description = "returns a lot of low-value output"
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}, "required": []}

    def __init__(self, output: str, success: bool = True):
        self.output, self.success = output, success

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=self.success, output=self.output)


class ScriptedClient(LLMClient):
    """Returns tool calls first, then a plain answer; records every request."""

    def __init__(self, tool_name: str | None = None, arguments: dict | None = None):
        super().__init__(force_mock=True)
        self.tool_name, self.arguments = tool_name, arguments or {}
        self.turn = 0
        self.summarize_calls: list[list[dict]] = []

    def complete(self, **kwargs):
        messages = list(kwargs.get("messages") or [])
        if any("compressing the output" in str(m.get("content", "")) for m in messages):
            self.summarize_calls.append(messages)
            return LLMResponse(content="SUMMARY: 300 passing tests, nothing failed.")
        self.turn += 1
        if self.turn == 1 and self.tool_name:
            return LLMResponse(content="", tool_calls=[{"id": "c1", "name": self.tool_name,
                                                        "arguments": self.arguments}])
        return LLMResponse(content="Done.")


def _agent(tmp_path, client, tool, **kwargs) -> DeveloperAgent:
    return DeveloperAgent(llm_client=client, tools=[tool], workspace_root=str(tmp_path),
                          preferences_dir=tmp_path / "prefs", forced_mode="coding", **kwargs)


def _last_tool_message(agent: DeveloperAgent) -> dict:
    return next(m for m in reversed(agent.messages) if m.get("role") == "tool")


def test_noisy_output_is_filtered_and_the_run_still_works(tmp_path: Path):
    client = ScriptedClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(NOISY))
    events = list(agent.run_stream("run the tests"))

    filtered = [e for e in events if e.event_type == "output_filtered"]
    assert filtered, "the gate should have fired on pure chatter"
    payload = filtered[0].payload
    assert payload["tool"] == "run_bash"
    assert payload["token"].startswith("OUTPUT-")
    assert payload["raw_chars"] > payload["kept_chars"]

    content = _last_tool_message(agent)["content"]
    assert "SUMMARY" in content
    assert payload["token"] in content
    # The transcript really did shrink.
    assert len(content) < len(NOISY)


def test_the_model_can_recover_the_full_output_with_the_token(tmp_path: Path):
    client = ScriptedClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(NOISY))
    list(agent.run_stream("run the tests"))

    # Recover using the token the transcript itself carries.
    content = _last_tool_message(agent)["content"]
    match = re.search(r"OUTPUT-\d+", content)
    assert match, content
    token = match.group(0)
    assert token == "OUTPUT-1"

    result = agent.tools["read_full_output"].execute(token=token)
    assert result.success
    assert NOISY in result.output


def test_useful_output_is_never_filtered(tmp_path: Path):
    client = ScriptedClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(USEFUL * 30))
    events = list(agent.run_stream("run the tests"))
    assert not [e for e in events if e.event_type == "output_filtered"]
    assert USEFUL * 30 in _last_tool_message(agent)["content"]


def test_a_failing_tool_is_never_filtered_even_when_long_and_noisy(tmp_path: Path):
    client = ScriptedClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(NOISY, success=False))
    events = list(agent.run_stream("run the tests"))
    assert not [e for e in events if e.event_type == "output_filtered"]
    assert NOISY in _last_tool_message(agent)["content"]


def test_a_configured_secondary_model_is_used_for_summarising(tmp_path: Path):
    client = ScriptedClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(NOISY), secondary_model="cheap/small-model")
    # A distinct model must not be summarised by the primary client.
    secondary = agent._secondary_llm_client()
    assert secondary is not client
    assert secondary.default_model == "cheap/small-model"
    # The primary client is untouched by the compression step.
    list(agent.run_stream("run the tests"))
    assert not client.summarize_calls


def test_the_summarise_instruction_names_the_tool_and_comes_from_the_registry(tmp_path: Path):
    client = ScriptedClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(NOISY))
    instruction = agent.prompts.get("tool_filter.summarize", tool="run_bash")
    assert "run_bash" in instruction
    assert "compressed" in instruction.lower()


def test_an_unconfigured_secondary_model_falls_back_to_the_default(tmp_path: Path):
    client = ScriptedClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(NOISY))
    assert agent.secondary_model is None
    assert agent._secondary_llm_client().default_model == client.default_model
    list(agent.run_stream("run the tests"))
    assert client.summarize_calls, "filtering still happens with the default model"


def test_the_filter_can_be_switched_off(tmp_path: Path):
    client = ScriptedClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(NOISY), filter_tool_output=False)
    assert agent.tool_output_filter is None
    assert "read_full_output" not in agent.tools
    events = list(agent.run_stream("run the tests"))
    assert not [e for e in events if e.event_type == "output_filtered"]
    assert NOISY in _last_tool_message(agent)["content"]


def test_the_recall_tool_cannot_be_filtered_away(tmp_path: Path):
    """It must stay reachable, or the token is a dead end."""
    client = ScriptedClient()
    agent = _agent(tmp_path, client, NoisyBash(NOISY))
    for mode in (DomainMode.CODING, DomainMode.RESEARCH, DomainMode.SCIENCE, DomainMode.AUDIT):
        agent.forced_mode = mode
        list(agent.run_stream("hello"))
    assert "read_full_output" in agent.tools


def test_a_broken_secondary_model_does_not_break_the_run(tmp_path: Path):
    class BrokenClient(ScriptedClient):
        def complete(self, **kwargs):
            messages = list(kwargs.get("messages") or [])
            if any("compressing the output" in str(m.get("content", "")) for m in messages):
                raise RuntimeError("secondary model is down")
            return super().complete(**kwargs)

    client = BrokenClient("run_bash", {"command": "pytest"})
    agent = _agent(tmp_path, client, NoisyBash(NOISY))
    events = list(agent.run_stream("run the tests"))
    # The gate still ran, but the raw text survived the summariser's failure.
    assert "SUMMARY" not in _last_tool_message(agent)["content"]
    assert not [e for e in events if e.event_type == "output_filtered"]
