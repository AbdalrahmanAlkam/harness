"""Step-limit policies: classifier-controlled stopping, fixed budgets, and unbounded runs."""

from __future__ import annotations

from pathlib import Path

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.llm.mock_client import LLMResponse, ToolCall
from adaptive_harness.tools.base import Tool, ToolResult


class LoopingClient:
    """Never finishes; repeats the same tool call forever."""

    default_model = "test/model"

    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return LLMResponse(model=self.default_model, tool_calls=[ToolCall(
            id=str(self.calls), name="run_bash", arguments={"command": "echo ok"})])


class CountingClient:
    """Never finishes, but varies each tool call so no loop is detected."""

    default_model = "test/model"

    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return LLMResponse(model=self.default_model, tool_calls=[ToolCall(
            id=str(self.calls), name="run_bash", arguments={"command": f"echo {self.calls}"})])


class EchoTool(Tool):
    name = "run_bash"
    description = "Echo a value"
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}}

    def execute(self, **kwargs):
        return ToolResult(success=True, output="ok")


def test_classifier_policy_stops_circling_loops_with_visible_injections(tmp_path: Path):
    agent = DeveloperAgent(llm_client=LoopingClient(), tools=[EchoTool()],
                           workspace_root=str(tmp_path), forced_mode="coding")
    events = list(agent.run_stream("check status"))
    response = next(event.payload for event in events if event.event_type == "response")
    assert response["stop_reason"] == "classifier_stop"
    assert not response["success"]
    assert response["steps"] == 4

    injections = [event.payload for event in events if event.event_type == "prompt_injection"]
    assert injections and all(item["source"] == "runtime_overseer" for item in injections)
    assert any("STOP-CIRCLING" in item["content"] for item in injections)
    verdicts = [item for item in injections if "TERMINATION VERDICT" in item["content"]]
    assert verdicts and verdicts[-1]["state"] in {"LOOPING_DETECTED", "PROGRESS_STALLED"}

    # The injected prompts reached the model conversation as system messages.
    system_texts = [str(message.get("content")) for message in agent.messages if message.get("role") == "system"]
    assert any("STOP-CIRCLING" in text for text in system_texts)
    assert any("TERMINATION VERDICT" in text for text in system_texts)
    assert "TERMINATION VERDICT" in response["content"]


def test_fixed_policy_restores_hardcoded_budgets(tmp_path: Path):
    agent = DeveloperAgent(llm_client=CountingClient(), tools=[EchoTool()],
                           workspace_root=str(tmp_path), forced_mode="coding",
                           forced_thinking="none", step_policy="fixed")
    events = list(agent.run_stream("check status"))
    policy = next(event.payload for event in events if event.event_type == "step_policy")
    assert policy["policy"] == "fixed"
    assert policy["limit_source"] == "fixed-budget"
    assert policy["max_steps"] == 4  # ThinkingLevel.NONE budget
    response = next(event.payload for event in events if event.event_type == "response")
    assert response["stop_reason"] == "step_limit"
    assert response["steps"] == 4


def test_unbounded_policy_has_no_cap_or_classifier_interventions(tmp_path: Path):
    agent = DeveloperAgent(llm_client=LoopingClient(), tools=[EchoTool()],
                           workspace_root=str(tmp_path), forced_mode="coding",
                           step_policy="unbounded")
    events = list(agent.run_stream("check status", max_steps=4))
    policy = next(event.payload for event in events if event.event_type == "step_policy")
    assert policy["classifier_supervision"] is False
    assert not [event for event in events if event.event_type == "prompt_injection"]
    response = next(event.payload for event in events if event.event_type == "response")
    assert response["stop_reason"] == "step_limit"
    assert response["steps"] == 4


def test_explicit_max_steps_overrides_any_policy(tmp_path: Path):
    agent = DeveloperAgent(llm_client=LoopingClient(), tools=[EchoTool()],
                           workspace_root=str(tmp_path), forced_mode="coding")
    events = list(agent.run_stream("check status", max_steps=2))
    policy = next(event.payload for event in events if event.event_type == "step_policy")
    assert policy["limit_source"] == "explicit"
    assert policy["max_steps"] == 2
    response = next(event.payload for event in events if event.event_type == "response")
    assert response["stop_reason"] == "step_limit"
    assert response["steps"] == 2


def test_ingested_system_prompt_is_visible(tmp_path: Path):
    class TextClient:
        default_model = "test/model"

        def complete(self, **kwargs):
            return LLMResponse(model=self.default_model, content="All good.")

    agent = DeveloperAgent(llm_client=TextClient(), tools=[EchoTool()],
                           workspace_root=str(tmp_path), forced_mode="coding")
    events = list(agent.run_stream("say hello"))
    prompt = next(event.payload for event in events if event.event_type == "system_prompt")
    assert "Workspace:" in prompt["content"]
    assert prompt["ingested"]["mode"] == "coding"
    assert prompt["ingested"]["workspace"] == str(tmp_path)


def test_harness_and_claim_injections_are_visible(tmp_path: Path):
    class ClaimingClient:
        default_model = "test/model"

        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(model=self.default_model, tool_calls=[ToolCall(
                    id="1", name="run_bash", arguments={"command": "echo ok"})])
            if self.calls == 2:
                return LLMResponse(model=self.default_model,
                                   content="Everything is verified and the file exists.")
            return LLMResponse(model=self.default_model, content="Done.")

    class FailingTool(Tool):
        name = "run_bash"
        description = "Always fails"
        parameters = {"type": "object", "properties": {"command": {"type": "string"}}}

        def execute(self, **kwargs):
            return ToolResult(success=False, output="", error="file not found")

    agent = DeveloperAgent(llm_client=ClaimingClient(), tools=[FailingTool()],
                           workspace_root=str(tmp_path), forced_mode="coding")
    events = list(agent.run_stream("check status"))
    sources = {event.payload["source"] for event in events if event.event_type == "prompt_injection"}
    assert "claim_check" in sources
    assert "harness" in sources
