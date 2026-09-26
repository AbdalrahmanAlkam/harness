"""Model-specific reasoning effort selection and OpenRouter payloads."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from adaptive_harness.cli import app as cli_app
from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.classifiers.thinking_classifier import ThinkingLevel, parse_thinking_level
from adaptive_harness.llm.client import LLMClient, MODEL_TIERS
from adaptive_harness.llm.mock_client import LLMResponse
from adaptive_harness.llm.effort import EFFORTS, supported_efforts
from adaptive_harness.llm.providers import PROVIDERS, context_window
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.widgets import QuickSelectModal


def test_space_bunny_is_standard_and_exposes_all_efforts():
    model = "stealth/space-bunny-alpha"
    assert MODEL_TIERS["standard"] == PROVIDERS["openrouter"].default_model == model
    assert supported_efforts(model) == EFFORTS
    assert context_window(model) == 1_000_000
    assert parse_thinking_level("high") is ThinkingLevel.HIGH
    assert parse_thinking_level("xhigh") is ThinkingLevel.XHIGH
    assert parse_thinking_level("max") is ThinkingLevel.MAX
    assert parse_thinking_level("deep") is ThinkingLevel.DEEP
    assert supported_efforts("openai/gpt-4o") == ()


def test_cli_rejects_effort_unavailable_to_selected_model(tmp_path):
    result = CliRunner().invoke(cli_app, ["dev", "list files", "--offline", "--workspace", str(tmp_path),
                                          "--db", str(tmp_path / "run.db"),
                                          "--model", "openai/gpt-4o", "--thinking", "max"])
    assert result.exit_code != 0
    assert "openai/gpt-4o supports reasoning efforts" in result.output


def test_agent_keeps_max_effort_on_manual_model(tmp_path):
    class Client:
        default_model = "stealth/space-bunny-alpha"
        provider = "openrouter"
        def __init__(self):
            self.requests = []
        def complete(self, **kwargs):
            self.requests.append(kwargs)
            return LLMResponse(content="Listed files.")

    client = Client()
    agent = DeveloperAgent(llm_client=client, workspace_root=str(tmp_path),
                           forced_thinking="max", preferences_dir=tmp_path / "prefs")
    events = list(agent.run_stream("list files", max_steps=1))
    thinking = next(item.payload for item in events if item.event_type == "thinking_budget")
    routing = next(item.payload for item in events if item.event_type == "model_routing")
    assert thinking["level"] == "max" and thinking["tokens"] == 32000
    assert routing["model"] == "stealth/space-bunny-alpha"
    assert client.requests[0]["reasoning_effort"] == "max"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_space_bunny_effort_reaches_openrouter(effort):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="ok", tool_calls=None), finish_reason="stop")], usage=None, model=kwargs["model"])

    client = LLMClient(api_key="test-key")
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = client.complete([{"role": "user", "content": "hello"}],
        reasoning_effort=effort, reasoning_budget_tokens=4000)
    assert response.content == "ok"
    assert requests[-1]["model"] == "stealth/space-bunny-alpha"
    assert requests[-1]["extra_body"]["reasoning"] == {"effort": effort}
    assert "temperature" not in requests[-1]


@pytest.mark.anyio
async def test_tui_effort_options_follow_selected_model(tmp_path):
    app = AdaptiveHarnessApp(workspace_root=str(tmp_path), db_path=tmp_path / "ui.db",
                             config_dir=tmp_path / "prefs")
    async with app.run_test() as pilot:
        app._handle_slash_command("/thinking")
        await pilot.pause()
        assert isinstance(app.screen, QuickSelectModal)
        assert [choice[0] for choice in app.screen.choices] == ["auto", *EFFORTS]
        app.pop_screen()
        await pilot.pause()
        app._handle_slash_command("/thinking xhigh")
        assert app.agent.forced_thinking is ThinkingLevel.XHIGH
        app._set_model("openai/gpt-4o")
        assert app.agent.forced_thinking is None
        app._handle_slash_command("/thinking max")
        assert app.agent.forced_thinking is None
        app._set_model("stealth/space-bunny-alpha")
        app._handle_slash_command("/thinking max")
        assert app.agent.forced_thinking is ThinkingLevel.MAX
