"""Runtime supervision, conditional context, and provider routing regressions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import pytest

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.agent.context_window import estimate_tokens, prepare_context
from adaptive_harness.classifiers.runtime_overseer import RuntimeOverseer, OverseerState
from adaptive_harness.data.credentials import CredentialsManager
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.llm.mock_client import LLMResponse, ToolCall
from adaptive_harness.tools.base import Tool, ToolResult


def test_overseer_detects_loop_stall_and_false_claim():
    loop = RuntimeOverseer("fix the parser")
    for _ in range(2):
        assert loop.observe("edit_file", {"path": "parser.py", "old": "a", "new": "b"},
                            success=True).state == OverseerState.HEALTHY_PROGRESS
    decision = loop.observe("edit_file", {"path": "parser.py", "old": "a", "new": "b"}, success=True)
    assert decision.state == OverseerState.LOOPING_DETECTED
    assert "OVERSEER INTERVENTION" in decision.directive

    stalled = RuntimeOverseer("fix tests")
    for _ in range(3):
        decision = stalled.observe("run_pytest", {"test_path": "tests"}, success=False,
                                   error="3 tests failed")
    assert decision.state == OverseerState.PROGRESS_STALLED

    improving = RuntimeOverseer("fix tests")
    for index, count in enumerate((3, 2, 1)):
        decision = improving.observe("run_pytest", {"test_path": f"tests/part{index}"},
                                     success=False, output=f"{count} failed")
    assert decision.state == OverseerState.HEALTHY_PROGRESS

    claim = RuntimeOverseer("inspect missing symbol")
    claim.observe("search_files", {"pattern": "missing_symbol"}, success=False,
                  error="symbol not found")
    decision = claim.check_claim("The file exists and everything is verified.")
    assert decision and decision.state == OverseerState.HALLUCINATION_DETECTED
    immediate = RuntimeOverseer("find missing symbol")
    decision = immediate.observe("search_files", {"pattern": "missing_symbol"}, success=False,
                                 error="symbol not found", model_text="I confirmed it exists")
    assert decision.state == OverseerState.HALLUCINATION_DETECTED

    drift = RuntimeOverseer("Only edit parser.py")
    decision = drift.observe("edit_file", {"path": "payments.py"}, success=True)
    assert decision.state == OverseerState.SEMANTIC_DRIFT


def test_agent_injects_overseer_directive_after_repeated_tool_calls(tmp_path: Path):
    class RepeatingClient:
        default_model = "test/model"

        def __init__(self):
            self.calls = 0
            self.seen = []

        def complete(self, **kwargs):
            self.calls += 1
            self.seen.append(copy.deepcopy(kwargs["messages"]))
            if self.calls <= 3:
                return LLMResponse(model=self.default_model, tool_calls=[ToolCall(
                    id=str(self.calls), name="run_bash", arguments={"command": "echo ok"})])
            return LLMResponse(model=self.default_model, content="Finished after a different strategy")

    class RepeatTool(Tool):
        name = "run_bash"
        description = "Repeat a harmless check"
        parameters = {"type": "object", "properties": {"command": {"type": "string"}}}

        def execute(self, **kwargs):
            return ToolResult(success=True, output="ok")

    client = RepeatingClient()
    agent = DeveloperAgent(llm_client=client, tools=[RepeatTool()], workspace_root=str(tmp_path),
                           forced_mode="coding")
    events = list(agent.run_stream("check status", max_steps=4))
    assert any(event.event_type == "overseer" and event.payload["state"] == "LOOPING_DETECTED"
               for event in events)
    assert any("OVERSEER INTERVENTION" in str(item.get("content")) for item in client.seen[3])


def test_context_compacts_strictly_above_threshold_and_keeps_source():
    small = [{"role": "system", "content": "Guardrail"}, {"role": "user", "content": "Hello"}]
    prepared, info = prepare_context(small, "local-model", limit=1000)
    assert prepared is small and not info["compacted"]
    messages = [{"role": "system", "content": "Guardrail"},
                {"role": "user", "content": "old request"},
                {"role": "assistant", "tool_calls": [{"id": "1"}], "content": ""},
                {"role": "tool", "name": "run_bash", "tool_call_id": "1", "content": "verbose\n" * 900},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new request"},
                {"role": "assistant", "content": "recent answer"},
                *({"role": "user", "content": "more context"},
                  {"role": "assistant", "content": "more answer"}) * 4]
    original = copy.deepcopy(messages)
    boundary = (estimate_tokens(messages) * 4 + 2) // 3
    unchanged, boundary_info = prepare_context(messages, "local-model", limit=boundary)
    assert unchanged is messages and boundary_info["utilization"] <= 0.75
    prepared, info = prepare_context(messages, "local-model", limit=1200)
    assert info["compacted"] and info["compacted_tokens"] > 0
    assert prepared[0] == messages[0]
    assert messages == original

    intervention = {"role": "system", "content": "OVERSEER INTERVENTION: inspect again"}
    diff = {"role": "tool", "name": "edit_file", "content": "Diff:\n--- a/a.py\n+++ b/a.py\n" + "+x\n" * 300}
    with_diff = [messages[0], intervention, *messages[1:], diff]
    copied, _ = prepare_context(with_diff, "local-model", limit=1200)
    assert intervention in copied and diff in copied


def test_private_provider_credentials_and_direct_dispatch(tmp_path: Path):
    store = CredentialsManager(tmp_path)
    store.set("deepseek", "secret-deepseek-key")
    assert store.load() == {"deepseek": "secret-deepseek-key"}
    assert store.path.stat().st_mode & 0o777 == 0o600
    client = LLMClient(provider="deepseek", provider_keys=store.load())
    assert client.base_url == "https://api.deepseek.com"
    assert client.api_key == "secret-deepseek-key"
    assert client.get_model_for_tier("reasoning") == "deepseek-reasoner"
    store.clear("deepseek")
    assert store.load() == {}


@pytest.mark.parametrize("provider,expected_url", [
    ("anthropic", "https://api.anthropic.com/v1"),
    ("openai", "https://api.openai.com/v1"),
    ("google", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("groq", "https://api.groq.com/openai/v1"),
    ("local", "http://localhost:11434/v1"),
])
def test_provider_endpoint_dispatch(provider: str, expected_url: str):
    client = LLMClient(provider=provider, provider_keys={provider: "test-key"})
    assert client.base_url == expected_url
    assert client.get_model_for_tier("standard")
    if provider == "local":
        assert client.api_key == "test-key"
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Direct response", tool_calls=None), finish_reason="stop")],
            usage=None, model=kwargs["model"])

    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = client.complete([{"role": "user", "content": "hi"}], tier="standard")
    assert response.content == "Direct response"
    assert requests[0]["model"] == client.get_model_for_tier("standard")


def test_overseer_and_context_are_visible_in_telemetry():
    from rich.console import Console
    from adaptive_harness.tui.widgets import ClassifierTelemetryWidget

    widget = ClassifierTelemetryWidget()
    widget.update_telemetry(overseer_state="LOOPING_DETECTED", overseer_latency_ms=1.2,
                            overseer_tier="semantic:onnx", context_used=90_000,
                            context_capacity=128_000, context_compacted=42_000)
    console = Console(width=60, color_system=None)
    with console.capture() as captured:
        console.print(widget.render())
    output = captured.get()
    assert "LOOPING DETECTED" in output
    assert "semantic:onnx" in output
    assert "Context:" in output and "Compacted" in output


@pytest.mark.anyio
async def test_theme_hover_previews_and_escape_restores(tmp_path: Path):
    from adaptive_harness.tui.app import AdaptiveHarnessApp
    from adaptive_harness.tui.widgets import ThemePickerModal

    app = AdaptiveHarnessApp(db_path=tmp_path / "theme.db", config_dir=tmp_path / "prefs")
    async with app.run_test(size=(110, 29)) as pilot:
        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, ThemePickerModal)
        await pilot.hover("#theme-2")
        await pilot.pause()
        assert app.theme == "tokyo-night"
        await pilot.press("escape")
        await pilot.pause()
        assert app.theme == "textual-dark"


def test_provider_failover_on_rate_limit(monkeypatch):
    class RateLimited(Exception):
        status_code = 429

    calls = []

    def fake_openai(*, base_url, **kwargs):
        def create(**request):
            calls.append((base_url, request["model"]))
            if "deepseek" in base_url:
                raise RateLimited("rate limited")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="Recovered", tool_calls=None), finish_reason="stop")],
                usage=None, model=request["model"])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("adaptive_harness.llm.client.OpenAI", fake_openai)
    client = LLMClient(provider="deepseek", provider_keys={"deepseek": "one", "openrouter": "two"},
                       backup_providers=("openrouter",))
    response = client.complete([{"role": "user", "content": "hello"}], model="deepseek-chat")
    assert response.content == "Recovered"
    assert response.metadata["failed_over_from"] == "deepseek"
    assert client.provider == "openrouter"
    assert calls == [("https://api.deepseek.com", "deepseek-chat"),
                     ("https://openrouter.ai/api/v1", "stealth/space-bunny-alpha")]


def test_failed_backup_restores_primary_provider(monkeypatch):
    class ServerError(Exception):
        status_code = 503

    def fake_openai(**kwargs):
        def create(**request):
            raise ServerError("unavailable")
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("adaptive_harness.llm.client.OpenAI", fake_openai)
    client = LLMClient(provider="deepseek", provider_keys={"deepseek": "one", "openrouter": "two"},
                       backup_providers=("openrouter",))
    response = client.complete([{"role": "user", "content": "hello"}])
    assert response.finish_reason == "error"
    assert client.provider == "deepseek"
    assert client.api_key == "one"


def test_direct_anthropic_thinking_does_not_replay_unsigned_tool_history():
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Grounded answer", tool_calls=None), finish_reason="stop")],
            usage=None, model=kwargs["model"])

    client = LLMClient(provider="anthropic", api_key="test-key")
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    messages = [{"role": "user", "content": "Check"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                  "function": {"name": "read_file", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "1", "content": "Result"}]
    response = client.complete(messages, model="claude-sonnet-4-6", reasoning_effort="high",
                               reasoning_budget_tokens=4000)
    assert "thinking" not in requests[0].get("extra_body", {})
    assert response.metadata["thinking_fallback"] is True


@pytest.mark.anyio
async def test_provider_switch_and_key_persist_in_tui(tmp_path: Path, monkeypatch):
    from adaptive_harness.tui.app import AdaptiveHarnessApp

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AdaptiveHarnessApp(db_path=tmp_path / "providers.db", config_dir=tmp_path / "prefs")
    async with app.run_test(size=(110, 29)):
        app._handle_slash_command("/key anthropic test-secret-anthropic")
        assert CredentialsManager(tmp_path / "prefs").load()["anthropic"] == "test-secret-anthropic"
        app._handle_slash_command("/provider anthropic")
        assert app.agent.llm_client.provider == "anthropic"
        assert app.agent.llm_client.api_key == "test-secret-anthropic"
        saved_id = app.session.id
    reopened = AdaptiveHarnessApp(db_path=tmp_path / "providers.db", config_dir=tmp_path / "prefs",
                                  session_id=saved_id)
    assert reopened.agent.llm_client.provider == "anthropic"
    reopened.session_store.close()
