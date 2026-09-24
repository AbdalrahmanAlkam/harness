"""Behavioral checks for persistent terminal preferences and keyboard workflows."""

import json
from pathlib import Path

import pytest
from rich.console import Console
from textual.widgets import Footer, Input

from adaptive_harness.data.config import ConfigManager, PromptHistoryStore
from adaptive_harness.agent.agent import AgentEvent, DeveloperAgent
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.widgets import (ClassifierTelemetryWidget, HistoryInput, ThemePickerModal,
                                          QuickSelectModal, OutputViewerModal)
from adaptive_harness.tui.formatting import format_model_markdown
from adaptive_harness.llm.catalog import CatalogModel, fetch_models
from adaptive_harness.llm.client import MODEL_TIERS
from adaptive_harness.data.sessions import SessionStore


def test_private_config_and_bounded_history(tmp_path: Path):
    config = ConfigManager(tmp_path / "preferences")
    config.save_key("secret-key")
    config.save_theme("nord")
    assert config.load() == {"api_key": "secret-key", "theme": "nord"}
    assert config.path.stat().st_mode & 0o777 == 0o600
    assert config.directory.stat().st_mode & 0o777 == 0o700
    config.clear_key()
    assert config.load() == {"theme": "nord"}
    history = PromptHistoryStore(tmp_path / "preferences")
    assert history.record("first")
    assert not history.record("first")
    assert not history.record("/key secret-key")
    for number in range(501):
        history.record(f"prompt {number}")
    assert len(PromptHistoryStore(tmp_path / "preferences").entries) == 500
    assert "secret-key" not in history.path.read_text()
    assert history.path.stat().st_mode & 0o777 == 0o600


def test_credential_precedence_and_clear(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "preferences"
    ConfigManager(config_dir).save_key("saved-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")
    app = AdaptiveHarnessApp(api_key="flag-key", db_path=tmp_path / "flag.db", config_dir=config_dir)
    assert (app.api_key, app.key_source) == ("flag-key", "CLI flag")
    app.session_store.close()
    app = AdaptiveHarnessApp(db_path=tmp_path / "env.db", config_dir=config_dir)
    assert (app.api_key, app.key_source) == ("environment-key", "environment")
    app.session_store.close()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    app = AdaptiveHarnessApp(db_path=tmp_path / "saved.db", config_dir=config_dir)
    assert (app.api_key, app.key_source) == ("saved-key", "saved configuration")
    app.session_store.close()


def test_telemetry_renders_full_labels_without_truncation():
    telemetry = ClassifierTelemetryWidget()
    console = Console(width=48, color_system=None)
    with console.capture() as capture:
        console.print(telemetry.render())
    output = capture.get()
    skill_rows = [line for line in output.splitlines() if "░" in line]
    assert len(skill_rows) == 6
    assert all("…" not in line for line in skill_rows)
    assert "Search & Explore" in output
    assert "Clarification" in output
    assert all("0.0%" in line for line in skill_rows)


def test_agent_reports_token_usage_for_tui(tmp_path: Path):
    agent = DeveloperAgent(llm_client=LLMClient(force_mock=True), workspace_root=str(tmp_path))
    response = next(event.payload for event in agent.run_stream("say hello", max_steps=1)
                    if event.event_type == "response")
    assert response["usage"] == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                                 "reasoning_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0}


def test_model_catalog_parses_text_models(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self):
            return json.dumps({"data": [
                {"id": "z-ai/glm-5.3-flash", "name": "GLM 5.3 Flash", "context_length": 1000000},
                {"id": "image/only", "architecture": {"output_modalities": ["image"]}},
            ]}).encode()
    monkeypatch.setattr("adaptive_harness.llm.catalog.urlopen", lambda request, timeout: Response())
    assert fetch_models() == [CatalogModel("z-ai/glm-5.3-flash", "GLM 5.3 Flash", 1000000)]
    assert MODEL_TIERS["standard"] == "z-ai/glm-5.3-flash"


def test_launch_model_override_wins_saved_session(tmp_path: Path):
    database = tmp_path / "sessions.db"
    store = SessionStore(database)
    saved = store.create(str(tmp_path))
    saved.model = "saved/model"
    saved.settings = {"safety": "cautious"}
    store.save(saved)
    store.close()
    app = AdaptiveHarnessApp(db_path=database, session_id=saved.id,
                             default_model="cli/model", config_dir=tmp_path / "prefs")
    assert app.agent.explicit_model == "cli/model"
    assert app.agent.llm_client.default_model == "cli/model"
    assert app.agent.safety_profile == "cautious"
    app.session_store.close()


@pytest.mark.anyio
async def test_quick_picker_model_and_session_keyboard(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("adaptive_harness.tui.app.fetch_models", lambda key: [
        CatalogModel("z-ai/glm-5.3-flash", "GLM 5.3 Flash", 1000000),
        CatalogModel("openai/test-model", "Test Model", 10000),
    ])
    app = AdaptiveHarnessApp(db_path=tmp_path / "picker.db", workspace_root=str(tmp_path),
                             config_dir=tmp_path / "prefs")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("f4")
        for _ in range(5):
            await pilot.pause()
            if app.screen.query("#quick-search"):
                break
        assert isinstance(app.screen, QuickSelectModal)
        task_input = app.query_one("#prompt-input", HistoryInput)
        task_input.value = "keep this prompt draft"
        prior_messages = list(app.agent.messages)
        search = app.screen.query_one("#quick-search", Input)
        search.value = "test-model"
        await pilot.pause()
        assert len(app.screen.visible_choices) == 1
        await pilot.press("enter")
        await pilot.pause()
        assert app.agent.explicit_model == "openai/test-model"
        assert app.agent.messages == prior_messages
        assert task_input.value == "keep this prompt draft"
        app._reset_session(new=True, title="Other task")
        await pilot.press("f5")
        await pilot.pause()
        assert isinstance(app.screen, QuickSelectModal)
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.session.title != "Other task"


@pytest.mark.anyio
async def test_activity_and_safety_controls(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "activity.db", workspace_root=str(tmp_path),
                             config_dir=tmp_path / "prefs")
    async with app.run_test(size=(120, 30)):
        app._render_event(AgentEvent(
            "agent_stage", {"stage": "thinking", "step": 1, "model": "test", "thinking_tokens": 1000}))
        assert "Thinking" in app._activity and "1,000 budget tokens" in app._activity
        app._render_event(AgentEvent(
            "agent_stage", {"stage": "tool_running", "step": 1, "tool": "run_bash"}))
        assert "Running run_bash" in app._activity
        app._handle_slash_command("/safety cautious")
        assert app.agent.safety_profile == "cautious"
        assert app.session.settings["safety"] == "cautious"


def test_math_and_channel_markers_render_readably():
    rendered = format_model_markdown(r"Answer: $x^2 + \frac{1}{2} = \sqrt{4}$ [analysis]")
    assert "x²" in rendered and "(1)/(2)" in rendered and "√(4)" in rendered
    assert "[analysis]" not in rendered


@pytest.mark.anyio
async def test_output_popup_is_selectable_and_usage_persists(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "output.db", workspace_root=str(tmp_path),
                             config_dir=tmp_path / "prefs")
    async with app.run_test(size=(100, 28)) as pilot:
        copied = []
        app.copy_to_clipboard = copied.append
        app._last_agent_content = "Result: x² = 4"
        app._handle_slash_command("/output")
        await pilot.pause()
        assert isinstance(app.screen, OutputViewerModal)
        assert app.screen.query_one("#output-text").text == "Result: x² = 4"
        await pilot.press("ctrl+shift+c")
        assert copied == ["Result: x² = 4"]
        app.screen.dismiss(None)
        await pilot.pause()
        app._last_tool_output = "Full tool trace"
        app._handle_slash_command("/tool-output")
        await pilot.pause()
        assert app.screen.query_one("#output-text").text == "Full tool trace"
        app.screen.dismiss(None)
        app.prompt_tokens, app.completion_tokens = 300, 80
        app._reasoning_tokens, app._cached_tokens, app._cache_write_tokens = 24, 90, 12
        app._reported_cost_usd, app._cost_reported = 0.00125, True
        app.agent.messages.append({"role": "assistant", "content": "Saved answer"})
        app._save_session()
        saved_id = app.session.id
    restored = AdaptiveHarnessApp(db_path=tmp_path / "output.db", workspace_root=str(tmp_path),
                                  config_dir=tmp_path / "prefs", session_id=saved_id)
    assert restored._usage_snapshot() == {
        "total_tokens": 380, "input_tokens": 300, "output_tokens": 80, "reasoning_tokens": 24,
        "cache_read_tokens": 90, "cache_write_tokens": 12, "cost_usd": 0.00125,
        "cost_source": "provider-reported"}
    assert restored._last_agent_content == "Saved answer"
    restored.session_store.close()


@pytest.mark.anyio
async def test_tui_history_theme_reset_and_export(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = AdaptiveHarnessApp(db_path=tmp_path / "ui.db", workspace_root=str(tmp_path),
                             config_dir=tmp_path / "preferences")
    async with app.run_test(size=(120, 30)) as pilot:
        app._handle_slash_command("/key secret-for-test")
        assert app.config.path.stat().st_mode & 0o777 == 0o600
        app._handle_slash_command("/key clear")
        assert app.agent.llm_client.is_mock
        assert "api_key" not in app.config.load()

        prompt = app.query_one("#prompt-input", HistoryInput)
        app.history_store.record("first prompt")
        app.history_store.record("second prompt")
        prompt.history = list(app.history_store.entries)
        prompt.value = "draft"
        prompt.focus()
        await pilot.press("up")
        assert prompt.value == "second prompt"
        await pilot.press("up")
        assert prompt.value == "first prompt"
        await pilot.press("down", "down")
        assert prompt.value == "draft"

        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, ThemePickerModal)
        await pilot.press("down")
        assert app.theme == "nord"
        await pilot.press("escape")
        await pilot.pause()
        assert app.theme == "textual-dark"
        await pilot.press("f2", "down", "enter")
        await pilot.pause()
        assert app.theme == "nord"
        assert app.config.load()["theme"] == "nord"

        app.agent.messages.append({"role": "user", "content": "inspect project"})
        app.agent.messages.append({"role": "assistant", "content": "done", "tool_calls": [
            {"function": {"name": "run_bash", "arguments": '{"command":"pwd"}'}}]})
        app.agent.messages.append({"role": "tool", "content": "project directory"})
        app._handle_slash_command("/export json")
        exports = list((tmp_path / "output" / "sessions").glob("*.json"))
        assert len(exports) == 1
        assert json.loads(exports[0].read_text())["messages"][-1]["content"] == "project directory"
        app._handle_slash_command("/export markdown")
        markdown = next((tmp_path / "output" / "sessions").glob("*.md")).read_text()
        assert "run_bash" in markdown
        assert "project directory" in markdown

        telemetry = app.query_one("#telemetry", ClassifierTelemetryWidget)
        telemetry.update_telemetry(probabilities={"code_edit": 1.0}, entropy=1.5,
                                   margin=0.8, risk_level="high")
        old_id = app.session.id
        app._handle_slash_command("/new")
        assert app.session.id != old_id
        assert len(app.agent.messages) == 1
        assert all(probability == 0 for probability in telemetry.probabilities.values())
        assert (telemetry.entropy, telemetry.margin, telemetry.risk_level) == (0, 0, "idle")
        app.agent.messages.append({"role": "user", "content": "new task"})
        app._handle_slash_command("/reset")
        assert len(app.agent.messages) == 1
        assert app.session.id != old_id

    reopened = AdaptiveHarnessApp(db_path=tmp_path / "next.db", config_dir=tmp_path / "preferences")
    assert reopened.saved_theme == "nord"
    assert reopened.api_key is None
    reopened.session_store.close()


@pytest.mark.anyio
async def test_footer_clearance_and_mode_controls(tmp_path: Path):
    db_path = tmp_path / "layout.db"
    app = AdaptiveHarnessApp(db_path=db_path, config_dir=tmp_path / "preferences",
                             workspace_root=str(tmp_path))
    async with app.run_test(size=(120, 30)) as pilot:
        prompt = app.query_one("#prompt-input", HistoryInput)
        footer = app.query_one(Footer)
        for width, height in ((120, 30), (70, 20), (50, 14)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            assert prompt.region.y + prompt.region.height < footer.region.y
            prompt.value = "/"
            await pilot.pause()
            assert prompt.region.y + prompt.region.height < footer.region.y
            app.query_one("#waiting-indicator").add_class("visible")
            await pilot.pause()
            assert prompt.region.y + prompt.region.height < footer.region.y
            app.query_one("#waiting-indicator").remove_class("visible")
            prompt.value = ""

        app._handle_slash_command("/mode security")
        app._handle_slash_command("/thinking deep")
        telemetry = app.query_one("#telemetry", ClassifierTelemetryWidget)
        assert app.agent.forced_mode.value == "audit"
        assert app.agent.forced_thinking.value == "deep"
        assert telemetry.domain_selection == telemetry.thinking_selection == "forced"
        assert telemetry.thinking_tokens == 16000
        assert "SECURITY" in str(app.query_one("#status-line").render())
        app._handle_slash_command("/mode research")
        session_id = app.session.id
        assert app.session_store.load(session_id).settings["mode"] == "research"
        assert app.session_store.load(session_id).settings["thinking"] == "deep"
        app._handle_slash_command("/thinking auto")
        assert app.agent.forced_thinking is None
        app._handle_slash_command("/thinking deep")

    restored = AdaptiveHarnessApp(db_path=db_path, config_dir=tmp_path / "preferences", session_id=session_id)
    assert restored.agent.forced_mode.value == "research"
    assert restored.agent.forced_thinking.value == "deep"
    restored.session_store.close()
