"""Behavioral checks for persistent terminal preferences and keyboard workflows."""

import json
from pathlib import Path

import pytest
from rich.console import Console

from adaptive_harness.data.config import ConfigManager, PromptHistoryStore
from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.widgets import ClassifierTelemetryWidget, HistoryInput, ThemePickerModal


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
    assert response["usage"] == {"prompt_tokens": 100, "completion_tokens": 50}


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
