"""The settings screen must be the one place every setting can be found.

These cover the two halves that are easy to get wrong: that the screen lists
every persistent setting, and that changing one there actually changes the live
agent, the stored session, and the remembered profile.

Anything that touches the app's widgets runs inside ``run_test``, because the
status refresh and the chat log need a mounted screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_harness.data.config import ConfigManager
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.settings import SettingsScreen


def _config_dir(tmp_path: Path) -> Path:
    path = tmp_path / "prefs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _app(tmp_path: Path, name: str = "settings.db", **kwargs) -> AdaptiveHarnessApp:
    return AdaptiveHarnessApp(db_path=tmp_path / name, workspace_root=str(tmp_path),
                              config_dir=_config_dir(tmp_path), **kwargs)


# -- the screen lists everything -------------------------------------------


def test_every_persistent_setting_is_reachable_from_the_screen(tmp_path: Path):
    app = _app(tmp_path)
    keys = {row.key for row in app._setting_rows()}
    for expected in ("model", "secondary_model", "output_filter", "mode", "thinking",
                     "safety", "step_policy", "max_steps", "swarm_mode",
                     "isolation_mode", "provider", "classifier", "theme"):
        assert expected in keys, f"{expected} is not in the settings screen"
    app.session_store.close()


def test_the_screen_shows_live_values_not_stale_copies(tmp_path: Path):
    app = _app(tmp_path)
    values = {row.key: row.value for row in app._setting_rows()}
    assert values["mode"] == "coding"
    assert values["swarm_mode"] == "off"
    assert values["output_filter"] == "on"
    # With no secondary model configured the row says so rather than showing blank.
    assert "same as model" in values["secondary_model"]
    app.session_store.close()


@pytest.mark.anyio
async def test_f7_opens_the_settings_screen(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        assert "Secondary model" in [row.label for row in app._setting_rows()]


@pytest.mark.anyio
async def test_the_slash_command_opens_it_too(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)) as pilot:
        app._handle_slash_command("/settings")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


@pytest.mark.anyio
async def test_escape_closes_the_screen(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SettingsScreen)


@pytest.mark.anyio
async def test_the_screen_renders_a_row_per_setting(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        rows = app.screen.query(".settings-row")
        assert len(rows) == len(app._setting_rows())


# -- cycling a setting changes everything ----------------------------------


@pytest.mark.anyio
async def test_cycling_the_output_filter_reconfigures_the_agent_and_persists(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)):
        assert app.agent.tool_output_filter is not None
        app._apply_setting("output_filter", "off")
        assert app.agent.tool_output_filter is None
        assert "read_full_output" not in app.agent.tools
        assert ConfigManager(_config_dir(tmp_path)).preferences()["output_filter"] == "off"
        # Turning it back on restores the tool.
        app._apply_setting("output_filter", "on")
        assert app.agent.tool_output_filter is not None
        assert "read_full_output" in app.agent.tools


@pytest.mark.anyio
async def test_cycling_a_mode_updates_agent_session_and_profile(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)):
        app._apply_setting("mode", "research")
        assert app.agent.forced_mode.value == "research"
        assert app.session.settings["mode"] == "research"
        assert ConfigManager(_config_dir(tmp_path)).preferences()["mode"] == "research"


@pytest.mark.anyio
async def test_cycling_swarm_registers_and_unregisters_the_tool(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)):
        app._apply_setting("swarm_mode", "on")
        assert "delegate_subagent" in app.agent.tools
        app._apply_setting("swarm_mode", "off")
        assert "delegate_subagent" not in app.agent.tools


# -- the secondary model ---------------------------------------------------


@pytest.mark.anyio
async def test_choosing_a_secondary_model_is_stored_and_applied(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)):
        app._set_secondary_model("cheap/small")
        assert app.secondary_model == "cheap/small"
        assert app.agent.secondary_model == "cheap/small"
        assert app.agent._secondary_llm_client().default_model == "cheap/small"
        stored = ConfigManager(_config_dir(tmp_path)).preferences()
        assert stored["secondary_model"] == "cheap/small"
        assert app.session.settings["secondary_model"] == "cheap/small"


@pytest.mark.anyio
async def test_clearing_the_secondary_model_falls_back_to_the_primary(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)):
        app._set_secondary_model("cheap/small")
        app._set_secondary_model(None)
        assert app.secondary_model is None
        assert app.agent._secondary_llm_client().default_model == app.agent.llm_client.default_model
        # An empty value is stored as absent rather than as an empty string.
        assert "secondary_model" not in ConfigManager(_config_dir(tmp_path)).preferences()


@pytest.mark.anyio
async def test_the_secondary_model_survives_a_restart(tmp_path: Path):
    first = _app(tmp_path)
    async with first.run_test(size=(120, 34)):
        first._set_secondary_model("cheap/small")

    second = _app(tmp_path, name="settings2.db")
    async with second.run_test(size=(120, 34)):
        assert second.secondary_model == "cheap/small"
        assert second.agent.secondary_model == "cheap/small"


def test_a_launch_flag_sets_the_secondary_model(tmp_path: Path):
    app = _app(tmp_path, secondary_model="cheap/flag")
    assert app.secondary_model == "cheap/flag"
    assert app.agent.secondary_model == "cheap/flag"
    app.session_store.close()


def test_the_flag_belongs_to_tui_and_not_to_dev():
    """It was once wired into `dev`, where the consuming call did not exist."""
    import inspect

    from adaptive_harness import cli
    tui_params = inspect.signature(cli.tui).parameters
    dev_params = inspect.signature(cli.dev).parameters
    assert "secondary_model" in tui_params
    assert "secondary_model" not in dev_params


@pytest.mark.anyio
async def test_the_filter_toggle_survives_a_restart(tmp_path: Path):
    first = _app(tmp_path)
    async with first.run_test(size=(120, 34)):
        first._apply_setting("output_filter", "off")

    second = _app(tmp_path, name="settings3.db")
    async with second.run_test(size=(120, 34)):
        assert second.agent.tool_output_filter is None


@pytest.mark.anyio
async def test_the_config_file_holds_the_new_settings_and_stays_private(tmp_path: Path):
    app = _app(tmp_path)
    async with app.run_test(size=(120, 34)):
        app._set_secondary_model("cheap/small")
        app._apply_setting("output_filter", "off")
    path = _config_dir(tmp_path) / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["secondary_model"] == "cheap/small"
    assert data["output_filter"] == "off"
    assert oct(path.stat().st_mode)[-3:] == "600"
