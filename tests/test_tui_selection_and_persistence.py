"""Regressions for chat-log text selection, clipboard delivery, and sticky settings.

These three behaviours are easy to break silently and impossible to notice
without a real terminal, so each is asserted directly:

* ``PinnedRichLog`` must expose its content to Textual's selection machinery.
  ``RichLog`` is a ``ScrollView``, so the inherited ``Widget.get_selection``
  returns ``None`` for every drag and copying is dead.
* Clipboard delivery must never report success it cannot back up, and must
  always leave a recoverable file behind.
* Mode, reasoning, safety, and swarm settings must survive ``/new`` and an
  application restart instead of snapping back to defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.segment import Segment
from rich.style import Style
from textual.geometry import Offset
from textual.selection import Selection
from textual.strip import Strip

from adaptive_harness.data.config import ConfigManager, STICKY_PREFERENCE_KEYS
from adaptive_harness.data.sessions import SessionStore
from adaptive_harness.tui import clipboard
from adaptive_harness.tui.app import AdaptiveHarnessApp, HarnessScreen
from adaptive_harness.tui.widgets import ClassifierTelemetryWidget, PinnedRichLog


@pytest.fixture(autouse=True)
def _isolated_clipboard_probe(monkeypatch):
    """Never touch a developer's real clipboard while running the suite."""
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH2_AUTH_SOCK", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("STY", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
    monkeypatch.setenv("TERM", "xterm-256color")
    clipboard.reset_capability_cache()
    yield
    clipboard.reset_capability_cache()


def _config_dir(tmp_path: Path) -> Path:
    return tmp_path / "preferences"


# --------------------------------------------------------------------------
# 1. The chat log is genuinely selectable
# --------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chat_log_exposes_selected_text_to_the_screen(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "select.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        log.clear()
        log.write("ALPHA BRAVO CHARLIE")
        log.write("DELTA ECHO FOXTROT")
        await pilot.pause()

        # A drag across the first line only.
        log.mouse_down = True
        app.screen.selections = {log: Selection(Offset(6, 0), Offset(11, 0))}
        log.selection_updated(app.screen.selections[log])
        await pilot.pause()

        assert app.screen.get_selected_text() == "BRAVO"

        # The base RichLog implementation would raise IndexError/return None here.
        assert log.get_selection(Selection(Offset(0, 0), Offset(0, 2))) is not None

        # A selection that runs to the end of a line must not include padding.
        assert log.get_selection(Selection(Offset(6, 0), Offset(999, 0)))[0] == "BRAVO CHARLIE"
        assert log.get_selection(Selection(None, None))[0] == (
            "ALPHA BRAVO CHARLIE\nDELTA ECHO FOXTROT")
        log.mouse_down = False


@pytest.mark.anyio
async def test_a_rich_rendered_panel_is_selectable(tmp_path: Path):
    """A ``Panel``-rendering widget must still yield copyable text.

    ``Widget.get_selection`` only understands ``Text``/``Content`` renders, so
    every Rich-rendered panel in the app highlighted on drag and then copied
    nothing at all.
    """
    app = AdaptiveHarnessApp(db_path=tmp_path / "panel.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(120, 30)) as pilot:
        telemetry = app.query_one("#telemetry", ClassifierTelemetryWidget)
        await pilot.pause()

        header = telemetry.get_selection(Selection(Offset(0, 0), Offset(0, 1)))
        assert header is not None and header[0].strip()
        # The panel border is part of what the user can see and select.
        assert header[0].lstrip().startswith(("╭", "┌"))

        first_row = telemetry.get_selection(Selection(Offset(0, 0), Offset(12, 0)))
        assert first_row is not None and first_row[0]

        # A drag that runs off the end of the content must not raise.
        assert telemetry.get_selection(Selection(Offset(0, 0), Offset(0, 5000))) is not None
        # A selection that starts past the end has nothing to extract.
        assert telemetry.get_selection(Selection(Offset(0, 9000), Offset(0, 9001))) is None


@pytest.mark.anyio
async def test_dragging_over_the_chat_log_copies_the_dragged_range(tmp_path: Path):
    """End-to-end: real mouse events, real compositor offsets, real copy."""
    app = AdaptiveHarnessApp(db_path=tmp_path / "drag.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        log.clear()
        log.write("COPYME-UNIQUE-TOKEN")
        await pilot.pause()
        await pilot.pause()

        await pilot.mouse_down(log, offset=(0, 0))
        await pilot.hover(log, offset=(10, 0))
        await pilot.pause()
        await pilot.mouse_up(log, offset=(10, 0))
        await pilot.pause()

        # Textual treats the drag end as inclusive.
        assert app.screen.get_selected_text() == "COPYME-UNIQ"

        copied: list[str] = []
        app.copy_to_clipboard = copied.append
        app._handle_slash_command("/copy")
        # Delivery runs on a worker thread so a slow clipboard helper cannot
        # freeze the interface, so wait for it rather than assuming it is done.
        for _ in range(40):
            if copied:
                break
            await pilot.pause()
        assert copied == ["COPYME-UNIQ"]


@pytest.mark.anyio
async def test_drag_starting_at_the_left_edge_selects_the_right_rows(tmp_path: Path):
    """Regression: the log's border/padding gutter produced wrong offsets.

    Textual's compositor cannot map a pointer inside a widget's border or
    padding back to content and returns raw viewport coordinates instead, so a
    drag that began in the gutter selected wildly wrong rows. The log therefore
    lives inside a framed container and keeps a zero gutter of its own.
    """
    app = AdaptiveHarnessApp(db_path=tmp_path / "gutter.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        log.clear()
        for index in range(60):
            log.write(f"row{index:02d} the quick brown fox")
        await pilot.pause()
        await pilot.pause()
        assert log.scroll_offset.y > 0, "the log must be scrolled for this to matter"

        # Start at the very first column of the chat area.
        await pilot.mouse_down(log, offset=(0, 2))
        await pilot.hover(log, offset=(12, 3))
        await pilot.pause()
        await pilot.mouse_up(log, offset=(12, 3))
        await pilot.pause()

        selection = app.screen.selections[log]
        top = log.scroll_offset.y + 2
        assert selection.start == Offset(0, top), selection
        assert selection.end == Offset(13, top + 1), selection

        text = app.screen.get_selected_text()
        assert text is not None
        rows = text.splitlines()
        # The first row runs to its end, the last stops at the drag column.
        assert rows == [f"row{top:02d} the quick brown fox", f"row{top + 1:02d} the qui"]


@pytest.mark.anyio
async def test_chat_log_has_no_selection_gutter(tmp_path: Path):
    """Every point inside the log must also be a point inside its content."""
    app = AdaptiveHarnessApp(db_path=tmp_path / "gutter2.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        await pilot.pause()
        assert log.gutter.top_left == Offset(0, 0)
        assert log.region == log.content_region
        # The frame still draws the border and padding the design calls for.
        frame = app.query_one("#chat-frame")
        assert frame.gutter.top_left == Offset(2, 2)


@pytest.mark.anyio
async def test_selection_highlight_is_painted_on_the_chat_log(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "highlight.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        log.clear()
        log.write("HIGHLIGHTED")
        await pilot.pause()

        plain = log.render_line(0)
        app.screen.selections = {log: Selection(Offset(0, 0), Offset(11, 0))}
        log.selection_updated(app.screen.selections[log])
        await pilot.pause()

        assert app.screen.get_component_rich_style("screen--selection") is not None
        highlighted = log.render_line(0)
        # Same text and same width; only the selected span is restyled.
        assert highlighted.text == plain.text
        assert highlighted.cell_length == plain.cell_length
        assert highlighted.text.startswith("HIGHLIGHTED")
        assert highlighted != plain
        assert [segment.style for segment in highlighted] != [segment.style for segment in plain]

        # A mid-line selection must not truncate the text after the highlight.
        app.screen.selections = {log: Selection(Offset(2, 0), Offset(5, 0))}
        log.selection_updated(app.screen.selections[log])
        await pilot.pause()
        middle = log.render_line(0)
        assert middle.text == plain.text
        assert middle.cell_length == plain.cell_length
        app.screen.clear_selection()


@pytest.mark.anyio
async def test_scrolled_chat_log_selects_the_right_content_rows(tmp_path: Path):
    """Selection offsets are content relative, so scrolling must not shift them."""
    app = AdaptiveHarnessApp(db_path=tmp_path / "scroll.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 20)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        log.clear()
        for index in range(200):
            log.write(f"ROW{index:03d}")
        log.scroll_to(y=120, animate=False)
        await pilot.pause()
        await pilot.pause()

        assert log.scroll_offset.y > 0
        row = log.scroll_offset.y + 2
        assert log.get_selection(Selection(Offset(0, row), Offset(6, row)))[0] == f"ROW{row:03d}"
        # The compositor must translate a screen row back to a content row.
        region = log.content_region
        widget, offset = app.screen.get_widget_and_offset_at(
            region.x + 3, region.y + 2)
        assert widget is log
        assert offset is not None and offset.y == log.scroll_offset.y + 2


def _log_with_strips(*rows: str) -> PinnedRichLog:
    """Build a log whose strips are exactly ``rows`` padded to 20 cells."""
    log = PinnedRichLog()
    log.lines = [Strip([Segment(row, Style(color="white"))], 20) for row in rows]
    return log


def test_pinned_log_extracts_text_without_trailing_padding():
    log = _log_with_strips("short", "next row")
    assert log.get_selection(Selection(Offset(0, 0), Offset(5, 0))) == ("short", "\n")
    # A drag past the right edge must not copy the strip's padding.
    assert log.get_selection(Selection(Offset(0, 0), Offset(999, 0)))[0] == "short"
    assert log.get_selection(Selection(Offset(0, 0), Offset(0, 1)))[0] == "short\n"
    assert log.get_selection(Selection(None, None))[0] == "short\nnext row"
    assert PinnedRichLog().get_selection(Selection(None, None)) is None


def test_pinned_log_never_raises_on_selections_past_the_last_line():
    """The log is far taller than its content; blank rows must be harmless."""
    log = _log_with_strips("only row")
    assert log.get_selection(Selection(Offset(0, 12), Offset(4, 15))) is None
    assert log.get_selection(Selection(Offset(0, 0), Offset(4, 30)))[0] == "only row"
    # A right-to-left drag yields the same span, in visual order.
    assert log.get_selection(Selection(Offset(9, 0), Offset(2, 0)))[0] == "ly row"


# --------------------------------------------------------------------------
# 2. Clipboard delivery is honest and always recoverable
# --------------------------------------------------------------------------

def test_clipboard_mirrors_payload_and_reports_osc52(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(clipboard, "system_clipboard_helper", lambda: ("xclip", "-selection", "clipboard"))
    delivery = clipboard.deliver("hello world", mirror_directory=tmp_path / "mirror")
    assert delivery.method == "osc52"
    assert delivery.delivered
    assert delivery.mirror_path is not None
    assert delivery.mirror_path.read_text() == "hello world"
    assert "osc 52" in delivery.summary().casefold()


def test_clipboard_falls_back_to_a_system_helper_when_osc52_is_dropped(
        tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    clipboard.reset_capability_cache()
    calls: list[tuple[str, ...]] = []

    def fake_run(helper, *args, input=None, **kwargs):
        calls.append(helper)
        assert input == "payload"
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(clipboard, "system_clipboard_helper", lambda: ("xclip", "-selection", "clipboard"))
    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    delivery = clipboard.deliver("payload", mirror_directory=tmp_path / "mirror")
    assert calls == [("xclip", "-selection", "clipboard")]
    assert delivery.method == "system:xclip"
    assert delivery.delivered
    clipboard.reset_capability_cache()


def test_clipboard_never_fakes_a_remote_local_clipboard(tmp_path: Path, monkeypatch):
    """Over SSH, xclip would set the *server's* clipboard. That must not happen."""
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 51000 10.0.0.2 22")
    clipboard.reset_capability_cache()
    monkeypatch.setattr(clipboard, "system_clipboard_helper", lambda: ("xclip",))

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a helper was executed over SSH")

    monkeypatch.setattr(clipboard.subprocess, "run", explode)
    delivery = clipboard.deliver("secret", mirror_directory=tmp_path / "mirror")
    assert delivery.method == "unsupported-remote"
    assert not delivery.delivered
    assert any("remote" in warning for warning in delivery.warnings)
    assert delivery.mirror_path.read_text() == "secret"
    clipboard.reset_capability_cache()


def test_clipboard_reports_when_no_helper_exists(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    clipboard.reset_capability_cache()
    monkeypatch.setattr(clipboard, "system_clipboard_helper", lambda: None)
    delivery = clipboard.deliver("text", mirror_directory=tmp_path / "mirror")
    assert not delivery.delivered
    assert any("wl-clipboard" in warning for warning in delivery.warnings)
    clipboard.reset_capability_cache()


def test_clipboard_mirror_directory_stays_bounded(tmp_path: Path):
    directory = tmp_path / "mirror"
    for _ in range(clipboard.MIRROR_LIMIT + 12):
        clipboard.deliver("x", mirror_directory=directory)
    remaining = list(directory.glob("clip-*.txt"))
    assert len(remaining) == clipboard.MIRROR_LIMIT


def test_clipboard_survives_an_unwritable_workspace(tmp_path: Path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    delivery = clipboard.deliver("text", mirror_directory=blocker / "mirror")
    assert delivery.mirror_path is None
    assert any("recovery file" in warning for warning in delivery.warnings)
    assert delivery.method == "osc52"


@pytest.mark.anyio
async def test_a_failed_terminal_write_still_saves_a_recovery_file(tmp_path: Path, monkeypatch):
    app = AdaptiveHarnessApp(db_path=tmp_path / "driver.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    monkeypatch.setattr(clipboard, "system_clipboard_helper", lambda: None)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    clipboard.reset_capability_cache()
    try:
        async with app.run_test(size=(100, 30)):
            def boom(_text):
                raise RuntimeError("driver closed")
            app.copy_to_clipboard = boom
            delivery = app.deliver_to_clipboard("never lost")
            assert not delivery.delivered
            assert delivery.mirror_path is not None
            assert delivery.mirror_path.read_text() == "never lost"
            assert any("terminal clipboard write failed" in w for w in delivery.warnings)
    finally:
        clipboard.reset_capability_cache()


@pytest.mark.anyio
async def test_copy_without_content_explains_what_to_do(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "empty.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 30)):
        app._last_agent_content = ""
        app._review_patch = ""
        app.action_copy_output()
        log = app.query_one("#chat-log", PinnedRichLog)
        assert "Nothing to copy yet" in "".join(strip.text for strip in log.lines)


@pytest.mark.anyio
async def test_ctrl_c_copies_when_selected_and_quits_otherwise(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "ctrlc.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#chat-log", PinnedRichLog)
        log.clear()
        log.write("CTRL-C-SELECTION")
        await pilot.pause()
        app.screen.selections = {log: Selection(Offset(0, 0), Offset(16, 0))}
        await pilot.pause()

        copied: list[str] = []
        app.copy_to_clipboard = copied.append
        await pilot.press("ctrl+c")
        assert copied and "CTRL-C-SELECTION" in copied[0]
        assert app.is_running

        app.screen.clear_selection()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running


def test_default_screen_rebinds_ctrl_c_to_copy_or_quit():
    app = AdaptiveHarnessApp(db_path=":memory:", workspace_root=str(Path.cwd()),
                             config_dir=Path("/tmp/adaptive-harness-binding-probe"))
    try:
        screen = app.get_default_screen()
        assert isinstance(screen, HarnessScreen)
        keys = {binding.key for binding in HarnessScreen.BINDINGS}
        assert any(key.startswith("ctrl+c") for key in keys)
        assert ("ctrl+c", "quit", "Quit") not in app.BINDINGS
    finally:
        app.session_store.close()


# --------------------------------------------------------------------------
# 3. Settings are sticky across /new, /reset, and restarts
# --------------------------------------------------------------------------

@pytest.mark.anyio
async def test_new_session_keeps_mode_thinking_safety_and_swarm(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "sticky.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(120, 30)):
        app._handle_slash_command("/mode research")
        app._handle_slash_command("/thinking deep")
        app._handle_slash_command("/safety cautious")
        app._handle_slash_command("/swarm on")
        app._handle_slash_command("/isolation on")
        first_id = app.session.id

        app._handle_slash_command("/new Second task")
        assert app.session.id != first_id
        assert app.agent.forced_mode.value == "research"
        assert app.agent.forced_thinking.value == "deep"
        assert app.agent.safety_profile == "cautious"
        assert app.swarm_mode == "on"
        assert app.isolation_mode == "on"

        # The fresh session row must already carry the settings forward.
        stored = app.session_store.load(app.session.id).settings
        assert stored["mode"] == "research"
        assert stored["thinking"] == "deep"
        assert stored["swarm_mode"] == "on"

        app._handle_slash_command("/reset")
        assert app.agent.forced_mode.value == "research"
        assert app.agent.forced_thinking.value == "deep"
        assert app.agent.safety_profile == "cautious"

        app._handle_slash_command("/reset defaults")
        assert app.agent.forced_mode.value == "coding"
        assert app.agent.forced_thinking is None
        assert app.agent.safety_profile == "turbo"
        assert app.swarm_mode == "off"
        assert app.isolation_mode == "off"


@pytest.mark.anyio
async def test_settings_survive_an_application_restart(tmp_path: Path):
    config_dir = _config_dir(tmp_path)
    app = AdaptiveHarnessApp(db_path=tmp_path / "restart.db", workspace_root=str(tmp_path),
                             config_dir=config_dir)
    async with app.run_test(size=(120, 30)):
        app._handle_slash_command("/mode science")
        app._handle_slash_command("/thinking deep")
        app._handle_slash_command("/safety strict")
        app._handle_slash_command("/swarm on")
        app._handle_slash_command("/isolation on")
        app._handle_slash_command("/steps 7")
    assert json.loads(json.dumps(app.config.preferences()))["mode"] == "science"

    reopened = AdaptiveHarnessApp(db_path=tmp_path / "restart.db", workspace_root=str(tmp_path),
                                  config_dir=config_dir)
    assert reopened.agent.forced_mode.value == "science"
    assert reopened.agent.forced_thinking.value == "deep"
    assert reopened.agent.safety_profile == "strict"
    assert reopened.swarm_mode == "on"
    assert reopened.isolation_mode == "on"
    assert reopened.agent.step_policy == "classifier"
    assert reopened.agent.max_steps == 7
    assert reopened.agent.swarm_enabled is True
    reopened.session_store.close()


@pytest.mark.anyio
async def test_command_line_flags_still_win_over_saved_settings(tmp_path: Path):
    config_dir = _config_dir(tmp_path)
    app = AdaptiveHarnessApp(db_path=tmp_path / "cli.db", workspace_root=str(tmp_path),
                             config_dir=config_dir)
    async with app.run_test(size=(120, 30)):
        app._handle_slash_command("/mode research")
        app._handle_slash_command("/swarm on")

    overridden = AdaptiveHarnessApp(db_path=tmp_path / "cli.db", workspace_root=str(tmp_path),
                                   config_dir=config_dir, mode="security", safety="balanced")
    assert overridden.agent.forced_mode.value == "audit"
    assert overridden.agent.safety_profile == "balanced"
    # Flags override, but do not erase what the user saved for other launches.
    assert overridden.swarm_mode == "on"
    overridden.session_store.close()

    # A one-off launch flag must not silently rewrite the stored profile.
    again = AdaptiveHarnessApp(db_path=tmp_path / "cli.db", workspace_root=str(tmp_path),
                               config_dir=config_dir)
    assert again.agent.forced_mode.value == "research"
    again.session_store.close()


@pytest.mark.anyio
async def test_explicit_reset_defaults_clears_the_stored_profile(tmp_path: Path):
    config_dir = _config_dir(tmp_path)
    app = AdaptiveHarnessApp(db_path=tmp_path / "cleared.db", workspace_root=str(tmp_path),
                             config_dir=config_dir)
    async with app.run_test(size=(120, 30)):
        app._handle_slash_command("/mode research")
        app._handle_slash_command("/swarm on")
        assert ConfigManager(config_dir).preferences()["mode"] == "research"
        app._handle_slash_command("/reset defaults")
    stored = ConfigManager(config_dir).preferences()
    assert stored["mode"] == "coding"
    assert stored["swarm_mode"] == "off"
    assert stored["thinking"] == "auto"


@pytest.mark.anyio
async def test_step_policy_survives_instead_of_being_clobbered_by_its_default(tmp_path: Path):
    """Regression: the CLI default made every step-policy restore unreachable."""
    config_dir = _config_dir(tmp_path)
    database = tmp_path / "steps.db"
    app = AdaptiveHarnessApp(db_path=database, workspace_root=str(tmp_path),
                             config_dir=config_dir)
    async with app.run_test(size=(120, 30)):
        app._handle_slash_command("/steps unbounded")
        session_id = app.session.id

    restored = AdaptiveHarnessApp(db_path=database, workspace_root=str(tmp_path),
                                  config_dir=config_dir, session_id=session_id)
    assert restored.agent.step_policy == "unbounded"
    restored.session_store.close()

    pinned = AdaptiveHarnessApp(db_path=database, workspace_root=str(tmp_path),
                                config_dir=config_dir, step_policy="fixed", max_steps=3)
    assert pinned.agent.step_policy == "fixed"
    assert pinned.agent.max_steps == 3
    pinned.session_store.close()


def test_a_saved_session_still_overrides_remembered_settings(tmp_path: Path):
    config_dir = _config_dir(tmp_path)
    ConfigManager(config_dir).save_preferences(mode="coding", thinking="low", swarm_mode="off")
    database = tmp_path / "sessions.db"
    store = SessionStore(database)
    saved = store.create(str(tmp_path))
    saved.settings = {"mode": "research", "thinking": "deep", "safety": "cautious",
                      "swarm_mode": "on"}
    store.save(saved)
    store.close()

    app = AdaptiveHarnessApp(db_path=database, session_id=saved.id, config_dir=config_dir)
    assert app.agent.forced_mode.value == "research"
    assert app.agent.forced_thinking.value == "deep"
    assert app.agent.safety_profile == "cautious"
    assert app.swarm_mode == "on"
    app.session_store.close()


def test_older_session_rows_fall_back_to_remembered_settings(tmp_path: Path):
    """A row saved before swarm/isolation existed must not reset them."""
    config_dir = _config_dir(tmp_path)
    ConfigManager(config_dir).save_preferences(swarm_mode="on", isolation_mode="on",
                                              safety="cautious")
    database = tmp_path / "legacy.db"
    store = SessionStore(database)
    saved = store.create(str(tmp_path))
    saved.settings = {"safety": "cautious"}
    store.save(saved)
    store.close()

    app = AdaptiveHarnessApp(db_path=database, session_id=saved.id, config_dir=config_dir)
    assert app.swarm_mode == "on"
    assert app.isolation_mode == "on"
    assert app.agent.safety_profile == "cautious"
    app.session_store.close()


def test_corrupt_saved_settings_degrade_to_defaults(tmp_path: Path):
    config_dir = _config_dir(tmp_path)
    ConfigManager(config_dir).update(mode="nonsense", thinking="nonsense",
                                    safety="nonsense", step_policy="nonsense",
                                    max_steps="-4", swarm_mode="nonsense")
    app = AdaptiveHarnessApp(db_path=tmp_path / "corrupt.db", workspace_root=str(tmp_path),
                             config_dir=config_dir)
    assert app.agent.forced_mode.value == "coding"
    assert app.agent.forced_thinking is None
    assert app.agent.safety_profile == "turbo"
    assert app.agent.step_policy == "classifier"
    assert app.agent.max_steps is None
    assert app.swarm_mode == "off"
    assert app._preference_warning
    app.session_store.close()


def test_a_fresh_session_starts_in_coding_mode_with_the_swarm_off(tmp_path: Path):
    """A new install must not spend tokens on swarm delegation by surprise.

    Coding is the harness's primary job, and multi-agent delegation multiplies
    cost and latency, so both are opt-in from a clean configuration.
    """
    app = AdaptiveHarnessApp(db_path=tmp_path / "defaults.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    assert app.agent.forced_mode.value == "coding"
    assert app.swarm_mode == "off"
    # "Off" must actually remove the tool, not merely flip a label.
    assert app.agent.swarm_enabled is False
    assert "delegate_subagent" not in app.agent.tools
    app.session_store.close()


def test_an_explicit_mode_flag_still_wins_over_the_coding_default(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "flag.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path), mode="research")
    assert app.agent.forced_mode.value == "research"
    assert app._cli_mode_override.value == "research"
    app.session_store.close()


def test_the_coding_default_does_not_block_a_remembered_mode(tmp_path: Path):
    """A saved preference must still win, or "sticky" would be a lie."""
    config_dir = _config_dir(tmp_path)
    ConfigManager(config_dir).save_preferences(mode="science", swarm_mode="on")
    app = AdaptiveHarnessApp(db_path=tmp_path / "sticky.db", workspace_root=str(tmp_path),
                             config_dir=config_dir)
    assert app.agent.forced_mode.value == "science"
    assert app.swarm_mode == "on"
    assert "delegate_subagent" in app.agent.tools
    app.session_store.close()


def test_sticky_preferences_round_trip_and_are_validated(tmp_path: Path):
    config = ConfigManager(tmp_path / "prefs")
    config.save_preferences(mode="research", safety="strict")
    assert config.preferences() == {"mode": "research", "safety": "strict"}
    assert set(config.preferences()) <= set(STICKY_PREFERENCE_KEYS)
    with pytest.raises(ValueError):
        config.save_preferences(api_key="leak")
    assert config.clear_preferences() == {}
    assert config.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.anyio
async def test_unmount_saves_the_session_and_closes_the_store(tmp_path: Path):
    """Regression: a second `on_unmount` used to shadow the real one."""
    app = AdaptiveHarnessApp(db_path=tmp_path / "unmount.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(100, 30)):
        app._handle_slash_command("/mode research")
        session_id = app.session.id
    with pytest.raises(Exception):
        app.session_store.load(session_id)
    assert SessionStore(tmp_path / "unmount.db").load(session_id).settings["mode"] == "research"
