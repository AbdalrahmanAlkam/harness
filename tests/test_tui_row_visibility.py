"""Row text must actually be painted, not just laid out.

A `Button` wears `border: tall` (one cell top and bottom) plus `line-pad: 1`,
so a row sized to two cells has **zero** content rows. The border and background
still paint, which reads as a highlighted bar, so the screen looks alive while
every label is invisible. This is invisible to a structural test that only
counts widgets, and to a screenshot check that only looks for *something* on the
screen; only the rendered strip proves the text is there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.settings import SettingsScreen
from adaptive_harness.tui.widgets import (
    THEME_CHOICES,
    QuickSelectModal,
    ThemePickerModal,
)


def _config_dir(tmp_path: Path) -> Path:
    path = tmp_path / "prefs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _painted(widget) -> str:
    """What the compositor would actually draw for this widget."""
    return "".join(widget.render_line(y).text for y in range(widget.size.height))


# -- the settings screen ---------------------------------------------------


@pytest.mark.anyio
async def test_every_settings_row_paints_its_label(tmp_path: Path):
    """Regression: the rows rendered as black bars with no text at all."""
    app = AdaptiveHarnessApp(db_path=tmp_path / "paint.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.press("f7")
        for _ in range(6):
            await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        rows = list(screen.query("SettingsRow"))
        settings = app._setting_rows()
        assert len(rows) == len(settings)
        for row, setting in zip(rows, settings):
            text = _painted(row).strip()
            assert row.size.height > 0, f"{setting.label} has no content row"
            assert text, f"{setting.label} painted nothing"
            assert setting.label in text, f"{setting.label} is missing from {text!r}"
            assert setting.value.split(" · ")[0] in text, (
                f"value for {setting.label} is missing from {text!r}")


@pytest.mark.anyio
async def test_the_selected_row_is_marked_and_still_readable(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "paint2.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.press("f7")
        for _ in range(6):
            await pilot.pause()
        screen = app.screen
        first = screen.query_one("SettingsRow#row-0")
        assert first.has_class("selected")
        # Moving down moves the marker *and* keeps both rows legible.
        await pilot.press("down")
        await pilot.pause()
        assert not first.has_class("selected")
        second = screen.query_one("SettingsRow#row-1")
        assert second.has_class("selected")
        assert "Secondary model" in _painted(second)
        assert "Model" in _painted(first)


# -- the pickers that share the same trap ---------------------------------


@pytest.mark.anyio
async def test_model_picker_choices_paint_their_labels(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "paint3.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = QuickSelectModal("Pick", [("alpha", "Alpha option"),
                                          ("beta", "Beta option")])
        await app.push_screen(modal)
        for _ in range(6):
            await pilot.pause()
        rows = list(modal.query(".quick-choice"))
        assert len(rows) == 2
        for row, needle in zip(rows, ("Alpha option", "Beta option")):
            assert row.size.height > 0
            assert needle in _painted(row)


@pytest.mark.anyio
async def test_theme_picker_options_paint_their_names(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "paint4.db", workspace_root=str(tmp_path),
                             config_dir=_config_dir(tmp_path))
    async with app.run_test(size=(120, 34)) as pilot:
        modal = ThemePickerModal("textual-dark", THEME_CHOICES[:3])
        await app.push_screen(modal)
        for _ in range(6):
            await pilot.pause()
        rows = list(modal.query(".theme-option"))
        assert rows
        for row in rows:
            assert row.size.height > 0
            assert _painted(row).strip(), f"{row.id} painted nothing"
