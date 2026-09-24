"""Keyboard palette and terminal formula rendering regressions."""

from pathlib import Path

import pytest

from adaptive_harness.tui.app import AdaptiveHarnessApp, matching_commands
from adaptive_harness.tui.formatting import format_model_markdown
from adaptive_harness.tui.widgets import CommandPalette, HistoryInput


def test_command_palette_filters_with_prefix_and_fuzzy_matches():
    assert matching_commands("/th")[0][0] == "/theme"
    assert [item[0] for item in matching_commands("/mo")[:2]] == ["/model", "/mode"]
    assert matching_commands("/pr")[0][0] == "/provider"
    assert matching_commands("/theme ") == []


@pytest.mark.anyio
async def test_palette_typing_navigation_completion_and_escape(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "palette.db", config_dir=tmp_path / "prefs")
    async with app.run_test(size=(118, 31)) as pilot:
        prompt = app.query_one("#prompt-input", HistoryInput)
        prompt.focus()
        await pilot.press("/")
        await pilot.pause()
        palette = app.query_one("#command-palette", CommandPalette)
        assert app.command_palette_visible
        assert palette.choices and palette.choices[0][0] == "/key"
        assert palette.region.height > 0
        assert palette.region.bottom <= prompt.region.y
        await pilot.press("m", "o")
        await pilot.pause()
        assert [choice[0] for choice in palette.choices[:2]] == ["/model", "/mode"]
        before = list(app.agent.messages)
        await pilot.press("down", "enter")
        await pilot.pause()
        assert prompt.value == "/mode "
        assert not app.command_palette_visible
        assert app.agent.messages == before
        prompt.value = "/pr"
        await pilot.pause()
        assert app.command_palette_visible
        await pilot.press("tab")
        await pilot.pause()
        assert prompt.value == "/provider "
        prompt.value = "/th"
        await pilot.pause()
        assert app.command_palette_visible
        await pilot.press("escape")
        await pilot.pause()
        assert prompt.value == "/th" and not app.command_palette_visible
        await pilot.press("e")
        await pilot.pause()
        assert app.command_palette_visible
        prompt.value = ""
        await pilot.pause()
        assert not app.command_palette_visible


def test_inline_math_symbols_scripts_and_scientific_units():
    source = (r"$\alpha + \beta + \theta + \lambda + \sigma + \Delta + \Omega$ "
              r"$\sum \prod \int \approx \le \ge \pm \infty \in \nabla$ "
              r"$x^2 + a_i + H_2O + 1.602 \times 10^{-19} + m/s^2$ ")
    rendered = format_model_markdown(source)
    assert "α + β + θ + λ + σ + Δ + Ω" in rendered
    assert "∑ ∏ ∫ ≈ ≤ ≥ ± ∞ ∈ ∇" in rendered
    assert "x² + aᵢ + H₂O + 1.602 × 10⁻¹⁹ + m/s²" in rendered


def test_display_math_stacks_fraction_without_evaluating_code():
    source = "Area: " + r"$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$" + "\n"
    rendered = format_model_markdown(source)
    assert "```text" in rendered and "────" in rendered
    assert "√(π)" in rendered and "∫₀^∞" in rendered
    assert "dx = " in rendered
    assert "────" in format_model_markdown(source, plain=True)
    code = "```python\nformula = r'$\\frac{a}{b}$'\n```"
    assert format_model_markdown(code) == code
