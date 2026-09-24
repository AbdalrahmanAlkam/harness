"""Bounded chart rendering and optional REPL plotting support."""

from __future__ import annotations

import sys
from types import SimpleNamespace
import pytest

from adaptive_harness.tools.plotting import PlotTerminalTool
from adaptive_harness.tools.python_repl import RUNNER, RunPythonReplTool, _ANSI


def test_plotting_rejects_invalid_input_before_import():
    tool = PlotTerminalTool()
    assert not tool.execute("line", [0, 1], [1]).success
    assert not tool.execute("line", [0], [float("nan")]).success
    assert not tool.execute("scatter", ["bad"], [1]).success
    assert not tool.execute("bar", ["a\nb"], [1]).success
    assert not tool.execute("bar", ["a"], [1], width=500).success


def test_plotting_calls_plotext_and_returns_plain_terminal_chart(monkeypatch):
    calls = []

    def record(name):
        def call(*args):
            calls.append((name, args))
            if name == "show":
                print("\x1b[32m⣿⣿⣿\x1b[0m")
        return call

    fake = SimpleNamespace(**{name: record(name) for name in
                              ("clear_figure", "plotsize", "title", "plot", "scatter", "bar", "show")})
    monkeypatch.setitem(sys.modules, "plotext", fake)
    result = PlotTerminalTool().execute("line", [0, 1, 2], [1, 4, 2], title="Curve", width=50, height=14)
    assert result.success
    assert result.output == "⣿⣿⣿"
    assert result.metadata == {"kind": "line", "points": 3, "chart": True}
    assert ("plotsize", (50, 14)) in calls
    assert ("plot", ([0.0, 1.0, 2.0], [1.0, 4.0, 2.0])) in calls
    assert calls[-1][0] == "clear_figure"


def test_repl_preloads_plotext_if_available():
    assert "scope.update({\"plt\": plt, \"plotext\": plt})" in RUNNER
    assert "plt.show()" in RunPythonReplTool.description
    assert _ANSI.sub("", "\x1b[31m⣿\x1b[0m") == "⣿"


def test_real_plotext_renders_all_chart_kinds_when_installed():
    pytest.importorskip("plotext")
    tool = PlotTerminalTool()
    for kind, x in (("line", [0, 1, 2]), ("scatter", [0, 1, 2]),
                    ("bar", ["A", "B", "C"])):
        result = tool.execute(kind, x, [1, 3, 2], width=40, height=12)
        assert result.success, result.error
        assert "┌" in result.output and "┘" in result.output
        assert "\x1b" not in result.output
