"""Bounded terminal charts for numeric data, rendered by optional plotext."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import math
import re
from threading import Lock

from adaptive_harness.tools.base import Tool, ToolResult


_PLOT_LOCK = Lock()  # plotext's module-level figure is shared process state.
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


class PlotTerminalTool(Tool):
    """Render a line, scatter, or bar chart without executing supplied code."""

    name = "plot_terminal"
    description = ("Draw a compact terminal chart for science or benchmarks. "
                   "Uses optional plotext; supply numeric x/y arrays (or labels for a bar chart).")
    parameters = {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["line", "scatter", "bar"]},
        "x": {"type": "array", "items": {"type": ["number", "string"]},
              "description": "X coordinates, or category labels for bar charts."},
        "y": {"type": "array", "items": {"type": "number"}},
        "title": {"type": "string"},
        "width": {"type": "integer", "minimum": 30, "maximum": 100},
        "height": {"type": "integer", "minimum": 10, "maximum": 32},
    }, "required": ["kind", "x", "y"]}

    def execute(self, kind: str, x: list, y: list, title: str = "",
                width: int = 56, height: int = 18, **kwargs) -> ToolResult:
        if kind not in {"line", "scatter", "bar"}:
            return ToolResult(success=False, output="", error="Chart kind must be line, scatter, or bar")
        if (not isinstance(x, list) or not isinstance(y, list) or not x or len(x) != len(y)
                or len(x) > 500):
            return ToolResult(success=False, output="", error="x and y must be equal, non-empty arrays of at most 500 points")
        if (not isinstance(width, int) or isinstance(width, bool) or not 30 <= width <= 100
                or not isinstance(height, int) or isinstance(height, bool) or not 10 <= height <= 32):
            return ToolResult(success=False, output="", error="Chart size must be 30–100 columns by 10–32 rows")
        if not isinstance(title, str) or len(title) > 100 or any(ord(ch) < 32 for ch in title):
            return ToolResult(success=False, output="", error="Chart title must be short, single-line text")
        try:
            values = [_finite_float(v) for v in y]
            coordinates = ([str(v) for v in x] if kind == "bar"
                           else [_finite_float(v) for v in x])
            if kind == "bar" and any(len(label) > 24 or "\n" in label for label in coordinates):
                raise ValueError("Bar labels must be at most 24 characters")
        except (ValueError, TypeError) as exc:
            return ToolResult(success=False, output="", error=f"Invalid chart data: {exc}")
        try:
            import plotext as plt
        except ImportError:
            return ToolResult(success=False, output="", error="Terminal plotting needs the optional plotting extra: pip install 'adaptive-harness[plotting]'")
        try:
            with _PLOT_LOCK:
                plt.clear_figure()
                plt.plotsize(width, height)
                if title:
                    plt.title(title)
                getattr(plt, {"line": "plot", "scatter": "scatter", "bar": "bar"}[kind])(coordinates, values)
                stream = StringIO()
                with redirect_stdout(stream):
                    plt.show()
                chart = _ANSI.sub("", stream.getvalue()).strip("\n")
                plt.clear_figure()
            if not chart:
                return ToolResult(success=False, output="", error="plotext returned an empty chart")
            return ToolResult(success=True, output=chart[:12_000],
                              metadata={"kind": kind, "points": len(values), "chart": True})
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            return ToolResult(success=False, output="", error=f"Terminal chart failed: {exc}")


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("coordinates and values must be numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("coordinates and values must be finite")
    return result
