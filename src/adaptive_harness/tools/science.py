"""Local exact arithmetic check for scientific and mathematical work."""
from __future__ import annotations

import math

from adaptive_harness.strategies.arithmetic import SafeArithmeticEvaluator
from adaptive_harness.tools.base import Tool, ToolResult


class CalculateTool(Tool):
    name = "calculate"
    description = "Evaluate an arithmetic expression using a restricted AST; useful for checking numerical claims."
    parameters = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}

    def execute(self, expression: str, **kwargs) -> ToolResult:
        try:
            value = SafeArithmeticEvaluator.evaluate(expression)
            if not math.isfinite(float(value)):
                raise ValueError("Result is not finite")
            return ToolResult(success=True, output=f"{expression} = {value}", metadata={"value": value})
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult(success=False, output="", error=f"Calculation failed: {exc}")


class CheckConvergenceTool(Tool):
    """Check finite values and observed tail stability in a numerical sequence."""

    name = "check_convergence"
    description = ("Check whether the last three finite values of a numerical sequence are within a "
                   "tolerance, optionally against an expected limit. This is an empirical check, not a proof.")
    parameters = {"type": "object", "properties": {
        "values": {"type": "array", "items": {"type": "number"}, "minItems": 3},
        "tolerance": {"type": "number", "exclusiveMinimum": 0},
        "expected_limit": {"type": "number"},
    }, "required": ["values", "tolerance"]}

    def execute(self, values: list[float], tolerance: float,
                expected_limit: float | None = None, **kwargs) -> ToolResult:
        try:
            if not isinstance(values, list) or len(values) < 3:
                raise ValueError("At least three numerical values are required")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
                raise ValueError("Values must be numbers")
            samples = [float(value) for value in values]
            if not all(math.isfinite(value) for value in samples):
                raise ValueError("Values must be finite")
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
                raise ValueError("Tolerance must be a positive finite number")
            tolerance = float(tolerance)
            if not math.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("Tolerance must be a positive finite number")
            if expected_limit is not None:
                if isinstance(expected_limit, bool) or not isinstance(expected_limit, (int, float)):
                    raise ValueError("Expected limit must be a finite number")
                expected_limit = float(expected_limit)
                if not math.isfinite(expected_limit):
                    raise ValueError("Expected limit must be a finite number")
            tail = samples[-3:]
            span = max(tail) - min(tail)
            residual = abs(tail[-1] - expected_limit) if expected_limit is not None else None
            if not math.isfinite(span) or (residual is not None and not math.isfinite(residual)):
                raise ValueError("Precision overflow in tail span or limit residual")
            stable = span <= tolerance and (residual is None or residual <= tolerance)
            details = f"Last-three span: {span:.8g}; tolerance: {tolerance:.8g}"
            if residual is not None:
                details += f"; limit residual: {residual:.8g}"
            return ToolResult(success=stable, output=details,
                error=None if stable else "Observed numerical tail did not satisfy the tolerance",
                metadata={"tail_span": span, "limit_residual": residual, "tolerance": tolerance})
        except (ValueError, OverflowError) as exc:
            return ToolResult(success=False, output="", error=f"Convergence check failed: {exc}")
