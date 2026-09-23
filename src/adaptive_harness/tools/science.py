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
