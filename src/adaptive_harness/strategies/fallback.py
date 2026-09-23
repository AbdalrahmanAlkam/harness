"""Fallback strategy providing graceful degradation when no specialist can handle the task."""

from __future__ import annotations

from typing import Any, Dict, Optional

from adaptive_harness.models.domain import Task, Result, VerificationResult
from adaptive_harness.strategies.base import Strategy


class FallbackStrategy(Strategy):
    """Safety fallback strategy invoked when specialists fail or confidence is depleted."""

    name = "fallback"

    def can_handle(self, task: Task) -> bool:
        # Fallback can technically accept any task
        return True

    def execute(self, task: Task) -> Result:
        raw_text = task.text.strip()
        return Result(
            value={
                "message": "Task could not be resolved by available specialist strategies",
                "task_received": raw_text,
                "status": "degraded_fallback",
            },
            strategy_name=self.name,
            success=False,  # Signal that the specialist task was not solved, but structured fallback succeeded
            metadata={"fallback": True, "task_length": len(raw_text)},
            error="No specialist strategy could confidently execute and verify this task",
        )

    def verify(self, task: Task, result: Result) -> VerificationResult:
        # The fallback execution itself is considered validly executed as fallback
        return VerificationResult(
            success=True,
            reason="Fallback response generated properly",
            confidence=1.0,
            details={"is_fallback": True},
        )
