"""Verification orchestrator ensuring result validity and safety."""

from __future__ import annotations

from typing import Optional

from adaptive_harness.models.domain import Task, Result, VerificationResult
from adaptive_harness.strategies.base import Strategy


class Verifier:
    """Orchestrates strategy-level and system-level verification checks."""

    @staticmethod
    def verify(strategy: Strategy, task: Task, result: Result) -> VerificationResult:
        """Executes verification and validates invariants."""
        if not result.success:
            return VerificationResult(
                success=False,
                reason=result.error or f"Strategy {strategy.name} reported execution failure",
                confidence=1.0,
            )

        if result.value is None and strategy.name != "fallback":
            return VerificationResult(
                success=False,
                reason=f"Strategy {strategy.name} returned None value",
                confidence=1.0,
            )

        # Delegate to specialized mathematical or structural verification
        try:
            return strategy.verify(task, result)
        except Exception as e:
            return VerificationResult(
                success=False,
                reason=f"Verifier threw unexpected exception: {type(e).__name__}: {e}",
                confidence=0.5,
            )
