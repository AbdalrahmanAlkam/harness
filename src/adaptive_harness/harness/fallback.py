"""Fallback management and recovery policies for failed executions."""

from __future__ import annotations

from typing import List, Optional

from adaptive_harness.models.domain import ExecutionAttempt, Result, Task
from adaptive_harness.strategies.fallback import FallbackStrategy


class FallbackHandler:
    """Manages fallback strategy invocation when specialist strategies fail."""

    def __init__(self, fallback_strategy: Optional[FallbackStrategy] = None):
        self.fallback_strategy = fallback_strategy or FallbackStrategy()

    def handle(self, task: Task, failed_attempts: List[ExecutionAttempt]) -> Result:
        """Executes fallback strategy and annotates context with previous failed attempts."""
        res = self.fallback_strategy.run_with_timing(task)
        res.metadata["attempted_strategies"] = [a.strategy for a in failed_attempts]
        res.metadata["failure_reasons"] = [
            a.error or (a.verification.reason if a.verification else "unknown")
            for a in failed_attempts
        ]
        return res
