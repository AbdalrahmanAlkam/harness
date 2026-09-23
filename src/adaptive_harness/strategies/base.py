"""Abstract base strategy definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Optional

from adaptive_harness.models.domain import Task, Result, VerificationResult


class Strategy(ABC):
    """Abstract base class for all execution handlers."""

    name: str

    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        """Determines if the strategy could syntactically process the task.

        Note: Routing decisions are made by the harness classifier; can_handle is
        used by the strategy to check compatibility or by fallback verification.
        """
        ...

    @abstractmethod
    def execute(self, task: Task) -> Result:
        """Executes the task and returns a structured Result."""
        ...

    @abstractmethod
    def verify(self, task: Task, result: Result) -> VerificationResult:
        """Verifies the correctness of the execution result."""
        ...

    def run_with_timing(self, task: Task) -> Result:
        """Executes the task with high-resolution wall-clock timing."""
        start_time = time.perf_counter()
        try:
            result = self.execute(task)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            result.execution_time_ms = elapsed_ms
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                execution_time_ms=elapsed_ms,
                error=f"{type(e).__name__}: {str(e)}",
            )
