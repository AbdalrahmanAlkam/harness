"""Domain models for tasks, execution results, verification, and execution traces."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Task(BaseModel):
    """Represents an incoming task for the harness."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class VerificationResult(BaseModel):
    """Outcome of verifying a strategy execution."""

    success: bool
    reason: str = ""
    confidence: float = 1.0
    details: Dict[str, Any] = Field(default_factory=dict)


class Result(BaseModel):
    """Output produced by a strategy execution."""

    value: Any = None
    strategy_name: str
    success: bool
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ExecutionAttempt(BaseModel):
    """A single execution attempt by a strategy during harness processing."""

    strategy: str
    success: bool
    time_ms: float
    value: Any = None
    error: Optional[str] = None
    verification: Optional[VerificationResult] = None


class ExecutionTrace(BaseModel):
    """Complete structured trace of a task's journey through the harness."""

    task_id: str
    task_text: str
    classifier_probabilities: Dict[str, float]
    entropy: float
    confidence: float
    margin: float
    policy: str
    attempts: List[ExecutionAttempt] = Field(default_factory=list)
    final_strategy: str
    success: bool
    classifier_top1: str
    classifier_correct: Optional[bool] = None
    recovered: bool = False
    total_time_ms: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert trace to a concise representation for reporting."""
        return {
            "task": self.task_text,
            "classifier_top1": self.classifier_top1,
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "policy": self.policy,
            "final_strategy": self.final_strategy,
            "success": self.success,
            "attempts": len(self.attempts),
            "recovered": self.recovered,
            "total_time_ms": round(self.total_time_ms, 2),
        }
