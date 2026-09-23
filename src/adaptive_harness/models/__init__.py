"""Models package exposing classifiers, domain entities, calibration, and training."""

from adaptive_harness.models.domain import (
    Task,
    Result,
    VerificationResult,
    ExecutionAttempt,
    ExecutionTrace,
)
from adaptive_harness.models.classifier import TaskClassifier, DEFAULT_MODEL_PATH
from adaptive_harness.models.calibration import compute_ece, compute_brier_score, calibrate_model
from adaptive_harness.models.training import train_routing_model, retrain_from_experience, evaluate_classifier

__all__ = [
    "Task",
    "Result",
    "VerificationResult",
    "ExecutionAttempt",
    "ExecutionTrace",
    "TaskClassifier",
    "DEFAULT_MODEL_PATH",
    "compute_ece",
    "compute_brier_score",
    "calibrate_model",
    "train_routing_model",
    "retrain_from_experience",
    "evaluate_classifier",
]
