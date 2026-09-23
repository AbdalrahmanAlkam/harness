"""Tests for TaskClassifier, calibration, serialization, and mathematical metrics."""

from pathlib import Path
import numpy as np
import pytest

from adaptive_harness.evaluation.metrics import (
    brier_score_loss,
    confidence_margin,
    cross_entropy_loss,
    expected_calibration_error,
    shannon_entropy,
)
from adaptive_harness.models.classifier import TaskClassifier
from adaptive_harness.models.training import train_routing_model


def test_mathematical_metrics():
    # Shannon entropy of uniform 4-class distribution should be log2(4) = 2.0 bits
    uniform_p = [0.25, 0.25, 0.25, 0.25]
    assert pytest.approx(shannon_entropy(uniform_p), 0.001) == 2.0

    # Degenerate distribution (100% on one class) should have 0 entropy
    sharp_p = [1.0, 0.0, 0.0, 0.0]
    assert shannon_entropy(sharp_p) == pytest.approx(0.0, abs=0.001)

    # Confidence margin
    assert confidence_margin(0.85, 0.10) == pytest.approx(0.75)

    # Brier score: perfect prediction is 0.0
    classes = ["A", "B"]
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = ["A", "B"]
    assert pytest.approx(brier_score_loss(y_true, probs, classes), 0.001) == 0.0

    # Cross entropy: perfect prediction is 0.0
    assert pytest.approx(cross_entropy_loss(y_true, probs, classes), 0.001) == 0.0

    # ECE test with perfect calibration
    ece, diag = expected_calibration_error(y_true, probs, classes, n_bins=5)
    assert pytest.approx(ece, 0.01) == 0.0


def test_classifier_training_and_serialization(tmp_path):
    # Train small model
    clf, eval_results = train_routing_model(
        samples_per_class=100,
        calibration="sigmoid",
        save_path=tmp_path / "model.joblib",
    )

    assert eval_results["accuracy"] > 0.85
    assert len(clf.classes_) == 5

    # Check probabilities sum to 1.0
    test_task = "calculate 42 * 7"
    probs = clf.predict_proba(test_task)
    assert isinstance(probs, dict)
    assert pytest.approx(sum(probs.values()), 0.001) == 1.0
    assert probs["arithmetic"] > 0.50

    # Serialization test
    loaded_clf = TaskClassifier.load(tmp_path / "model.joblib")
    loaded_probs = loaded_clf.predict_proba(test_task)
    for k in probs:
        assert pytest.approx(probs[k], 0.001) == loaded_probs[k]
