"""Model probability calibration, Expected Calibration Error (ECE), and reliability metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import PredefinedSplit


def compute_ece(
    y_true: Union[List[str], np.ndarray],
    y_probs: np.ndarray,
    classes: List[str],
    n_bins: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """Calculates Expected Calibration Error (ECE) and reliability bin statistics.

    Args:
        y_true: Ground truth string labels or integer indices.
        y_probs: Array of shape (N, K) with predicted probability distributions.
        classes: List of class label names matching column order in y_probs.
        n_bins: Number of equal-width confidence bins between 0 and 1.

    Returns:
        Tuple of (ECE float, dictionary with bin accuracies, confidences, counts, and edges).
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if isinstance(y_true[0], str):
        y_true_indices = np.array([class_to_idx[y] for y in y_true])
    else:
        y_true_indices = np.array(y_true)

    pred_indices = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (pred_indices == y_true_indices).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    total_samples = len(y_true_indices)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include right edge for last bin
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            ece += (count / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((low + high) / 2.0))

    diagram_data = {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": float(ece),
    }

    return float(ece), diagram_data


def compute_brier_score(
    y_true: Union[List[str], np.ndarray],
    y_probs: np.ndarray,
    classes: List[str],
) -> float:
    """Calculates multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_samples = len(y_true)
    n_classes = len(classes)

    one_hot = np.zeros((n_samples, n_classes))
    for i, y in enumerate(y_true):
        idx = class_to_idx[y] if isinstance(y, str) else int(y)
        one_hot[i, idx] = 1.0

    return float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))


def calibrate_model(
    base_pipeline: Any,
    X_train: List[str],
    y_train: List[str],
    X_val: Optional[List[str]] = None,
    y_val: Optional[List[str]] = None,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Calibrates a classifier pipeline using Platt scaling or isotonic regression.

    Uses holdout validation with PredefinedSplit if validation data is provided,
    otherwise uses 3-fold cross validation on training data.
    """
    if X_val is not None and y_val is not None and len(X_val) > 0:
        X_all = list(X_train) + list(X_val)
        y_all = list(y_train) + list(y_val)
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        cv_split = PredefinedSplit(test_fold)
        calibrated = CalibratedClassifierCV(
            estimator=base_pipeline,
            method=method,
            cv=cv_split,
        )
        calibrated.fit(X_all, y_all)
    else:
        calibrated = CalibratedClassifierCV(
            estimator=base_pipeline,
            method=method,
            cv=3,
        )
        calibrated.fit(X_train, y_train)

    return calibrated
