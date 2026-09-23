"""Mathematical evaluation metrics for classification and agent harness performance."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def shannon_entropy(probabilities: Union[List[float], np.ndarray], base: float = 2.0) -> float:
    """Computes Shannon Entropy: H(p) = - sum_i p_i * log_b(p_i).

    Measures routing uncertainty. A sharp distribution concentrated on one strategy
    yields entropy near 0, whereas an ambiguous task has high entropy (up to log_b(K)).
    """
    p_arr = np.array(probabilities, dtype=float)
    p_pos = p_arr[p_arr > 0]
    if len(p_pos) == 0:
        return 0.0
    p_norm = p_pos / np.sum(p_pos)
    log_func = np.log2 if base == 2.0 else np.log
    return float(-np.sum(p_norm * log_func(p_norm)))


def cross_entropy_loss(
    y_true: Union[List[str], np.ndarray],
    y_probs: np.ndarray,
    classes: List[str],
) -> float:
    """Computes Multi-Class Cross Entropy Loss: L = - (1/N) * sum_i sum_k y_ik * log(p_ik).

    Penalizes overconfident incorrect routing predictions heavily.
    """
    eps = 1e-12
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_samples = len(y_true)
    total_loss = 0.0

    for i, y in enumerate(y_true):
        idx = class_to_idx[y] if isinstance(y, str) else int(y)
        prob = max(y_probs[i, idx], eps)
        total_loss -= math.log(prob)

    return float(total_loss / max(n_samples, 1))


def brier_score_loss(
    y_true: Union[List[str], np.ndarray],
    y_probs: np.ndarray,
    classes: List[str],
) -> float:
    """Computes Brier Score: BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2.

    Strictly proper scoring rule assessing both discrimination and calibration.
    Ranges between 0 (perfect probabilistic prediction) and 2 (worst).
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_samples = len(y_true)
    n_classes = len(classes)

    one_hot = np.zeros((n_samples, n_classes))
    for i, y in enumerate(y_true):
        idx = class_to_idx[y] if isinstance(y, str) else int(y)
        one_hot[i, idx] = 1.0

    return float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))


def confidence_margin(top1_prob: float, top2_prob: float) -> float:
    """Computes Margin: margin = p_top1 - p_top2.

    A high margin signifies decisive routing separation between the first and second choice.
    """
    return float(top1_prob - top2_prob)


def expected_calibration_error(
    y_true: Union[List[str], np.ndarray],
    y_probs: np.ndarray,
    classes: List[str],
    n_bins: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_to_idx[y] if isinstance(y, str) else int(y) for y in y_true])
    pred_idx = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (pred_idx == y_true_idx).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    total = len(y_true_idx)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= low) & (confidences <= high) if i == n_bins - 1 else (confidences >= low) & (confidences < high)
        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            b_acc = float(np.mean(accuracies[mask]))
            b_conf = float(np.mean(confidences[mask]))
            bin_accuracies.append(b_acc)
            bin_confidences.append(b_conf)
            ece += (count / total) * abs(b_acc - b_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((low + high) / 2.0))

    return float(ece), {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": float(ece),
    }


def compute_harness_aggregate_metrics(traces: List[Any]) -> Dict[str, Any]:
    """Calculates comprehensive aggregate harness and classifier metrics from a list of ExecutionTrace objects."""
    if not traces:
        return {}

    total = len(traces)
    classifier_correct_count = 0
    top2_correct_count = 0
    harness_success_count = 0
    recovered_count = 0
    total_attempts = 0
    total_time_ms = 0.0
    fallback_count = 0
    confidences = []
    entropies = []
    margins = []

    for t in traces:
        gt = t.metadata.get("ground_truth")
        if gt is not None:
            if t.classifier_top1 == gt:
                classifier_correct_count += 1
            # Check top-2
            ranked = sorted(t.classifier_probabilities.items(), key=lambda x: x[1], reverse=True)
            top2 = [name for name, _ in ranked[:2]]
            if gt in top2:
                top2_correct_count += 1

        if t.success:
            harness_success_count += 1
        if t.recovered:
            recovered_count += 1
        if t.final_strategy == "fallback":
            fallback_count += 1

        total_attempts += len(t.attempts)
        total_time_ms += t.total_time_ms
        confidences.append(t.confidence)
        entropies.append(t.entropy)
        margins.append(t.margin)

    # Mispredictions that could potentially be recovered
    mispredicted_count = total - classifier_correct_count
    recovery_rate = (recovered_count / mispredicted_count) if mispredicted_count > 0 else 1.0

    return {
        "total_tasks": total,
        "classifier_accuracy": round(classifier_correct_count / total, 4) if total else 0.0,
        "classifier_top2_accuracy": round(top2_correct_count / total, 4) if total else 0.0,
        "harness_success_rate": round(harness_success_count / total, 4) if total else 0.0,
        "recovery_count": recovered_count,
        "recovery_rate": round(recovery_rate, 4),
        "average_attempts": round(total_attempts / total, 2) if total else 0.0,
        "average_time_ms": round(total_time_ms / total, 2) if total else 0.0,
        "fallback_rate": round(fallback_count / total, 4) if total else 0.0,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "mean_entropy": round(float(np.mean(entropies)), 4) if entropies else 0.0,
        "mean_margin": round(float(np.mean(margins)), 4) if margins else 0.0,
    }
