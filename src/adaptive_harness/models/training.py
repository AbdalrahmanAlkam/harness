"""Training and retraining pipelines for task routing models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from adaptive_harness.data.dataset import build_synthetic_splits
from adaptive_harness.models.calibration import compute_brier_score, compute_ece
from adaptive_harness.models.classifier import DEFAULT_MODEL_PATH, TaskClassifier


def evaluate_classifier(
    classifier: TaskClassifier,
    texts: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics for a classifier on a dataset."""
    preds = classifier.predict(texts)
    probs_matrix = classifier.predict_proba_matrix(texts)
    classes = classifier.classes_

    acc = float(accuracy_score(labels, preds))

    # Top-2 accuracy
    top2_correct = 0
    for i, true_label in enumerate(labels):
        top2_indices = np.argsort(probs_matrix[i])[-2:]
        top2_classes = [classes[idx] for idx in top2_indices]
        if true_label in top2_classes:
            top2_correct += 1
    top2_acc = float(top2_correct / len(labels))

    ece, ece_diagram = compute_ece(labels, probs_matrix, classes)
    brier = compute_brier_score(labels, probs_matrix, classes)

    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "ece_diagram": ece_diagram,
        "classification_report": report,
        "sample_count": len(labels),
    }


def train_routing_model(
    samples_per_class: int = 800,
    seed: int = 42,
    calibration: Optional[str] = "sigmoid",
    save_path: Optional[Path | str] = DEFAULT_MODEL_PATH,
) -> Tuple[TaskClassifier, Dict[str, Any]]:
    """Generates synthetic dataset, trains classifier, and computes test metrics."""
    splits = build_synthetic_splits(samples_per_class=samples_per_class, seed=seed)

    clf = TaskClassifier(calibration=calibration)
    clf.train(
        train_texts=splits.train_texts,
        train_labels=splits.train_labels,
        val_texts=splits.val_texts,
        val_labels=splits.val_labels,
    )

    eval_results = evaluate_classifier(clf, splits.test_texts, splits.test_labels)

    if save_path:
        saved_file = clf.save(save_path)
        eval_results["saved_to"] = str(saved_file)

    return clf, eval_results


def retrain_from_experience(
    db_path: Path | str = "output/experience.db",
    model_path: Path | str = DEFAULT_MODEL_PATH,
    samples_per_class: int = 600,
) -> Tuple[Optional[TaskClassifier], Dict[str, Any]]:
    """Extracts verified executions from SQLite, merges with base data, and retrains."""
    from adaptive_harness.data.storage import ExperienceRepository
    repo = ExperienceRepository(db_path)
    experience_data = repo.get_verified_training_examples(exclude_fallback=True)

    if not experience_data:
        return None, {
            "retrained": False,
            "reason": "No verified execution records found in database",
            "experience_samples": 0,
        }

    # Load baseline splits
    splits = build_synthetic_splits(samples_per_class=samples_per_class, seed=42)

    # Augment training set with verified real-world experiences
    exp_texts = [t for t, _ in experience_data]
    exp_labels = [l for _, l in experience_data]

    augmented_train_texts = splits.train_texts + exp_texts
    augmented_train_labels = splits.train_labels + exp_labels

    # Train candidate model
    candidate_clf = TaskClassifier(calibration="sigmoid")
    candidate_clf.train(
        train_texts=augmented_train_texts,
        train_labels=augmented_train_labels,
        val_texts=splits.val_texts,
        val_labels=splits.val_labels,
    )

    candidate_eval = evaluate_classifier(candidate_clf, splits.val_texts, splits.val_labels)

    # Compare with existing model if it exists
    model_path_obj = Path(model_path)
    should_save = True
    if model_path_obj.exists():
        existing_clf = TaskClassifier.load(model_path_obj)
        existing_eval = evaluate_classifier(existing_clf, splits.val_texts, splits.val_labels)
        if candidate_eval["accuracy"] < existing_eval["accuracy"] - 0.02:
            should_save = False

    if should_save:
        candidate_clf.save(model_path)

    return candidate_clf, {
        "retrained": True,
        "saved": should_save,
        "experience_samples_used": len(experience_data),
        "validation_accuracy": candidate_eval["accuracy"],
        "validation_ece": candidate_eval["ece"],
    }
