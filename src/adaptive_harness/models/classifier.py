"""Machine learning classifier for task routing with probability prediction and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from adaptive_harness.models.calibration import calibrate_model
from adaptive_harness.models.features import build_text_feature_union


DEFAULT_MODEL_PATH = Path("output/models/routing_classifier.joblib")


class TaskClassifier:
    """TF-IDF and Logistic Regression routing classifier with probability outputs."""

    def __init__(
        self,
        c_param: float = 2.0,
        max_iter: int = 1000,
        calibration: Optional[str] = None,
    ):
        self.c_param = c_param
        self.max_iter = max_iter
        self.calibration = calibration
        self.pipeline: Optional[Pipeline] = None
        self.calibrated_pipeline: Optional[Any] = None
        self.classes_: List[str] = []
        self._is_trained = False

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
    ) -> "TaskClassifier":
        """Trains the TF-IDF + LogisticRegression pipeline and applies calibration if specified."""
        base_pipeline = Pipeline(
            [
                ("features", build_text_feature_union()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.c_param,
                        max_iter=self.max_iter,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

        base_pipeline.fit(train_texts, train_labels)
        self.pipeline = base_pipeline
        self.classes_ = list(base_pipeline.named_steps["clf"].classes_)

        if self.calibration:
            self.calibrated_pipeline = calibrate_model(
                base_pipeline=base_pipeline,
                X_train=train_texts,
                y_train=train_labels,
                X_val=val_texts,
                y_val=val_labels,
                method=self.calibration,
            )
        else:
            self.calibrated_pipeline = None

        self._is_trained = True
        return self

    @property
    def active_model(self) -> Any:
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
        return self.calibrated_pipeline if self.calibrated_pipeline is not None else self.pipeline

    def predict(self, text_or_texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """Predicts the most probable strategy label."""
        is_single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if is_single else text_or_texts
        preds = self.active_model.predict(texts).tolist()
        return preds[0] if is_single else preds

    def predict_proba(
        self, text_or_texts: Union[str, List[str]]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Outputs probability distribution dictionary across all strategies."""
        is_single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if is_single else text_or_texts
        probs_matrix = self.active_model.predict_proba(texts)

        results: List[Dict[str, float]] = []
        for row in probs_matrix:
            dist = {cls_name: float(prob) for cls_name, prob in zip(self.classes_, row)}
            results.append(dist)

        return results[0] if is_single else results

    def predict_proba_matrix(self, texts: List[str]) -> np.ndarray:
        """Returns raw (N, K) numpy probability array aligned with self.classes_."""
        return self.active_model.predict_proba(texts)

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
        """Serializes the classifier, classes, and metadata to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline,
            "calibrated_pipeline": self.calibrated_pipeline,
            "classes": self.classes_,
            "calibration": self.calibration,
            "c_param": self.c_param,
            "max_iter": self.max_iter,
            "is_trained": self._is_trained,
        }
        joblib.dump(payload, target_path)
        return target_path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL_PATH) -> "TaskClassifier":
        """Loads a persisted classifier from disk."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at {target_path}")

        payload = joblib.load(target_path)
        instance = cls(
            c_param=payload.get("c_param", 2.0),
            max_iter=payload.get("max_iter", 1000),
            calibration=payload.get("calibration"),
        )
        instance.pipeline = payload["pipeline"]
        instance.calibrated_pipeline = payload.get("calibrated_pipeline")
        instance.classes_ = payload["classes"]
        instance._is_trained = payload["is_trained"]
        return instance
