"""Router for probability scoring, strategy ranking, and uncertainty quantification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

from adaptive_harness.models.classifier import TaskClassifier
from adaptive_harness.models.domain import Task


@dataclass
class RoutingUncertainty:
    """Uncertainty metrics calculated from predicted probability distribution."""

    confidence: float  # Maximum probability max(p)
    entropy: float  # Shannon entropy H(p) in bits
    normalized_entropy: float  # H(p) / log2(K)
    margin: float  # Difference between top-1 and top-2 probabilities
    top1_strategy: str
    top2_strategy: str
    ranked_strategies: List[Tuple[str, float]]


class Router:
    """Analyzes classifier output to extract ranked strategies and uncertainty signals."""

    def __init__(self, classifier: TaskClassifier):
        self.classifier = classifier

    def score(self, task: Task) -> Tuple[Dict[str, float], RoutingUncertainty]:
        """Predicts probabilities and quantifies distribution uncertainty."""
        raw_probs = self.classifier.predict_proba(task.text)
        if isinstance(raw_probs, list):
            raw_probs = raw_probs[0]

        # Rank strategies by descending probability
        ranked = sorted(raw_probs.items(), key=lambda item: item[1], reverse=True)
        top1_strat, top1_prob = ranked[0]
        top2_strat, top2_prob = ranked[1] if len(ranked) > 1 else (ranked[0][0], 0.0)

        # Shannon Entropy H(p) = -sum p_i log2(p_i)
        eps = 1e-12
        num_classes = max(len(ranked), 1)
        entropy = 0.0
        for _, p in ranked:
            p_clamped = max(p, eps)
            entropy -= p_clamped * math.log2(p_clamped)

        max_possible_entropy = math.log2(num_classes) if num_classes > 1 else 1.0
        normalized_entropy = entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0
        margin = top1_prob - top2_prob

        uncertainty = RoutingUncertainty(
            confidence=round(top1_prob, 4),
            entropy=round(entropy, 4),
            normalized_entropy=round(normalized_entropy, 4),
            margin=round(margin, 4),
            top1_strategy=top1_strat,
            top2_strategy=top2_strat,
            ranked_strategies=ranked,
        )

        return raw_probs, uncertainty
