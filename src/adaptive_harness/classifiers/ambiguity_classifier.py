"""Ambiguity & Uncertainty Classifier determining when to halt and ask clarifying questions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Optional

from adaptive_harness.classifiers.skill_classifier import SkillClassificationResult


@dataclass
class AmbiguityAssessment:
    """Evaluation of whether the agent should halt and query the developer."""

    should_ask_question: bool
    entropy: float
    confidence_margin: float
    risk_level: str  # 'low', 'medium', 'high'
    reason: str
    suggested_question: Optional[str] = None
    suggested_options: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_ask_question": self.should_ask_question,
            "entropy": round(self.entropy, 4),
            "confidence_margin": round(self.confidence_margin, 4),
            "risk_level": self.risk_level,
            "reason": self.reason,
            "suggested_question": self.suggested_question,
            "suggested_options": self.suggested_options or [],
        }


class AmbiguityClassifier:
    """Analyzes task clarity, entropy, and risk to trigger interactive clarification."""

    def __init__(
        self,
        entropy_threshold: float = 1.25,
        margin_threshold: float = 0.25,
    ):
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold

    def evaluate(
        self,
        task_text: str,
        skill_result: SkillClassificationResult,
    ) -> AmbiguityAssessment:
        """Evaluates whether the agent must pause and ask a clarifying question."""
        text_lower = task_text.lower().strip()

        # 1. Compute Shannon Entropy H(p) = -sum p log2(p)
        probs = list(skill_result.probabilities.values())
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)

        # 2. Confidence Margin
        ranked = skill_result.ranked_skills
        p1 = ranked[0][1] if len(ranked) > 0 else 1.0
        p2 = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = p1 - p2

        # 3. Detect high-risk actions
        high_risk_words = ["rm ", "delete all", "drop table", "git reset --hard", "wipe", "format disk"]
        is_high_risk = any(w in text_lower for w in high_risk_words)

        # 4. Detect ambiguity and underspecified phrases
        ambiguity_markers = [
            "maybe", "or something", "either", "or should i", "choose one",
            "not sure", "what do you think", "which approach", "could be"
        ]
        has_ambiguity_phrase = any(m in text_lower for m in ambiguity_markers)

        # Decision logic
        should_ask = False
        risk_level = "low"
        reason = "Task is well-specified and unambiguous."
        suggested_q = None
        suggested_opts = []

        if is_high_risk:
            should_ask = True
            risk_level = "high"
            reason = "High-risk or potentially destructive action detected."
            suggested_q = f"Are you sure you want to execute this potentially destructive operation: '{task_text}'?"
            suggested_opts = [
                "Proceed with caution",
                "Abort operation",
                "Dry-run / show preview first",
            ]
        elif skill_result.primary_skill == "ask_clarification" or has_ambiguity_phrase:
            should_ask = True
            risk_level = "medium"
            reason = "User prompt contains inherent ambiguity or explicitly solicits design selection."
            suggested_q = "How would you prefer to proceed with this task?"
            if "either" in text_lower or " or " in text_lower:
                suggested_opts = ["Option A (Standard approach)", "Option B (Alternative approach)", "Explain trade-offs first"]
            else:
                suggested_opts = ["Proceed with recommended plan", "Clarify requirements in detail", "Run tests first"]
        elif entropy > self.entropy_threshold and margin < self.margin_threshold:
            should_ask = True
            risk_level = "medium"
            reason = f"High intent distribution entropy (H={entropy:.2f} bits, margin={margin:.2f}); intent is underspecified."
            top_skills = [s for s, _ in ranked[:2]]
            suggested_q = f"Your request could involve {' or '.join(top_skills)}. What is your primary objective?"
            suggested_opts = [
                f"Perform {top_skills[0]}",
                f"Perform {top_skills[1]}",
                "Inspect workspace first",
            ]

        return AmbiguityAssessment(
            should_ask_question=should_ask,
            entropy=round(entropy, 4),
            confidence_margin=round(margin, 4),
            risk_level=risk_level,
            reason=reason,
            suggested_question=suggested_q,
            suggested_options=suggested_opts,
        )
