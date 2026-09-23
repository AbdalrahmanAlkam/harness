"""Task reasoning budget prediction."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThinkingLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    DEEP = "deep"
    EXTREME = "extreme"


BUDGET_TOKENS = {ThinkingLevel.NONE: 0, ThinkingLevel.LOW: 1000,
                 ThinkingLevel.MEDIUM: 4000, ThinkingLevel.DEEP: 8000,
                 ThinkingLevel.EXTREME: 16000}


@dataclass
class ThinkingAssessment:
    level: ThinkingLevel
    budget_tokens: int
    effort: str | None


class ThinkingClassifier:
    def classify(self, text: str) -> ThinkingAssessment:
        low = text.lower()
        if any(cue in low for cue in ("formal proof", "distributed architecture", "deadlock", "race condition", "large-scale", "concurrency")):
            level = ThinkingLevel.DEEP
        elif any(cue in low for cue in ("architecture", "cross-module", "multi-file", "algorithm", "pipeline", "refactor")) or len(text.split()) > 80:
            level = ThinkingLevel.MEDIUM
        elif any(cue in low for cue in ("git status", "list files", "read file", "show file", "directory listing")) and len(text.split()) < 20:
            level = ThinkingLevel.NONE
        else:
            level = ThinkingLevel.LOW
        effort = {ThinkingLevel.NONE: None, ThinkingLevel.LOW: "low", ThinkingLevel.MEDIUM: "medium",
                  ThinkingLevel.DEEP: "high", ThinkingLevel.EXTREME: "high"}[level]
        return ThinkingAssessment(level, BUDGET_TOKENS[level], effort)
