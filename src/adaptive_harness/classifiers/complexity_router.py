"""Complexity Router directing tasks to optimal LLM model tiers (Fast, Standard, Reasoning)."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional

from adaptive_harness.llm.client import MODEL_TIERS


@dataclass
class ComplexityRoutingResult:
    tier: str  # 'fast', 'standard', 'reasoning'
    recommended_model: str
    confidence: float
    reasoning: str


class ComplexityRouter:
    """Classifies task cognitive complexity to dynamically route between cost-effective and deep reasoning models."""

    def __init__(self, custom_tier_models: Optional[Dict[str, str]] = None):
        self.tier_models = custom_tier_models or MODEL_TIERS

    def route(self, task_text: str) -> ComplexityRoutingResult:
        low = task_text.lower().strip()
        word_count = len(re.findall(r"\b\w+\b", task_text))

        # 1. Reasoning tier signals: multi-file, architecture, math/proof, subtle bug, algorithm design
        reasoning_cues = [
            "architect", "design", "refactor architecture", "proof", "concurrency",
            "deadlock", "race condition", "deep learning", "cryptic", "complex algorithm",
            "multi-file", "system redesign", "benchmark comparison", "formal verification",
        ]
        if any(cue in low for cue in reasoning_cues) or word_count > 80:
            return ComplexityRoutingResult(
                tier="reasoning",
                recommended_model=self.tier_models["reasoning"],
                confidence=0.88,
                reasoning="Task involves deep architectural design, complex reasoning, or extensive multi-file scope.",
            )

        # 2. Fast tier signals: simple commands, reads, listings, git status, minor typo
        fast_cues = [
            "git status", "list files", "read file", "view file", "check version",
            "show me", "where is", "quick check", "typo", "word count", "format",
        ]
        if any(cue in low for cue in fast_cues) and word_count < 25:
            return ComplexityRoutingResult(
                tier="fast",
                recommended_model=self.tier_models["fast"],
                confidence=0.92,
                reasoning="Task is a straightforward inspection, query, or lightweight command.",
            )

        # 3. Standard tier: code editing, testing, standard bugfixes
        return ComplexityRoutingResult(
            tier="standard",
            recommended_model=self.tier_models["standard"],
            confidence=0.85,
            reasoning="Standard software engineering implementation, editing, or testing task.",
        )
