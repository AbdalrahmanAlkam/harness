"""Text statistics strategy for linguistic and information-theoretic text analysis."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional

from adaptive_harness.models.domain import Task, Result, VerificationResult
from adaptive_harness.strategies.base import Strategy


def shannon_entropy(text: str) -> float:
    """Computes Shannon entropy in bits for character distribution of text."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return round(ent, 4)


class TextStatisticsStrategy(Strategy):
    """Specialist handler for text statistics, frequency analysis, and entropy."""

    name = "text_stats"

    def can_handle(self, task: Task) -> bool:
        text = task.text.lower()
        stat_markers = [
            "word-count:", "word count", "count words",
            "character-frequency:", "char frequency", "letter frequency", "frequent characters", "most frequent",
            "entropy:", "text entropy", "reading time",
            "analyze text", "text statistics", "summarize this text"
        ]
        return any(m in text for m in stat_markers)

    def execute(self, task: Task) -> Result:
        raw_text = task.text.strip()
        text = raw_text.lower()

        # Extract target text payload
        target_text = self._extract_payload(raw_text)

        if "word-count" in text or "word count" in text or "count words" in text:
            words = re.findall(r"\b\w+\b", target_text)
            unique = set(w.lower() for w in words)
            return Result(
                value={
                    "total_words": len(words),
                    "unique_words": len(unique),
                    "lexical_diversity": round(len(unique) / max(len(words), 1), 4),
                },
                strategy_name=self.name,
                success=True,
                metadata={"operation": "word_count", "text_length": len(target_text)},
            )

        if any(m in text for m in ["character-frequency", "char frequency", "letter frequency", "frequent characters", "most frequent"]):
            filtered = [c.lower() for c in target_text if not c.isspace()]
            counts = dict(Counter(filtered).most_common(10))
            return Result(
                value={"top_frequencies": counts, "total_non_space": len(filtered)},
                strategy_name=self.name,
                success=True,
                metadata={"operation": "char_frequency", "total_chars": len(filtered)},
            )

        if "entropy" in text:
            ent = shannon_entropy(target_text)
            unique_chars = len(set(target_text))
            max_possible = math.log2(max(unique_chars, 1))
            return Result(
                value={
                    "entropy_bits": ent,
                    "unique_chars": unique_chars,
                    "normalized_entropy": round(ent / max(max_possible, 1e-9), 4) if max_possible > 0 else 0.0,
                },
                strategy_name=self.name,
                success=True,
                metadata={"operation": "entropy", "text_length": len(target_text)},
            )

        # General text statistics summary if explicitly requested
        if any(m in text for m in ["analyze text", "text statistics", "summarize this text", "reading time", "stats for text", "text:"]):
            words = re.findall(r"\b\w+\b", target_text)
            word_count = len(words)
            char_count = len(target_text)
            reading_time_sec = round((word_count / 200.0) * 60.0, 1)  # 200 WPM
            ent = shannon_entropy(target_text)

            return Result(
                value={
                    "word_count": word_count,
                    "character_count": char_count,
                    "estimated_reading_time_seconds": reading_time_sec,
                    "shannon_entropy": ent,
                },
                strategy_name=self.name,
                success=True,
                metadata={"operation": "general_stats", "text_length": char_count},
            )

        return Result(
            value=None,
            strategy_name=self.name,
            success=False,
            error="Task text contains no recognizable text statistics or linguistic directives",
        )

    def _extract_payload(self, text: str) -> str:
        """Extracts the subject text following markers or prefixes."""
        # Check explicit colon marker
        if ":" in text:
            colon_idx = text.find(":")
            if colon_idx < 40:
                return text[colon_idx + 1:].strip()

        # Natural language: "count words in: Hello world" or "entropy of abc"
        cleaned = re.sub(
            r"^(?:word count (?:of|in)|count words in|entropy of|summarize this text:?|character frequency of|find most frequent characters in)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def verify(self, task: Task, result: Result) -> VerificationResult:
        if not result.success or result.value is None:
            return VerificationResult(
                success=False,
                reason=f"Text statistics failed: {result.error or 'no output'}",
            )

        val = result.value
        op = result.metadata.get("operation")

        if op == "word_count":
            tw = val.get("total_words", 0)
            uw = val.get("unique_words", 0)
            if tw < 0 or uw < 0 or uw > tw:
                return VerificationResult(success=False, reason="Invalid word count invariants")
            return VerificationResult(success=True, reason="Word count invariants verified")

        elif op == "char_frequency":
            top = val.get("top_frequencies", {})
            total = val.get("total_non_space", 0)
            if any(cnt <= 0 for cnt in top.values()):
                return VerificationResult(success=False, reason="Character frequencies must be strictly positive")
            if sum(top.values()) > total:
                return VerificationResult(success=False, reason="Top character frequency sum exceeds total count")
            return VerificationResult(success=True, reason="Character frequency invariants verified")

        elif op == "entropy":
            ent = val.get("entropy_bits", -1.0)
            if ent < 0.0:
                return VerificationResult(success=False, reason="Entropy cannot be negative")
            return VerificationResult(success=True, reason="Entropy bounds verified")

        return VerificationResult(success=True, reason="Text statistics invariants verified")
