"""Tests for TextStatisticsStrategy and FallbackStrategy."""

import pytest
from adaptive_harness.models.domain import Task
from adaptive_harness.strategies.fallback import FallbackStrategy
from adaptive_harness.strategies.text_stats import TextStatisticsStrategy, shannon_entropy


def test_shannon_entropy():
    # Constant string has 0 entropy
    assert shannon_entropy("aaaaa") == 0.0
    # Equal distribution of 2 chars has 1 bit
    assert shannon_entropy("ab") == 1.0


def test_text_stats_word_count():
    strat = TextStatisticsStrategy()
    t = Task(text="word-count: The quick brown fox jumps over the lazy dog")
    r = strat.run_with_timing(t)
    assert r.success
    assert r.value["total_words"] == 9
    assert r.value["unique_words"] == 8
    assert strat.verify(t, r).success


def test_text_stats_character_frequency():
    strat = TextStatisticsStrategy()
    t = Task(text="character-frequency: aabbbcccc")
    r = strat.run_with_timing(t)
    assert r.success
    top = r.value["top_frequencies"]
    assert top["c"] == 4
    assert top["b"] == 3
    assert top["a"] == 2
    assert strat.verify(t, r).success


def test_text_stats_entropy():
    strat = TextStatisticsStrategy()
    t = Task(text="entropy: sample text for entropy measurement")
    r = strat.run_with_timing(t)
    assert r.success
    assert r.value["entropy_bits"] > 0
    assert strat.verify(t, r).success


def test_fallback_strategy():
    strat = FallbackStrategy()
    t = Task(text="who painted the Mona Lisa?")
    assert strat.can_handle(t)
    r = strat.run_with_timing(t)
    # Success is false because specialist didn't solve it
    assert not r.success
    assert r.metadata["fallback"] is True
    # Verification succeeds because fallback executed properly
    assert strat.verify(t, r).success
