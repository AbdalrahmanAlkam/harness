"""Strategies module exporting all specialists and registry."""

from __future__ import annotations

from typing import Dict, List, Type

from adaptive_harness.strategies.base import Strategy
from adaptive_harness.strategies.arithmetic import ArithmeticStrategy
from adaptive_harness.strategies.equations import EquationStrategy
from adaptive_harness.strategies.algorithms import AlgorithmStrategy
from adaptive_harness.strategies.text_stats import TextStatisticsStrategy
from adaptive_harness.strategies.fallback import FallbackStrategy

AVAILABLE_STRATEGIES: List[Type[Strategy]] = [
    ArithmeticStrategy,
    EquationStrategy,
    AlgorithmStrategy,
    TextStatisticsStrategy,
    FallbackStrategy,
]


def get_default_strategies() -> Dict[str, Strategy]:
    """Instantiates a mapping of strategy names to strategy instances."""
    return {
        cls.name: cls()
        for cls in AVAILABLE_STRATEGIES
    }


__all__ = [
    "Strategy",
    "ArithmeticStrategy",
    "EquationStrategy",
    "AlgorithmStrategy",
    "TextStatisticsStrategy",
    "FallbackStrategy",
    "AVAILABLE_STRATEGIES",
    "get_default_strategies",
]
