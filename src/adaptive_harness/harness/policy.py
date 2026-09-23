"""Execution policies governing strategy trial order, retry escalation, and exploration."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import List, Optional, Tuple

from adaptive_harness.harness.router import RoutingUncertainty


class RoutingPolicy(ABC):
    """Abstract policy defining how candidates are scheduled for execution."""

    name: str

    @abstractmethod
    def plan_execution(
        self,
        candidates: List[Tuple[str, float]],
        uncertainty: RoutingUncertainty,
    ) -> List[str]:
        """Returns an ordered sequence of strategy names to attempt."""
        ...


class GreedyPolicy(RoutingPolicy):
    """Always schedules the highest-probability strategy first, followed by fallback."""

    name = "greedy"

    def __init__(self, allow_fallback: bool = True):
        self.allow_fallback = allow_fallback

    def plan_execution(
        self,
        candidates: List[Tuple[str, float]],
        uncertainty: RoutingUncertainty,
    ) -> List[str]:
        plan = [uncertainty.top1_strategy]
        if self.allow_fallback and "fallback" not in plan:
            plan.append("fallback")
        return plan


class ThresholdPolicy(RoutingPolicy):
    """Dynamically adjusts retry depth and candidate escalation based on uncertainty."""

    name = "threshold"

    def __init__(
        self,
        high_confidence: float = 0.85,
        low_confidence: float = 0.55,
        margin_threshold: float = 0.20,
        max_attempts: int = 3,
    ):
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.margin_threshold = margin_threshold
        self.max_attempts = max_attempts

    def plan_execution(
        self,
        candidates: List[Tuple[str, float]],
        uncertainty: RoutingUncertainty,
    ) -> List[str]:
        top1 = uncertainty.top1_strategy
        top2 = uncertainty.top2_strategy
        conf = uncertainty.confidence
        margin = uncertainty.margin

        plan: List[str] = []

        if conf >= self.high_confidence and margin >= self.margin_threshold:
            # High certainty: single shot then fallback
            plan = [top1]
        elif conf >= self.low_confidence:
            # Moderate uncertainty: try top1, escalate to top2
            plan = [top1]
            if top2 != top1:
                plan.append(top2)
        else:
            # High uncertainty: try top candidates up to max_attempts
            plan = [strat for strat, _ in candidates if strat != "fallback"][: self.max_attempts]

        if "fallback" not in plan:
            plan.append("fallback")

        return plan


class ExplorePolicy(RoutingPolicy):
    """Exploratory policy using epsilon-greedy routing to discover alternative handlers."""

    name = "explore"

    def __init__(
        self,
        epsilon: float = 0.15,
        base_policy: Optional[RoutingPolicy] = None,
        seed: Optional[int] = None,
    ):
        self.epsilon = epsilon
        self.base_policy = base_policy or ThresholdPolicy()
        self.rng = random.Random(seed)

    def plan_execution(
        self,
        candidates: List[Tuple[str, float]],
        uncertainty: RoutingUncertainty,
    ) -> List[str]:
        # With probability epsilon, inject exploration: try an alternate candidate first
        if self.rng.random() < self.epsilon:
            non_top1 = [strat for strat, _ in candidates if strat != uncertainty.top1_strategy and strat != "fallback"]
            if non_top1:
                explored = self.rng.choice(non_top1)
                base_plan = self.base_policy.plan_execution(candidates, uncertainty)
                # Put explored strategy first, then remaining
                remaining = [s for s in base_plan if s != explored]
                return [explored] + remaining

        return self.base_policy.plan_execution(candidates, uncertainty)


class CascadingPolicy(RoutingPolicy):
    """Escalates down candidates with non-trivial probability until success."""

    name = "cascading"

    def __init__(self, min_prob: float = 0.05, max_attempts: int = 4):
        self.min_prob = min_prob
        self.max_attempts = max_attempts

    def plan_execution(
        self,
        candidates: List[Tuple[str, float]],
        uncertainty: RoutingUncertainty,
    ) -> List[str]:
        plan = [
            strat
            for strat, p in candidates
            if p >= self.min_prob and strat != "fallback"
        ][: self.max_attempts]

        if not plan:
            plan = [uncertainty.top1_strategy]

        if "fallback" not in plan:
            plan.append("fallback")

        return plan
