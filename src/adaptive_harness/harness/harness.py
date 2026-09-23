"""Core Harness coordinating routing, execution, verification, recovery, and persistence."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Union

from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.harness.fallback import FallbackHandler
from adaptive_harness.harness.policy import RoutingPolicy, ThresholdPolicy
from adaptive_harness.harness.router import Router, RoutingUncertainty
from adaptive_harness.harness.verifier import Verifier
from adaptive_harness.models.classifier import TaskClassifier
from adaptive_harness.models.domain import ExecutionAttempt, ExecutionTrace, Result, Task, VerificationResult
from adaptive_harness.strategies import Strategy, get_default_strategies


class Harness:
    """Adaptive Agent Harness executing tasks through ML routing, active verification, and recovery."""

    def __init__(
        self,
        classifier: TaskClassifier,
        policy: Optional[RoutingPolicy] = None,
        strategies: Optional[Dict[str, Strategy]] = None,
        repository: Optional[ExperienceRepository] = None,
        router: Optional[Router] = None,
    ):
        self.classifier = classifier
        self.policy = policy or ThresholdPolicy()
        self.strategies = strategies or get_default_strategies()
        self.repository = repository
        self.router = router or Router(classifier)
        self.verifier = Verifier()
        self.fallback_handler = FallbackHandler()

    def run(
        self,
        task_or_text: Union[Task, str],
        ground_truth: Optional[str] = None,
    ) -> ExecutionTrace:
        """Executes a task through the complete harness routing and recovery loop."""
        start_wall_time = time.perf_counter()

        if isinstance(task_or_text, str):
            task = Task(text=task_or_text)
        else:
            task = task_or_text

        # 1. Router computes probabilities and uncertainty metrics
        probabilities, uncertainty = self.router.score(task)

        # 2. Decision policy determines planned trial sequence
        trial_plan = self.policy.plan_execution(uncertainty.ranked_strategies, uncertainty)

        # 3. Strategy execution and verification loop
        attempts: List[ExecutionAttempt] = []
        final_strategy = uncertainty.top1_strategy
        harness_success = False
        final_result: Optional[Result] = None

        for strat_name in trial_plan:
            strategy = self.strategies.get(strat_name)
            if not strategy:
                continue

            # Execute strategy
            result = strategy.run_with_timing(task)

            # Active verification
            v_res = self.verifier.verify(strategy, task, result)

            attempt = ExecutionAttempt(
                strategy=strat_name,
                success=result.success and v_res.success,
                time_ms=round(result.execution_time_ms, 2),
                value=result.value,
                error=result.error if not result.success else (None if v_res.success else v_res.reason),
                verification=v_res,
            )
            attempts.append(attempt)

            if attempt.success:
                final_strategy = strat_name
                harness_success = (strat_name != "fallback")  # Fallback is graceful degradation, not task success
                final_result = result
                break

        # 4. If all planned strategies failed, trigger fallback if not already executed
        if not harness_success:
            if not attempts or attempts[-1].strategy != "fallback":
                fallback_res = self.fallback_handler.handle(task, attempts)
                fb_v = self.verifier.verify(self.strategies["fallback"], task, fallback_res)
                attempts.append(
                    ExecutionAttempt(
                        strategy="fallback",
                        success=False,
                        time_ms=round(fallback_res.execution_time_ms, 2),
                        value=fallback_res.value,
                        error=fallback_res.error,
                        verification=fb_v,
                    )
                )
            final_strategy = "fallback"

        total_wall_ms = (time.perf_counter() - start_wall_time) * 1000.0

        # Check recovery: classifier top-1 failed or was wrong, but harness succeeded!
        recovered = (
            harness_success
            and final_strategy != uncertainty.top1_strategy
            and len(attempts) > 1
        )

        classifier_correct = (
            (uncertainty.top1_strategy == ground_truth)
            if ground_truth is not None
            else None
        )

        trace = ExecutionTrace(
            task_id=task.id,
            task_text=task.text,
            classifier_probabilities=probabilities,
            entropy=uncertainty.entropy,
            confidence=uncertainty.confidence,
            margin=uncertainty.margin,
            policy=self.policy.name,
            attempts=attempts,
            final_strategy=final_strategy,
            success=harness_success,
            classifier_top1=uncertainty.top1_strategy,
            classifier_correct=classifier_correct,
            recovered=recovered,
            total_time_ms=round(total_wall_ms, 2),
            metadata={
                "ground_truth": ground_truth,
                "trial_plan": trial_plan,
            },
        )

        # 5. Persist experience to repository if available
        if self.repository is not None:
            try:
                self.repository.record_trace(trace)
            except Exception:
                pass  # Do not allow database logging failure to crash harness execution

        return trace
