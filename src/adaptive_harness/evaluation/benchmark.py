"""Comprehensive benchmark runner evaluating classifier vs harness across policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from adaptive_harness.data.dataset import build_synthetic_splits
from adaptive_harness.data.generator import SyntheticDataGenerator
from adaptive_harness.evaluation.metrics import compute_harness_aggregate_metrics
from adaptive_harness.harness.harness import Harness
from adaptive_harness.harness.policy import (
    CascadingPolicy,
    ExplorePolicy,
    GreedyPolicy,
    RoutingPolicy,
    ThresholdPolicy,
)
from adaptive_harness.models.classifier import TaskClassifier
from adaptive_harness.models.domain import ExecutionTrace, Task


@dataclass
class BenchmarkResult:
    """Encapsulates outcomes of running the benchmark suite."""

    policy_name: str
    metrics: Dict[str, Any]
    traces: List[ExecutionTrace]
    failure_cases: List[Dict[str, Any]]
    recovered_cases: List[Dict[str, Any]]


class BenchmarkRunner:
    """Executes multi-policy ablation benchmarks on unseen task distributions."""

    def __init__(
        self,
        classifier: TaskClassifier,
        benchmark_tasks: Optional[List[Tuple[str, str]]] = None,
        n_samples_per_class: int = 100,
        seed: int = 1234,
    ):
        self.classifier = classifier
        if benchmark_tasks is not None:
            self.tasks = benchmark_tasks
        else:
            # Generate held-out unseen benchmark tasks with a separate seed
            gen = SyntheticDataGenerator(seed=seed)
            self.tasks = gen.generate_all(samples_per_class=n_samples_per_class)

    def run_policy(
        self,
        policy: RoutingPolicy,
        policy_label: Optional[str] = None,
    ) -> BenchmarkResult:
        """Evaluates the harness configured with a specific routing policy."""
        harness = Harness(classifier=self.classifier, policy=policy)
        traces: List[ExecutionTrace] = []

        for text, true_label in self.tasks:
            trace = harness.run(text, ground_truth=true_label)
            traces.append(trace)

        label = policy_label or policy.name
        metrics = compute_harness_aggregate_metrics(traces)
        metrics["policy"] = label

        # Analyze confident mistakes and recoveries
        failure_cases = []
        recovered_cases = []

        for t in traces:
            gt = t.metadata.get("ground_truth")
            is_mispredicted = (gt is not None and t.classifier_top1 != gt)

            if is_mispredicted and t.recovered:
                recovered_cases.append({
                    "task": t.task_text,
                    "predicted": t.classifier_top1,
                    "confidence": t.confidence,
                    "ground_truth": gt,
                    "final_strategy": t.final_strategy,
                    "attempts": [a.strategy for a in t.attempts],
                    "recovered": True,
                })
            elif not t.success and gt != "fallback":
                failure_cases.append({
                    "task": t.task_text,
                    "predicted": t.classifier_top1,
                    "confidence": t.confidence,
                    "ground_truth": gt,
                    "final_strategy": t.final_strategy,
                    "attempts": [a.strategy for a in t.attempts],
                    "recovered": False,
                })

        # Sort by confidence descending to spotlight most confident mispredictions
        recovered_cases.sort(key=lambda x: x["confidence"], reverse=True)
        failure_cases.sort(key=lambda x: x["confidence"], reverse=True)

        return BenchmarkResult(
            policy_name=label,
            metrics=metrics,
            traces=traces,
            failure_cases=failure_cases,
            recovered_cases=recovered_cases,
        )

    def run_ablation_suite(
        self,
        calibrated_classifier: Optional[TaskClassifier] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, BenchmarkResult]]:
        """Runs the complete ablation suite comparing all policy configurations."""
        configs = [
            ("Greedy (No Retry)", GreedyPolicy(allow_fallback=False)),
            ("Greedy + Fallback", GreedyPolicy(allow_fallback=True)),
            ("Threshold Policy", ThresholdPolicy(high_confidence=0.85, low_confidence=0.55)),
            ("Explore Policy (eps=0.15)", ExplorePolicy(epsilon=0.15, seed=42)),
            ("Cascading Policy", CascadingPolicy(min_prob=0.05)),
        ]

        results: Dict[str, BenchmarkResult] = {}
        rows = []

        for label, policy in configs:
            res = self.run_policy(policy, policy_label=label)
            results[label] = res
            m = res.metrics
            rows.append({
                "Policy": label,
                "Classifier Acc": f"{m['classifier_accuracy']*100:.1f}%",
                "Harness Success": f"{m['harness_success_rate']*100:.1f}%",
                "Recovery Rate": f"{m['recovery_rate']*100:.1f}%",
                "Avg Attempts": m["average_attempts"],
                "Avg Latency (ms)": m["average_time_ms"],
                "Fallback Rate": f"{m['fallback_rate']*100:.1f}%",
            })

        # If a calibrated classifier is available, add Calibrated + Threshold
        if calibrated_classifier is not None:
            cal_runner = BenchmarkRunner(
                classifier=calibrated_classifier,
                benchmark_tasks=self.tasks,
            )
            cal_res = cal_runner.run_policy(
                ThresholdPolicy(high_confidence=0.85, low_confidence=0.55),
                policy_label="Calibrated + Threshold",
            )
            results["Calibrated + Threshold"] = cal_res
            m = cal_res.metrics
            rows.append({
                "Policy": "Calibrated + Threshold",
                "Classifier Acc": f"{m['classifier_accuracy']*100:.1f}%",
                "Harness Success": f"{m['harness_success_rate']*100:.1f}%",
                "Recovery Rate": f"{m['recovery_rate']*100:.1f}%",
                "Avg Attempts": m["average_attempts"],
                "Avg Latency (ms)": m["average_time_ms"],
                "Fallback Rate": f"{m['fallback_rate']*100:.1f}%",
            })

        df = pd.DataFrame(rows)
        return df, results
