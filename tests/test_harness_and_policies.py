"""Tests for Router, Routing Policies, and Harness recovery execution."""

import pytest
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.harness.harness import Harness
from adaptive_harness.harness.policy import (
    CascadingPolicy,
    ExplorePolicy,
    GreedyPolicy,
    ThresholdPolicy,
)
from adaptive_harness.harness.router import Router, RoutingUncertainty
from adaptive_harness.models.classifier import TaskClassifier
from adaptive_harness.models.domain import Task
from adaptive_harness.models.training import train_routing_model


@pytest.fixture(scope="module")
def shared_model():
    clf, _ = train_routing_model(samples_per_class=120, save_path=None)
    return clf


def test_router_scoring(shared_model):
    router = Router(shared_model)
    task = Task(text="calculate 12 + 34")
    probs, uncertainty = router.score(task)

    assert isinstance(probs, dict)
    assert uncertainty.confidence > 0.4
    assert uncertainty.entropy >= 0.0
    assert uncertainty.top1_strategy in shared_model.classes_
    assert len(uncertainty.ranked_strategies) == len(shared_model.classes_)


def test_policies_planning():
    unc_high = RoutingUncertainty(
        confidence=0.92,
        entropy=0.3,
        normalized_entropy=0.1,
        margin=0.8,
        top1_strategy="arithmetic",
        top2_strategy="equations",
        ranked_strategies=[("arithmetic", 0.92), ("equations", 0.05), ("fallback", 0.03)],
    )

    unc_low = RoutingUncertainty(
        confidence=0.45,
        entropy=1.8,
        normalized_entropy=0.8,
        margin=0.05,
        top1_strategy="arithmetic",
        top2_strategy="equations",
        ranked_strategies=[("arithmetic", 0.45), ("equations", 0.40), ("fallback", 0.15)],
    )

    # Greedy always does top1
    greedy = GreedyPolicy()
    assert greedy.plan_execution(unc_high.ranked_strategies, unc_high)[0] == "arithmetic"

    # Threshold with high confidence gives [top1, fallback]
    threshold = ThresholdPolicy()
    plan_high = threshold.plan_execution(unc_high.ranked_strategies, unc_high)
    assert plan_high == ["arithmetic", "fallback"]

    # Threshold with low confidence escalates candidates
    plan_low = threshold.plan_execution(unc_low.ranked_strategies, unc_low)
    assert "equations" in plan_low

    # Cascading includes non-trivial probabilities
    cascading = CascadingPolicy(min_prob=0.10)
    plan_casc = cascading.plan_execution(unc_low.ranked_strategies, unc_low)
    assert "arithmetic" in plan_casc and "equations" in plan_casc


def test_harness_execution_success(shared_model):
    harness = Harness(classifier=shared_model, policy=ThresholdPolicy())
    trace = harness.run("calculate 25 * 4")
    assert trace.success
    assert trace.final_strategy == "arithmetic"
    assert len(trace.attempts) == 1
    assert trace.attempts[0].value == 100


def test_harness_recovery_from_error(shared_model):
    # Construct an ambiguous task that tricks classifier into arithmetic but is algebraically an equation
    ambiguous_query = "calculate the value of x in 5*x + 15 = 45"
    harness = Harness(classifier=shared_model, policy=ThresholdPolicy(high_confidence=0.99, low_confidence=0.10))
    trace = harness.run(ambiguous_query, ground_truth="equations")

    # The harness must successfully solve the task
    assert trace.success
    assert trace.final_strategy == "equations"
    # Succeeded via recovery across multiple attempts
    if trace.classifier_top1 != "equations":
        assert trace.recovered is True
        assert len(trace.attempts) >= 2


def test_harness_fallback_on_unsolvable(shared_model):
    harness = Harness(classifier=shared_model, policy=ThresholdPolicy())
    trace = harness.run("what is the airspeed velocity of an unladen swallow?")
    # Fallback degradation
    assert not trace.success
    assert trace.final_strategy == "fallback"
    assert trace.attempts[-1].strategy == "fallback"


def test_harness_sqlite_recording(shared_model, tmp_path):
    repo = ExperienceRepository(tmp_path / "exp.db")
    harness = Harness(classifier=shared_model, repository=repo)

    trace = harness.run("calculate 10 + 20")
    recent = repo.get_recent_traces(limit=5)
    assert len(recent) == 1
    assert recent[0]["task_text"] == "calculate 10 + 20"
    assert recent[0]["harness_success"] == 1
