"""Tests for SQLite experience storage and continual learning retraining."""

import pytest
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.models.classifier import TaskClassifier
from adaptive_harness.models.domain import ExecutionAttempt, ExecutionTrace, VerificationResult
from adaptive_harness.models.training import retrain_from_experience


@pytest.fixture
def repo(tmp_path):
    return ExperienceRepository(tmp_path / "test_repo.db")


def test_experience_repository_crud(repo):
    trace = ExecutionTrace(
        task_id="t-123",
        task_text="calculate 50 * 2",
        classifier_probabilities={"arithmetic": 0.9, "equations": 0.1},
        entropy=0.3,
        confidence=0.9,
        margin=0.8,
        policy="threshold",
        attempts=[
            ExecutionAttempt(
                strategy="arithmetic",
                success=True,
                time_ms=1.2,
                value=100,
                verification=VerificationResult(success=True, reason="valid"),
            )
        ],
        final_strategy="arithmetic",
        success=True,
        classifier_top1="arithmetic",
        classifier_correct=True,
        recovered=False,
        total_time_ms=1.5,
    )

    row_id = repo.record_trace(trace)
    assert row_id > 0

    recent = repo.get_recent_traces(limit=10)
    assert len(recent) == 1
    assert recent[0]["task_id"] == "t-123"
    assert recent[0]["harness_success"] == 1
    assert recent[0]["verified_strategy"] == "arithmetic"

    stats = repo.get_statistics()
    assert stats["total_executions"] == 1
    assert stats["success_rate"] == 1.0

    examples = repo.get_verified_training_examples()
    assert len(examples) == 1
    assert examples[0] == ("calculate 50 * 2", "arithmetic")

    repo.clear()
    assert repo.get_statistics()["total_executions"] == 0


def test_retrain_from_experience(tmp_path):
    db_file = tmp_path / "exp.db"
    model_file = tmp_path / "retrained_model.joblib"
    repo = ExperienceRepository(db_file)

    # When DB is empty
    _, res_empty = retrain_from_experience(db_path=db_file, model_path=model_file)
    assert res_empty["retrained"] is False

    # Seed 5 verified records
    for i in range(5):
        trace = ExecutionTrace(
            task_id=f"t-{i}",
            task_text=f"calculate {10+i} + {20+i}",
            classifier_probabilities={"arithmetic": 0.8, "fallback": 0.2},
            entropy=0.4,
            confidence=0.8,
            margin=0.6,
            policy="threshold",
            attempts=[
                ExecutionAttempt(
                    strategy="arithmetic",
                    success=True,
                    time_ms=1.0,
                    value=30 + 2 * i,
                    verification=VerificationResult(success=True, reason="valid"),
                )
            ],
            final_strategy="arithmetic",
            success=True,
            classifier_top1="arithmetic",
            total_time_ms=1.2,
        )
        repo.record_trace(trace)

    clf, res = retrain_from_experience(db_path=db_file, model_path=model_file, samples_per_class=80)
    assert res["retrained"] is True
    assert res["experience_samples_used"] == 5
    assert model_file.exists()
    assert isinstance(clf, TaskClassifier)
