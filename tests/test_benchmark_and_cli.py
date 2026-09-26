"""Tests for BenchmarkRunner and Typer CLI subcommands."""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from adaptive_harness.cli import app
from adaptive_harness.data.generator import SyntheticDataGenerator
from adaptive_harness.evaluation.benchmark import BenchmarkRunner
from adaptive_harness.harness.policy import ThresholdPolicy
from adaptive_harness.models.training import train_routing_model

runner = CliRunner()


@pytest.fixture(scope="module")
def quick_model(tmp_path_factory):
    fn = tmp_path_factory.mktemp("models") / "quick_model.joblib"
    clf, _ = train_routing_model(samples_per_class=100, save_path=fn)
    return clf, fn


def test_benchmark_runner(quick_model):
    clf, _ = quick_model
    runner_bench = BenchmarkRunner(classifier=clf, n_samples_per_class=15, seed=123)
    res = runner_bench.run_policy(ThresholdPolicy())

    assert "classifier_accuracy" in res.metrics
    assert "harness_success_rate" in res.metrics
    assert "recovery_rate" in res.metrics
    assert len(res.traces) == 15 * 5

    df, ablation_dict = runner_bench.run_ablation_suite()
    assert len(df) >= 5
    assert "Threshold Policy" in ablation_dict


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Adaptive Agent Harness CLI" in result.output


def test_dev_offline_implementation_failure_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["dev", "create hello.py with a greet function", "--offline",
                                 "--workspace", str(tmp_path), "--db", str(tmp_path / "experience.db")])
    assert result.exit_code == 1
    assert "Offline mock cannot implement this edit" in result.output
    assert not (tmp_path / "hello.py").exists()


def test_cli_run_task(quick_model, tmp_path):
    _, model_path = quick_model
    db_path = tmp_path / "cli_exp.db"
    result = runner.invoke(
        app,
        [
            "run",
            "calculate 20 + 30",
            "--model",
            str(model_path),
            "--db",
            str(db_path),
        ],
    )
    assert result.exit_code == 0
    assert "arithmetic" in result.output.lower()
    assert "Execution Result" in result.output


def test_cli_history_and_report(quick_model, tmp_path):
    _, model_path = quick_model
    db_path = tmp_path / "cli_exp.db"

    # Run once to populate DB
    runner.invoke(app, ["run", "what is 15 squared", "--model", str(model_path), "--db", str(db_path)])

    # Test history
    res_hist = runner.invoke(app, ["history", "--db", str(db_path)])
    assert res_hist.exit_code == 0
    assert "Recent Execution History" in res_hist.output

    # Test report
    res_rep = runner.invoke(app, ["report", "--db", str(db_path)])
    assert res_rep.exit_code == 0
    assert "Historical Harness Execution Summary" in res_rep.output
