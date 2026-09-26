"""Regressions for research evidence that must not become a false verdict."""

from pathlib import Path

from adaptive_harness.research.experiment import ExperimentRunner
from adaptive_harness.research.swarm import ResearchSwarm, SwarmConfig
from adaptive_harness.research.claim import Verdict
from adaptive_harness.research.proof import ProofRunner
from adaptive_harness.research.synthesis import plan_research


def test_missing_experiment_has_well_formed_receipt(tmp_path: Path):
    receipt = ExperimentRunner(tmp_path).run_script("absent.py", seed=7)
    assert receipt.status == "MISSING"
    assert receipt.data_hashes == {}
    assert receipt.exit_code is None
    assert receipt.error == "experiment script does not exist"


def test_rewritten_data_artifact_is_detected(tmp_path: Path):
    runner = ExperimentRunner(tmp_path)
    data = tmp_path / "data.csv"
    data.write_text("old")
    before = runner._snapshot()
    data.write_text("new")
    old_ns = before["data.csv"]
    data.touch()
    assert isinstance(old_ns, int)
    assert "data.csv" in runner._hash_artifacts(before)


def test_inexact_script_cannot_adjudicate_a_claim(tmp_path: Path):
    swarm = ResearchSwarm("pareto heavy tailed delay: critical index and balanced routing split", root=tmp_path,
                          config=SwarmConfig(formalize=False))
    assert swarm.plan.propositions
    prop = swarm.plan.propositions[0]
    script = swarm.workspace.proof_dir / f"{prop.prop_id.lower()}.py"
    script.write_text("x = 1.5\nprint('holds')\n")
    verdict = swarm._adjudicate()
    assert verdict.headline is not Verdict.PROVEN
    assert verdict.adjudications[0].verdict is Verdict.UNTESTED


def test_gauss_induction_step_script_has_all_imports(tmp_path: Path):
    plan = plan_research("Gauss sum of the first natural numbers")
    step = next(item for item in plan.propositions if item.prop_id == "PROP-GAUSS-STEP")
    path = tmp_path / "gauss_step.py"
    path.write_text(step.script)
    receipt = ProofRunner(tmp_path).run_script(path)
    assert receipt.verified, receipt.stdout
