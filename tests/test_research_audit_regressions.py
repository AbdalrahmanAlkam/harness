"""Regressions for research evidence that must not become a false verdict."""

from pathlib import Path
import json
from io import BytesIO

from adaptive_harness.research.experiment import ExperimentRunner
from adaptive_harness.research.swarm import ResearchSwarm, SwarmConfig
from adaptive_harness.research.claim import Verdict
from adaptive_harness.research.claim import Proposition, TopicPlan
from adaptive_harness.research.proof import ProofRunner
from adaptive_harness.research.synthesis import plan_research
from adaptive_harness.research.lean_gate import LeanProofGate
from adaptive_harness.research.roles import Clearance, Division
from adaptive_harness.research.lean_gate import LeanProofReceipt
from adaptive_harness.research.paper import PaperBuilder, PaperInputs
from adaptive_harness.research.proof import ProofReceipt
from adaptive_harness.research.experiment import ExperimentReceipt
from adaptive_harness.tools.lean import LeanToolchain
from adaptive_harness.tools.research import WebSearchTool


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


def test_proof_gate_refreshes_claim_after_worker_repair(tmp_path: Path):
    swarm = ResearchSwarm("Gauss sum of the first natural numbers", root=tmp_path,
                          config=SwarmConfig(formalize=False))
    assert swarm.plan.propositions
    first = swarm.plan.propositions[0]
    path = swarm.workspace.proof_dir / f"{first.prop_id.lower()}.py"
    path.write_text("x = 1.5\n")
    swarm._evaluate_proofs()
    assert swarm.claims.adjudications[0].verdict is Verdict.UNTESTED
    path.write_text(first.script)
    swarm._evaluate_proofs()
    assert swarm.claims.adjudications[0].verdict is Verdict.PROVEN


def test_gauss_induction_step_script_has_all_imports(tmp_path: Path):
    plan = plan_research("Gauss sum of the first natural numbers")
    step = next(item for item in plan.propositions if item.prop_id == "PROP-GAUSS-STEP")
    path = tmp_path / "gauss_step.py"
    path.write_text(step.script)
    receipt = ProofRunner(tmp_path).run_script(path)
    assert receipt.verified, receipt.stdout


def test_lean_gate_writes_a_hashed_sidecar_and_requires_a_theorem(tmp_path: Path):
    if not LeanToolchain().available:
        return
    gate = LeanProofGate(tmp_path / "lean")
    valid = gate.write("Valid", "theorem one_plus_one : 1 + 1 = 2 := by rfl\n")
    receipt = gate.verify_file(valid)
    assert receipt.certified
    sidecar = json.loads(valid.with_name(valid.name + ".receipt.json").read_text())
    assert sidecar["hash"] == receipt.sha256
    assert sidecar["status"] == "LEAN_VERIFIED"
    empty = gate.write("NoTheorem", "def answer : Nat := 42\n")
    assert gate.verify_file(empty).status == "LEAN_NO_THEOREM"


def test_paper_labels_each_evidence_tier(tmp_path: Path):
    proof = ProofReceipt("P1", "proofs/p1.py", "sha256:abc", "VERIFIED_EXIT_0", 0)
    lean = LeanProofReceipt("LEAN-PROOF-001", "l1", "proofs/lean/l1.lean",
                            "sha256:def", "LEAN_VERIFIED", theorems=("lemma_one",))
    experiment = ExperimentReceipt("E1", "experiments/e1.py", 7, "REPLICATED",
                                   {"data.csv": "sha256:123"}, 0)
    paper = PaperBuilder(tmp_path).render(PaperInputs(
        title="Evidence", abstract="Evidence", topic="evidence", proofs=(proof,),
        lean_receipts=(lean,), experiments=(experiment,),
        evidence=({"id": "EVID_001", "citation": "Primary source"},)))
    source = paper.read_text()
    for tag in ("[LEAN-001]", "[PROOF-001]", "[EXP-001]", "[EVID-001]"):
        assert tag in source


def test_live_director_authors_claims_without_builtin_synthesis(tmp_path: Path, monkeypatch):
    swarm = ResearchSwarm("unseen topic", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))
    assert not swarm.plan.propositions
    visited = []

    def fake_worker(agent, division, directive, *, success_criterion, target, **_kwargs):
        visited.append(agent.role_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.name == "claim_manifest.json":
            target.write_text(json.dumps({"claims": [{"id": "PROP-01", "name": "Identity",
                "statement": "x + 0 = x", "hypotheses": ["x is an integer"],
                "sympy_expression": "x + 0 == x",
                "lean_statement": "example (x : Nat) : x + 0 = x := Nat.add_zero x"}],
                "empirical_required": False}))
            swarm.workspace.objective_spec.write_text("# Explicit objective\n")
        else:
            target.write_text("# Assignment\n")
        return {"ran": True, "success": True, "tool_calls": 2}

    monkeypatch.setattr(swarm, "_run_worker", fake_worker)
    swarm._prepare_live_research("prove a new identity")
    assert visited[0] == "Executive Director"
    assert len(visited) == 6  # Director and all five division leads
    assert swarm.plan.strategy == "llm-authored"
    assert swarm.plan.propositions[0].statement == "x + 0 = x"
    assert not list(swarm.workspace.proof_dir.glob("*.py"))


def test_live_experiment_requires_fresh_interval_and_csv(tmp_path: Path):
    runner = ExperimentRunner(tmp_path, require_predictions=True)
    (tmp_path / "e.py").write_text("open('data.csv', 'w').write('value\\n1\\n')\n")
    receipt = runner.run_script("e.py", seed=7)
    assert receipt.status == "REJECTED_NO_PREDICTIONS"
    (tmp_path / "e.py").write_text(
        "import json\n"
        "open('data.csv', 'w').write('value\\n1\\n')\n"
        "json.dump([dict(claim_id='C1', description='mean', predicted=1.0, "
        "observed=1.0, half_width_95=0.1, relative=False)], "
        "open('e.predictions.json', 'w'))\n")
    assert runner.run_script("e.py", seed=7).replicated


def test_live_literature_search_uses_public_metadata_without_brave_key(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    payload = {"message": {"items": [{"title": ["A checked paper"],
                                      "DOI": "10.1234/example"}]}}
    monkeypatch.setattr("adaptive_harness.tools.research.urlopen",
                        lambda request, timeout: BytesIO(json.dumps(payload).encode()))
    result = WebSearchTool(allow_public_metadata=True).execute("Pareto mean", count=2)
    assert result.success
    assert "A checked paper" in result.output
    assert "https://doi.org/10.1234/example" in result.output
    assert "metadata only" in result.output


def test_live_lean_gate_rejects_a_weaker_formal_theorem(tmp_path: Path):
    if not LeanToolchain().available:
        return
    swarm = ResearchSwarm("formal scope", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))
    swarm.plan = TopicPlan("formal scope", (Proposition(
        "PROP-01", "theorem", "Claim", statement="one plus one is two",
        lean_statement="1 + 1 = 2"),), strategy="llm-authored")
    (swarm.workspace.proof_dir / "prop-01.py").write_text("assert 1 + 1 == 2\n")
    (swarm.workspace.proof_dir / "prop-01.md").write_text("Exact arithmetic.\n")
    (swarm.workspace.lean_dir / "prop-01.lean").write_text(
        "theorem unrelated : 2 + 2 = 4 := by rfl\n")
    swarm._evaluate_proofs()
    ok, detail, _ = swarm._evaluate_lean_proofs()
    assert not ok
    assert "does not assert" in detail
    swarm.plan = TopicPlan("formal scope", (Proposition(
        "PROP-01", "theorem", "Claim", statement="one plus one is two",
        lean_statement="theorem one_plus_one : 1 + 1 = 2 := by rfl"),),
        strategy="llm-authored")
    (swarm.workspace.lean_dir / "prop-01.lean").write_text(
        "theorem one_plus_one : 1 + 1 = 2 := by rfl\n")
    ok, detail, _ = swarm._evaluate_lean_proofs()
    assert ok, detail


def test_live_counterexample_is_reassessed_after_artifact_repair(tmp_path: Path):
    for decision in (Clearance.COUNTEREXAMPLE, Clearance.CLEARED):
        swarm = ResearchSwarm("repair", root=tmp_path / decision.value,
                              config=SwarmConfig(llm_client_factory=lambda: object()))
        proof = swarm.workspace.proof_dir / "p.py"
        proof.write_text("assert False\n")
        worker_id = swarm.spawn_subagent("adversarial_lead_01", "Falsifier", "test the claim",
                                          ("read_file", "run_bash"))
        worker = swarm.agents[worker_id]
        worker.clearance = decision
        decisions = (swarm._objection_artifacts if decision is Clearance.COUNTEREXAMPLE
                     else swarm._clearance_artifacts)
        decisions[worker_id] = swarm._artifact_fingerprint()
        proof.write_text("assert True\n")
        swarm._evaluate_adversarial()
        assert worker.clearance is Clearance.PENDING
        assert worker_id not in decisions


def _unsolved_outcome(swarm: ResearchSwarm):
    """A realistic unsolved outcome, so the report is exercised on real data."""
    from adaptive_harness.research.gate import (ConvergenceOutcome, GateReport, Invariant,
                                                InvariantStatus, StopReason)
    report = GateReport(
        statuses=(InvariantStatus(invariant=Invariant.DOCUMENT_INTEGRITY, satisfied=False,
                                 detail="paper.pdf did not build cleanly"),
                  InvariantStatus(invariant=Invariant.MATHEMATICAL_SOUNDNESS, satisfied=False,
                                 detail="no proof script exists yet")),
        fingerprint="fp", cycle=1)
    return ConvergenceOutcome(False, StopReason.STAGNATION_ABORT, 1, report, (), 0)


def test_unsolved_live_paper_displays_terminal_status(tmp_path: Path):
    swarm = ResearchSwarm("unfinished", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))
    swarm.workspace.paper_typ.write_text("= Partial result\n")
    swarm._republish(_unsolved_outcome(swarm))
    delivered = swarm.workspace.paper_typ.read_text()
    assert "Research status: UNSOLVED" in delivered
    # The author's draft is preserved, not silently dropped.
    assert swarm.workspace.root.joinpath("paper_draft.typ").read_text() == "= Partial result\n"
    # And the delivered document is diagnostic rather than a draft.
    assert "Research progress report" in delivered
    assert "Which invariants failed" in delivered
    assert "document_integrity" in delivered and "paper.pdf did not build cleanly" in delivered
    assert "mathematical_soundness" in delivered and "no proof script exists yet" in delivered


def test_failed_live_author_leaves_an_honest_progress_pdf(tmp_path: Path):
    swarm = ResearchSwarm("unfinished", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))
    swarm._republish(_unsolved_outcome(swarm))
    assert "not a proof" in swarm.workspace.paper_typ.read_text()
    assert swarm.workspace.paper_pdf.is_file()


def test_invalid_live_draft_is_preserved_and_replaced_by_progress_pdf(tmp_path: Path):
    swarm = ResearchSwarm("unfinished", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))
    swarm.workspace.paper_typ.write_text("#let =\n")
    swarm._republish(_unsolved_outcome(swarm))
    assert (swarm.workspace.root / "paper_draft.typ").read_text() == "#let =\n"
    assert "Research status: UNSOLVED" in swarm.workspace.paper_typ.read_text()
    assert swarm.workspace.paper_pdf.is_file()


def test_missing_lead_assignment_is_visible_to_later_workers(tmp_path: Path, monkeypatch):
    swarm = ResearchSwarm("assignment", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))

    def only_director_writes(agent, division, directive, *, success_criterion, target, **_kwargs):
        if agent.role_name == "Executive Director":
            swarm.workspace.objective_spec.write_text("# Objective\n")
            target.write_text(json.dumps({"claims": [{"id": "PROP-01", "name": "Claim",
                "statement": "one plus one is two", "sympy_expression": "1 + 1 == 2",
                "lean_statement": "example : (1:Nat) + 1 = 2 := rfl"}]}))
        return {"ran": True, "success": False}

    monkeypatch.setattr(swarm, "_run_worker", only_director_writes)
    swarm._prepare_live_research("prove arithmetic")
    assignment = swarm.workspace.root / "formal_assignments.md"
    assert assignment.is_file()
    assert "lead did not leave a written assignment" in assignment.read_text()
    assert "Lean target: `example : (1:Nat) + 1 = 2 := rfl`" in assignment.read_text()


def test_a_claim_without_an_exact_expression_is_rejected_not_attempted(tmp_path: Path,
                                                                       monkeypatch):
    """A claim the exact computer cannot decide cannot be settled.

    Accepting one produced 15,000-line attempts to build a Lean evaluator in Python,
    every one of which failed. The manifest is now rejected, and the reason is
    recorded so the Director can be told what was wrong.
    """
    swarm = ResearchSwarm("undecidable", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))

    def director_writes_a_lean_flavoured_claim(agent, division, directive, *,
                                               success_criterion, target, **_kwargs):
        if agent.role_name == "Executive Director":
            swarm.workspace.objective_spec.write_text("# Objective\n")
            target.write_text(json.dumps({"claims": [{
                "id": "PROP-01", "name": "Reduction",
                "statement": "Nat.succ n + m reduces definitionally to Nat.succ (n + m)",
                "lean_statement": "example (n m : Nat) : Nat.succ n + m = Nat.succ (n + m) := rfl"}]}))
        return {"ran": True, "success": True, "tool_calls": 1}

    monkeypatch.setattr(swarm, "_run_worker", director_writes_a_lean_flavoured_claim)
    swarm._prepare_live_research("prove a reduction lemma")
    assert swarm.plan.propositions == ()
    assert "sympy_expression" in swarm.plan.notes
    # The leads are never dispatched, because there is nothing to decompose.
    assert not list(swarm.workspace.root.glob("*_assignments.md"))


def test_a_claim_without_a_lean_statement_is_also_rejected(tmp_path: Path, monkeypatch):
    swarm = ResearchSwarm("no lean", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))

    def director_writes(agent, division, directive, *, success_criterion, target, **_kwargs):
        if agent.role_name == "Executive Director":
            swarm.workspace.objective_spec.write_text("# Objective\n")
            target.write_text(json.dumps({"claims": [{
                "id": "PROP-01", "name": "Identity", "statement": "two plus two is four",
                "sympy_expression": "2 + 2 == 4"}]}))
        return {"ran": True, "success": True, "tool_calls": 1}

    monkeypatch.setattr(swarm, "_run_worker", director_writes)
    swarm._prepare_live_research("prove arithmetic")
    assert swarm.plan.propositions == ()
    assert "lean_statement" in swarm.plan.notes


def test_the_kernel_writes_the_decider_from_the_directors_expression(tmp_path: Path,
                                                                    monkeypatch):
    """The computational verdict is SymPy's, not a language model's."""
    import subprocess
    import sys

    swarm = ResearchSwarm("kernel decides", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))

    def director_writes(agent, division, directive, *, success_criterion, target, **_kwargs):
        if agent.role_name == "Executive Director":
            swarm.workspace.objective_spec.write_text("# Objective\n")
            target.write_text(json.dumps({"claims": [{
                "id": "PROP-01", "name": "Identity",
                "statement": "the sum of two and two is four",
                "sympy_expression": "2 + 2 == 4",
                "lean_statement": "example : (2:Nat) + 2 = 4 := rfl"}]}))
        return {"ran": True, "success": True, "tool_calls": 1}

    monkeypatch.setattr(swarm, "_run_worker", director_writes)
    swarm._prepare_live_research("prove arithmetic")
    assert [item.prop_id for item in swarm.plan.propositions] == ["PROP-01"]
    assert swarm.plan.propositions[0].exactly_decidable
    swarm._synthesize()
    script = swarm.workspace.proof_dir / "prop-01.py"
    assert script.is_file()
    completed = subprocess.run([sys.executable, str(script)], capture_output=True,
                               text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "residual is 0" in completed.stdout
    kernel_entries = [entry for entry in swarm.ledger.by_action("TASK_ASSIGNED")
                      if entry.recipient["agent_id"] == "derivation_kernel"]
    assert kernel_entries, "the kernel's decider emission was not recorded"
    assert kernel_entries[0].payload["kernel_decided"] == ["PROP-01"]
    assert kernel_entries[0].payload["scripts_written"] == ["prop-01.py"]


def test_live_artifact_worker_writes_target_before_verification_loop(tmp_path: Path, monkeypatch):
    swarm = ResearchSwarm("write first", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))
    swarm.plan = TopicPlan("write first", (Proposition(
        "PROP-01", "theorem", "Claim", statement="1 + 1 = 2",
        lean_statement="1 + 1 = 2"),), strategy="llm-authored")
    worker_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove claim",
                                      ("write_file", "run_bash"))
    calls = []

    def fake_worker(agent, division, directive, *, success_criterion, target, **kwargs):
        calls.append(kwargs)
        if kwargs.get("tool_names_override") == ("write_file",):
            target.write_text("assert 1 + 1 == 2\n")
        return {"ran": True, "success": True, "tool_calls": 1}

    monkeypatch.setattr(swarm, "_run_worker", fake_worker)
    swarm._produce(Division.THEORY, swarm.agents[worker_id],
                   "mathematical_soundness", [])
    assert calls[0]["tool_names_override"] == ("write_file",)
    # The second stage runs the real tool loop, not the write-only pass.
    assert "tool_names_override" not in calls[1]
    assert "max_steps_override" not in calls[1]
    assert (swarm.workspace.proof_dir / "prop-01.py").is_file()


def _ladder_swarm(tmp_path: Path, monkeypatch) -> ResearchSwarm:
    """A manifest shaped as a ladder: one conjecture, decidable claims beneath."""
    swarm = ResearchSwarm("open problem", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))

    def director_writes(agent, division, directive, *, success_criterion, target, **_kwargs):
        if agent.role_name == "Executive Director":
            swarm.workspace.objective_spec.write_text("# Objective\n")
            target.write_text(json.dumps({"claims": [
                {"id": "C-01", "kind": "conjecture", "name": "SNC",
                 "statement": "every 2-cycle-free digraph has a vertex v with |N++(v)| >= |N+(v)|",
                 "hypotheses": []},
                {"id": "T-01", "kind": "theorem", "name": "small cases",
                 "statement": "SNC holds for every 2-cycle-free digraph on at most 3 vertices",
                 "hypotheses": [], "sympy_expression": "2 + 2 == 4",
                 "lean_statement": "example : (2:Nat) + 2 = 4 := rfl"}]}))
        return {"ran": True, "success": True, "tool_calls": 1}

    monkeypatch.setattr(swarm, "_run_worker", director_writes)
    swarm._prepare_live_research("attack the open problem")
    return swarm


def test_an_open_problem_is_accepted_as_the_target_not_rejected(tmp_path: Path, monkeypatch):
    """Pointing the run at an open problem must not be a validation failure."""
    swarm = _ladder_swarm(tmp_path, monkeypatch)
    assert [item.prop_id for item in swarm.plan.propositions] == ["C-01", "T-01"]
    assert [item.prop_id for item in swarm.goals] == ["C-01"]
    assert [item.prop_id for item in swarm.decidable] == ["T-01"]


def test_a_conjecture_is_never_adjudicated_or_decided(tmp_path: Path, monkeypatch):
    """The top rung must never acquire a verdict, least of all 'proven'."""
    swarm = _ladder_swarm(tmp_path, monkeypatch)
    swarm._synthesize()
    # No decider is written for the conjecture.
    assert not (swarm.workspace.proof_dir / "c-01.py").exists()
    assert (swarm.workspace.proof_dir / "t-01.py").exists()
    ledger = swarm._adjudicate()
    assert [item.prop_id for item in ledger.adjudications] == ["T-01"]
    assert not any(item.prop_id == "C-01" for item in ledger.adjudications)


def test_progress_on_a_conjecture_is_reported_as_sub_claim_progress(tmp_path: Path,
                                                                    monkeypatch):
    """A run that settles a sub-claim must not imply it settled the problem."""
    swarm = _ladder_swarm(tmp_path, monkeypatch)
    swarm._synthesize()
    swarm._adjudicate()
    ok, detail, _ = swarm._evaluate_claim()
    assert ok, detail
    assert "1 open problem(s) stated as the target (C-01)" in detail
    assert "not a resolution" in detail
    text = swarm._progress_report(_unsolved_outcome(swarm))
    assert "Open problem under investigation: SNC" in text
    assert "Nothing in this report resolves it" in text
    assert "What remains open" in text
    assert "C-01" in text


def test_an_unkinded_claim_is_a_theorem_not_an_open_problem(tmp_path: Path, monkeypatch):
    """Regression: defaulting an unkinded claim to 'conjecture' would let any
    manifest that omits a kind escape validation and be carried as the target."""
    swarm = ResearchSwarm("default kind", root=tmp_path,
                          config=SwarmConfig(llm_client_factory=lambda: object()))

    def director_writes(agent, division, directive, *, success_criterion, target, **_kwargs):
        if agent.role_name == "Executive Director":
            swarm.workspace.objective_spec.write_text("# Objective\n")
            target.write_text(json.dumps({"claims": [
                {"id": "T-01", "name": "No kind", "statement": "unstated form"}]}))
        return {"ran": True, "success": True, "tool_calls": 1}

    monkeypatch.setattr(swarm, "_run_worker", director_writes)
    swarm._prepare_live_research("go")
    assert swarm.plan.propositions == ()
    assert "sympy_expression" in swarm.plan.notes
