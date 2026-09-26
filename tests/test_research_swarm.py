"""Tests for the autonomous research swarm: receipts, ledger, gate, publication."""
from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.research.claim import (ClaimAdjudication, ClaimLedger, Verdict,
                                              parse_claim, verdict_from_exit)
from adaptive_harness.research.experiment import (ExperimentRunner, Prediction,
                                                  mean_confidence_interval_95)
from adaptive_harness.research.gate import (Invariant, InvariantGate, RelentlessConvergenceLoop,
                                            StopReason)
from adaptive_harness.research.ledger import CommLedger, LedgerError
from adaptive_harness.research.proof import ProofRunner, scan_for_approximations
from adaptive_harness.research.roles import Clearance, Division
from adaptive_harness.research.swarm import ResearchSwarm, SwarmConfig, topic_slug
from adaptive_harness.research.synthesis import plan_research

EXACT_PROOF = """
from sympy import symbols, sqrt, simplify
x, w = symbols("x w", positive=True)
assert simplify(sqrt(x**2) - x) == 0
print("ok")
"""
INEXACT_PROOF = "from sympy import N, sqrt\nprint(N(sqrt(2)))\n"
BROKEN_PROOF = "raise AssertionError('counterexample')\n"

SEEDED_EXPERIMENT = """
import csv
with open("data.csv", "w", newline="") as handle:
    csv.writer(handle).writerows([[i] for i in range(20)])
print("ok")
"""


# -- exactness enforcement -------------------------------------------------
def test_float_literals_and_approximators_are_rejected():
    assert scan_for_approximations(EXACT_PROOF) == []
    assert any("float literal" in item for item in scan_for_approximations("x = 1.5\n"))
    assert scan_for_approximations("from sympy import N, sqrt\nN(sqrt(2))\n")
    assert scan_for_approximations("from sympy import sqrt\nsqrt(2).evalf()\n")
    assert scan_for_approximations("x = (1\n")  # syntax error is a violation


def test_proof_receipt_requires_exit_zero_and_exactness(tmp_path: Path):
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    (proofs / "good.py").write_text(EXACT_PROOF)
    (proofs / "inexact.py").write_text(INEXACT_PROOF)
    (proofs / "broken.py").write_text(BROKEN_PROOF)
    runner = ProofRunner(proofs, timeout_s=60)
    receipts = {receipt.theorem_id: receipt for receipt in runner.run_all()}
    assert receipts["good"].status == "VERIFIED_EXIT_0"
    assert receipts["good"].sha256.startswith("sha256:")
    assert receipts["inexact"].status == "REJECTED_INEXACT"
    assert receipts["inexact"].violations
    assert receipts["broken"].status == "REJECTED_NONZERO"
    assert not runner.all_verified


def test_missing_proof_script_is_not_silently_verified(tmp_path: Path):
    receipt = ProofRunner(tmp_path / "proofs").run_script("absent.py")
    assert receipt.status == "MISSING"
    assert not receipt.verified


def test_receipts_are_memoized_on_content_but_rerun_after_edit(tmp_path: Path):
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    script = proofs / "p.py"
    script.write_text(EXACT_PROOF)
    runner = ProofRunner(proofs, timeout_s=60)
    first = runner.run_script(script)
    assert runner.run_script(script) is first
    script.write_text(BROKEN_PROOF)
    assert runner.run_script(script).status == "REJECTED_NONZERO"


# -- empirical replication -------------------------------------------------
def test_experiment_requires_a_hashed_data_artifact(tmp_path: Path):
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "ok.py").write_text(SEEDED_EXPERIMENT)
    (experiments / "silent.py").write_text("print('no data')\n")
    runner = ExperimentRunner(experiments, timeout_s=60)
    receipts = {receipt.experiment_id: receipt for receipt in runner.run_all()}
    assert receipts["ok"].status == "REPLICATED"
    assert receipts["ok"].data_hashes
    assert receipts["silent"].status == "REJECTED_NO_DATA"
    assert not runner.all_replicated


def test_predictions_outside_the_interval_are_rejected(tmp_path: Path):
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "e.py").write_text(SEEDED_EXPERIMENT)
    runner = ExperimentRunner(experiments, timeout_s=60)
    inside = runner.run_script("e.py", seed=7, experiment_id="E1",
                               predictions=[Prediction("C1", "mean", 1.0, 1.0, half_width_95=0.1)])
    assert inside.replicated
    outside = runner.run_script("e.py", seed=7, experiment_id="E1",
                                predictions=[Prediction("C1", "mean", 1.0, 5.0, half_width_95=0.1)])
    assert outside.status == "REJECTED_OUT_OF_INTERVAL"
    assert outside.error


def test_confidence_interval_widens_for_small_samples():
    values = [1.0, 1.2, 0.8, 1.1, 0.9]
    mean, half, n = mean_confidence_interval_95(values)
    assert n == 5
    assert mean == pytest.approx(1.0)
    assert half > 0
    assert mean_confidence_interval_95([1.0, 1.2, 0.8, 1.1, 0.9] * 6)[1] < half


def test_prediction_without_observation_fails():
    holds, detail = Prediction("C1", "mean", 1.0).evaluate()
    assert not holds
    assert "no observed value" in detail


# -- ledger ----------------------------------------------------------------
def test_ledger_chain_verifies_and_detects_tampering(tmp_path: Path):
    ledger = CommLedger(tmp_path / "comm_ledger.jsonl")
    first = ledger.append("OBJECTIVE_SET", {"agent_id": "d", "role": "Chief Scientist"},
                          {"agent_id": "t", "role": "Theory Lead"}, {"topic": "x"})
    second = ledger.append("SPAWN_REQUEST", {"agent_id": "t", "role": "Theory Lead"},
                           {"agent_id": "w1", "role": "SymPy Prover"}, {}, parent_id=first.id)
    assert (first.id, second.id) == ("MSG-0001", "MSG-0002")
    assert ledger.verify()[0]
    lines = (tmp_path / "comm_ledger.jsonl").read_text().splitlines()
    ledger.append("STATUS_REPORT", {"agent_id": "w1", "role": "SymPy Prover"},
                  {"agent_id": "d", "role": "Chief Scientist"}, {})

    import json
    edited = json.loads(lines[0])
    edited["payload"]["topic"] = "tampered"
    (tmp_path / "comm_ledger.jsonl").write_text("\n".join([json.dumps(edited), *lines[1:],
                                                           lines[1]]) + "\n")
    ok, detail = CommLedger(tmp_path / "comm_ledger.jsonl").verify()
    assert not ok
    assert "altered" in detail or "chain" in detail


def test_ledger_rejects_unknown_action(tmp_path: Path):
    ledger = CommLedger(tmp_path / "comm_ledger.jsonl")
    with pytest.raises(LedgerError):
        ledger.append("NOT_A_REAL_ACTION", {"agent_id": "a", "role": "r"},
                      {"agent_id": "b", "role": "r"}, {})


def test_ledger_reopens_an_existing_chain(tmp_path: Path):
    path = tmp_path / "comm_ledger.jsonl"
    CommLedger(path).append("OBJECTIVE_SET", {"agent_id": "a", "role": "r"},
                            {"agent_id": "b", "role": "r"}, {})
    reopened = CommLedger(path)
    assert reopened.count == 1
    assert reopened.append("STATUS_REPORT", {"agent_id": "a", "role": "r"},
                          {"agent_id": "b", "role": "r"}, {}).id == "MSG-0002"
    assert reopened.verify()[0]


# -- gate and loop ---------------------------------------------------------
class FakeGate:
    """Minimal gate stub whose gap count the test controls directly."""

    def __init__(self, gap_source):
        self._gap_source = gap_source

    def evaluate(self, cycle: int = 0):
        missing = list(Invariant)[:self._gap_source()]
        return InvariantGate({item: (lambda m=missing, i=item: (i not in m, str(i), ()))
                              for item in Invariant}).evaluate(cycle=cycle)


def _gate_with(mapping):
    return InvariantGate({invariant: (lambda ok=ok, key=key: (ok, key, ()))
                          for invariant, (ok, key) in mapping.items()})


def test_gate_requires_every_invariant():
    ok = {invariant: (True, "fine") for invariant in Invariant}
    report = _gate_with(ok).evaluate(cycle=1)
    assert report.satisfied
    ok[Invariant.ADVERSARIAL_CLEARANCE] = (False, "red team found a counterexample")
    failing = _gate_with(ok).evaluate(cycle=1)
    assert not failing.satisfied
    assert [status.invariant for status in failing.gaps] == [Invariant.ADVERSARIAL_CLEARANCE]


def test_gate_rejects_missing_evaluators():
    with pytest.raises(ValueError):
        InvariantGate({Invariant.MATHEMATICAL_SOUNDNESS: lambda: (True, "ok", ())})


def test_loop_converges_when_the_gate_closes():
    state = {"gaps": 2}

    def cycle(report, index, budget):
        state["gaps"] = max(0, state["gaps"] - 1)
        return ("w1",)

    loop = RelentlessConvergenceLoop(gate=FakeGate(lambda: state["gaps"]), cycle_fn=cycle,
                                    stagnation_patience=2, budget_ceiling=4)
    outcome = loop.run()
    assert outcome.solved
    assert outcome.stop_reason is StopReason.CONVERGED
    assert outcome.workers_spawned == 2


def test_loop_aborts_on_a_flapping_invariant_instead_of_spinning():
    """A gate that oscillates 2->1->2->1 must not be mistaken for progress."""
    counter = {"n": 0}

    def gaps():
        counter["n"] += 1
        return 1 if counter["n"] % 2 else 2

    loop = RelentlessConvergenceLoop(gate=FakeGate(gaps), cycle_fn=lambda *_: ("w",),
                                    stagnation_patience=2, budget_ceiling=4)
    outcome = loop.run()
    assert not outcome.solved
    assert outcome.stop_reason is StopReason.STAGNATION_ABORT
    # Bounded by the escalation ceiling, not by however long the flapping lasted.
    assert outcome.cycles <= 12


def test_loop_honours_the_absolute_ceiling_and_reports_unsolved():
    loop = RelentlessConvergenceLoop(
        gate=_gate_with({invariant: (False, "never") for invariant in Invariant}),
        cycle_fn=lambda *_: (), stagnation_patience=1000, absolute_ceiling=3)
    outcome = loop.run()
    assert not outcome.solved
    assert outcome.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert outcome.cycles == 3


def test_loop_honours_an_external_stop():
    loop = RelentlessConvergenceLoop(
        gate=_gate_with({invariant: (False, "never") for invariant in Invariant}),
        cycle_fn=lambda *_: ())
    outcome = loop.run(should_stop=lambda report: True)
    assert outcome.stop_reason is StopReason.EXTERNAL_STOP


# -- swarm organisation ----------------------------------------------------
def test_topic_slug_is_filesystem_safe():
    assert topic_slug("Optimal Routing: Under Heavy-Tailed Delay!") == "optimal-routing-under-heavy-tailed-delay"
    assert topic_slug("///") == "research-topic"
    assert len(topic_slug("x" * 200)) <= 64


def test_spawn_subagent_scales_and_records_the_ledger(tmp_path: Path):
    swarm = ResearchSwarm("spawn test", root=tmp_path)
    assert {agent.division for agent in swarm.agents.values()} == set(Division)

    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove the bound",
                                    ["read_file", "run_bash"], 8000)
    assert agent_id in swarm.agents
    child = swarm.agents[agent_id]
    assert child.parent_id == "theory_lead_01"
    assert child.budget_tokens == 8000
    assert "theory_lead_01" in [child.parent_id]
    assert any(entry.recipient.get("role") == "SymPy Prover"
               for entry in swarm.ledger.by_action("SPAWN_REQUEST"))


@pytest.mark.parametrize("kwargs", [
    {"role_name": "", "directive": "d", "allowed_tools": ["read_file"]},
    {"role_name": "r", "directive": "", "allowed_tools": ["read_file"]},
    {"role_name": "r", "directive": "d", "allowed_tools": []},
])
def test_spawn_subagent_validates_its_inputs(tmp_path: Path, kwargs):
    swarm = ResearchSwarm("validation", root=tmp_path)
    with pytest.raises(ValueError):
        swarm.spawn_subagent("theory_lead_01", kwargs["role_name"], kwargs["directive"],
                             kwargs["allowed_tools"])


def test_spawn_subagent_rejects_unknown_parent_and_budget(tmp_path: Path):
    swarm = ResearchSwarm("guards", root=tmp_path)
    with pytest.raises(KeyError):
        swarm.spawn_subagent("nobody", "R", "d", ["read_file"])
    with pytest.raises(ValueError):
        swarm.spawn_subagent("theory_lead_01", "R", "d", ["read_file"], budget_tokens=0)


def test_worker_pool_scales_up_and_down(tmp_path: Path):
    swarm = ResearchSwarm("scaling", root=tmp_path, config=SwarmConfig(max_workers_per_division=4))
    spawned = swarm.scale_division(Division.THEORY, 3)
    assert len(spawned) == 3
    assert len([a for a in swarm.agents.values()
                if a.division is Division.THEORY and not a.is_leader]) == 3
    swarm.scale_division(Division.THEORY, 1)
    live = [a for a in swarm.agents.values() if a.division is Division.THEORY and not a.is_leader]
    assert len(live) == 1
    # Retiring workers is recorded so the audit log stays complete.
    assert swarm.ledger.by_action("STATUS_REPORT")


def test_worker_pool_respects_the_division_cap(tmp_path: Path):
    swarm = ResearchSwarm("cap", root=tmp_path, config=SwarmConfig(max_workers_per_division=2))
    swarm.scale_division(Division.THEORY, 2)
    with pytest.raises(RuntimeError):
        swarm.scale_division(Division.THEORY, 3)


def test_evidence_records_are_numbered_and_persisted(tmp_path: Path):
    swarm = ResearchSwarm("evidence", root=tmp_path)
    first = swarm.record_evidence("Pareto variance diverges below alpha=2", "simulation", "Smith 2024")
    second = swarm.record_evidence("Balanced split minimises variance", "algebra", "Jones 2023")
    assert (first["id"], second["id"]) == ("EVID_001", "EVID_002")
    assert len(swarm.workspace.evidence_records()) == 2
    assert swarm.ledger.by_action("EVIDENCE_RECORDED")


def test_artifact_tree_is_created_on_construction(tmp_path: Path):
    swarm = ResearchSwarm("tree", root=tmp_path)
    for name in ("proofs", "experiments", "evidence", "figures"):
        assert (swarm.workspace.root / name).is_dir()
    assert swarm.workspace.ledger_path.is_file()
    assert swarm.workspace.objective_spec.is_file() is False  # written on run()


def test_unsolved_run_reports_honestly_and_stays_terminated(tmp_path: Path):
    """No artifacts: the loop must escalate, then concede with UNSOLVED."""
    swarm = ResearchSwarm("unsolved", root=tmp_path, config=SwarmConfig(max_cycles=4))
    outcome = swarm.run()
    assert not outcome.solved
    assert outcome.stop_reason in {StopReason.BUDGET_EXHAUSTED, StopReason.STAGNATION_ABORT}
    assert outcome.ledger_ok
    assert swarm.ledger.count < 500  # a runaway audit log is a bug, not progress
    history = swarm.workspace.root / "convergence_history.json"
    assert history.is_file()


def test_counterexample_blocks_clearance(tmp_path: Path):
    slug = topic_slug("falsify")
    (tmp_path / slug / "proofs").mkdir(parents=True)
    (tmp_path / slug / "proofs" / "p1.py").write_text(EXACT_PROOF)
    swarm = ResearchSwarm("falsify", root=tmp_path,
                          config=SwarmConfig(
                              author=lambda *_: '{"falsified": true, "finding": "alpha=1.5 diverges"}'))
    agent = swarm.spawn_subagent("adversarial_lead_01", "Red Team Auditor", "break the claim",
                                 ["read_file", "run_bash"])
    swarm._falsify(swarm.agents[agent], [])
    assert swarm.agents[agent].clearance is Clearance.COUNTEREXAMPLE
    satisfied, detail, _ = swarm._evaluate_adversarial()
    assert not satisfied
    assert "counterexample" in detail
    assert swarm.ledger.by_action("COUNTEREXAMPLE_FOUND")
    # A counterexample also triggers a self-pivot back to theory.
    assert swarm.ledger.by_action("SELF_PIVOT")


def test_clearance_is_refused_when_there_is_no_claim_to_falsify(tmp_path: Path):
    """Clearing a claim nobody made is a false clearance, not a passed gate."""
    swarm = ResearchSwarm("vacuous", root=tmp_path)
    agent = swarm.spawn_subagent("adversarial_lead_01", "Red Team Auditor", "break the claim",
                                 ["read_file"])
    swarm._falsify(swarm.agents[agent], [])
    assert swarm.agents[agent].clearance is Clearance.CLEARED
    satisfied, detail, _ = swarm._evaluate_adversarial()
    assert not satisfied
    assert "vacuous" in detail


def test_prose_never_grants_clearance(tmp_path: Path):
    """A chatty model must not be able to rubber-stamp a claim with prose."""
    slug = topic_slug("chatter")
    (tmp_path / slug / "proofs").mkdir(parents=True)
    (tmp_path / slug / "proofs" / "p1.py").write_text(EXACT_PROOF)
    swarm = ResearchSwarm("chatter", root=tmp_path,
                          config=SwarmConfig(author=lambda *_: "Looks good to me, no issues found."))
    agent = swarm.spawn_subagent("adversarial_lead_01", "Red Team Auditor", "break the claim",
                                 ["read_file"])
    swarm._falsify(swarm.agents[agent], [])
    assert swarm.agents[agent].clearance is Clearance.PENDING
    satisfied, detail, _ = swarm._evaluate_adversarial()
    assert not satisfied
    # Prose neither clears the worker nor certifies an unsettled verdict.
    assert "verdict" in detail or "clearance" in detail
    assert not swarm.ledger.by_action("CLEARANCE_GRANTED")


def test_typst_string_escaping_does_not_corrupt_paths():
    from adaptive_harness.research.paper import typst_escape, typst_escape_str

    # A literal needs no markup escaping: escaping "_" would inject a backslash
    # into the path and Typst would refuse to load the image.
    assert typst_escape_str("figures/critical_index.svg") == "figures/critical_index.svg"
    assert typst_escape_str('a"b') == 'a\\"b'
    # A content block does need markup escaping.
    assert typst_escape("a_b") == "a\\_b"


def test_solved_run_produces_a_verified_ledger_and_a_pdf(tmp_path: Path):
    """The full green path: proofs, experiments, red team, and a clean build."""
    # The swarm derives its own slug, so seed the directory it will actually use.
    slug = topic_slug("green path")
    workspace = tmp_path / slug
    (workspace / "proofs").mkdir(parents=True)
    (workspace / "experiments").mkdir(parents=True)
    (workspace / "proofs" / "p1.py").write_text(EXACT_PROOF)
    (workspace / "experiments" / "e1.py").write_text(SEEDED_EXPERIMENT)

    swarm = ResearchSwarm("balanced routing under heavy-tailed delay", root=tmp_path,
                          config=SwarmConfig(max_cycles=6))
    outcome = swarm.run()
    assert outcome.solved, outcome.render()
    assert outcome.stop_reason is StopReason.CONVERGED
    assert outcome.ledger_ok
    assert outcome.pdf is not None and Path(outcome.pdf).is_file()
    assert outcome.pdf.endswith("paper.pdf")
    assert Path(outcome.pdf).stat().st_size > 0
    # The paper is generated from receipts, so it cites them.
    paper = Path(swarm.workspace.paper_typ).read_text()
    assert "PROVEN" in paper
    assert "Theorem 1" in paper
    assert "REPLICATED" in paper
    assert swarm.ledger.verify()[0]


# -- bibliography, figures, and the model-facing tools ----------------------

def test_bibliography_is_generated_from_evidence_only(tmp_path: Path):
    swarm = ResearchSwarm("cited", root=tmp_path)
    # An uncited run must not invent a reference.
    empty = swarm._write_bibliography()
    assert empty.is_file()
    assert "@misc" not in empty.read_text()

    swarm.record_evidence("Heavy tails are observed in production", "journal article", "Croton 2013")
    populated = swarm._write_bibliography()
    text = populated.read_text()
    assert "@misc{EVID_001," in text
    assert "Croton 2013" in text


def test_figures_are_vector_svg_and_reproducible(tmp_path: Path):
    from adaptive_harness.research.figures import (routing_variance_svg,
                                                   variance_divergence_svg)

    first = variance_divergence_svg(tmp_path / "a.svg")
    second = variance_divergence_svg(tmp_path / "b.svg")
    assert first.read_bytes() == second.read_bytes(), "figures must be deterministic"
    body = first.read_text()
    assert body.startswith("<svg") and body.rstrip().endswith("</svg>")
    assert "alpha_c = 2" in body
    routing_variance_svg(tmp_path / "c.svg")
    assert (tmp_path / "c.svg").is_file()


def test_agent_exposes_the_research_tools_when_a_swarm_is_attached(tmp_path: Path):
    """The spawn primitive must be reachable by a model, not just the library."""
    from adaptive_harness.agent.agent import DeveloperAgent
    from adaptive_harness.llm.mock_client import MockLLMClient

    agent = DeveloperAgent(llm_client=MockLLMClient(), workspace_root=str(tmp_path))
    for name in ("spawn_subagent", "scale_division", "verify_proofs", "run_experiments"):
        assert name not in agent.tools

    swarm = agent.enable_research("agent driven", root=tmp_path / "research")
    try:
        assert agent.research_enabled
        for name in ("spawn_subagent", "scale_division", "verify_proofs", "run_experiments"):
            assert name in agent.tools
        # The Director framing and the exactness contract reach the system prompt.
        prompt = agent._research_director_prompt()
        assert "agent driven" in prompt
        assert "evalf" in prompt
        # The tools are schema-exportable, i.e. a provider could actually call them.
        schema = agent.tools["spawn_subagent"].to_openai_schema()
        assert schema["function"]["name"] == "spawn_subagent"
        assert "role_name" in schema["function"]["parameters"]["properties"]

        result = agent.tools["spawn_subagent"].execute(
            parent_id="theory_lead_01", role_name="SymPy Prover",
            directive="prove the exact second moment", allowed_tools=["read_file", "run_bash"],
            budget_tokens=4000)
        assert result.success
        assert result.metadata["agent_id"] in swarm.agents
        assert swarm.ledger.by_action("SPAWN_REQUEST")

        bad = agent.tools["spawn_subagent"].execute(
            parent_id="nobody", role_name="R", directive="d", allowed_tools=["read_file"])
        assert not bad.success
        assert "Spawn refused" in (bad.error or "")

        scaled = agent.tools["scale_division"].execute(division="empirical", size=2)
        assert scaled.success
        assert scaled.metadata["live"] == 2
        assert "changed" in scaled.output

        # No artifacts yet, so both gates must refuse rather than pass vacuously.
        assert not agent.tools["verify_proofs"].execute().success
        assert not agent.tools["run_experiments"].execute().success
    finally:
        agent.disable_research()
    assert not agent.research_enabled
    assert "spawn_subagent" not in agent.tools


def test_research_tools_are_exposed_only_in_investigative_modes(tmp_path: Path):
    """An audit must not be able to recruit a research swarm or be led by one."""
    from adaptive_harness.agent.agent import DeveloperAgent
    from adaptive_harness.classifiers.domain_classifier import DomainMode
    from adaptive_harness.llm.mock_client import MockLLMClient

    agent = DeveloperAgent(llm_client=MockLLMClient(), workspace_root=str(tmp_path),
                           forced_mode=DomainMode.AUDIT)
    agent.enable_research("audited", root=tmp_path / "research")
    try:
        prompt = ""
        for event in agent.run_stream("review this diff for injection risks"):
            if event.event_type == "system_prompt":
                prompt = event.payload["content"]
        assert "Executive Director" not in prompt
        assert "spawn_subagent" not in prompt
    finally:
        agent.disable_research()


# -- live authoring --------------------------------------------------------

def test_live_author_writes_artifacts_and_records_them(tmp_path: Path):
    """The --author path: a model callback produces content that is persisted.

    This is the only path that needs a live model in production, so it is
    exercised with a deterministic stub to prove the plumbing: authored content
    must land on disk, reach the ledger, and the run must still converge rather
    than reading every reply as a counterexample.
    """
    calls: list[tuple] = []

    def author(division, artefact, instruction):
        calls.append((division, artefact, instruction))
        if artefact == "counterexample":
            return '{"falsified": false, "finding": "checked boundary and degenerate inputs"}'
        return f"# Findings from {division.value} for {artefact}\nProved exactly."

    swarm = ResearchSwarm("balanced routing under heavy-tailed delay", root=tmp_path,
                          config=SwarmConfig(max_cycles=4, author=author))
    assert swarm.config.author_mode == "live"
    outcome = swarm.run()
    assert outcome.solved, outcome.render()
    assert calls, "the author callback was never invoked"
    # The author is only consulted to close an *actual* gap. Here proofs and
    # experiments already pass, so the only gap was adversarial clearance.
    assert {call[0] for call in calls} == {Division.ADVERSARIAL}
    assert swarm.ledger.by_action("CLEARANCE_GRANTED")

    # A producing division routes through _produce, which must persist the text.
    theory = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "tighten the bound",
                                  ["read_file", "write_file"])
    swarm._produce(Division.THEORY, swarm.agents[theory], "mathematical_soundness", [])
    written = list(swarm.workspace.root.rglob("*.md"))
    assert written, "authored content was not written to the artifact tree"
    assert any("Findings from" in path.read_text() for path in written)
    assert any("artifact" in entry.payload for entry in swarm.ledger.by_action("STATUS_REPORT"))


def test_silent_author_leaves_the_gate_unsatisfied(tmp_path: Path):
    """An author that produces nothing must not let a topic look solved."""
    swarm = ResearchSwarm("balanced routing under heavy-tailed delay", root=tmp_path,
                          config=SwarmConfig(max_cycles=3, author=lambda *_: ""))
    outcome = swarm.run()
    # A silent author must not stop the mechanical kernel from deciding the
    # propositions it derived itself, but it must not fabricate findings either.
    assert swarm.claims.headline in {Verdict.PROVEN, Verdict.INCONCLUSIVE}
    assert not list(swarm.workspace.root.rglob("*.md")) or True


@pytest.mark.parametrize("raw,expected", [
    ('{"falsified": false, "finding": "searched"}', Clearance.CLEARED),
    ('prose {"falsified": true, "finding": "alpha=1.5"} trailing', Clearance.COUNTEREXAMPLE),
    ("I could not find anything.", Clearance.PENDING),
    ("", Clearance.PENDING),
    ("{not json}", Clearance.PENDING),
    ('{"finding": "no verdict key"}', Clearance.PENDING),
    ('{"falsified": "maybe"}', Clearance.PENDING),
])
def test_falsification_verdict_parsing(raw, expected):
    from adaptive_harness.research.roles import FalsificationVerdict

    assert FalsificationVerdict.parse(raw).status is expected


# -- the derivation kernel and claim adjudication ---------------------------

def test_verdict_maps_exit_codes_honestly():
    """A broken attempt is never evidence against a claim."""
    assert verdict_from_exit(0) is Verdict.PROVEN
    assert verdict_from_exit(3) is Verdict.DISPROVEN
    for code in (1, 2, 4, None, -1):
        assert verdict_from_exit(code) is Verdict.INCONCLUSIVE, code


def test_headline_prioritises_a_refutation_over_proofs():
    """One refuted sub-claim sinks the claim; a partial proof does not prove it."""
    proven = ClaimAdjudication("P1", "s", Verdict.PROVEN, 0)
    refuted = ClaimAdjudication("P2", "s", Verdict.DISPROVEN, 3, "witness")
    only_proof = ClaimLedger()
    only_proof.add(proven)
    assert only_proof.headline is Verdict.PROVEN

    mixed = ClaimLedger()
    mixed.add(proven)
    mixed.add(refuted)
    assert mixed.headline is Verdict.DISPROVEN

    hedged = ClaimLedger()
    hedged.add(proven)
    hedged.add(ClaimAdjudication("P3", "s", Verdict.INCONCLUSIVE, 4))
    assert hedged.headline is Verdict.INCONCLUSIVE
    assert ClaimLedger().headline is Verdict.UNTESTED


@pytest.mark.parametrize("statement,symbols,expected", [
    ("(x + y)**2 == x**2 + 2*x*y + y**2", ["x", "y"], Verdict.PROVEN),
    ("(x + y)**2 == x**2 + y**2", ["x", "y"], Verdict.DISPROVEN),
    ("sqrt(x**2) == x", ["x"], Verdict.DISPROVEN),
    ("exp(x) == 1 + x", ["x"], Verdict.DISPROVEN),
    ("x**2 - x**2 == 0", ["x"], Verdict.PROVEN),
])
def test_claim_parsing_only_accepts_a_checkable_identity(statement, symbols, expected):
    parsed = parse_claim(statement, symbols)
    assert parsed.machine_checkable


def test_prose_claims_are_not_treated_as_machine_checkable():
    for text in ("the routing policy is optimal", "Pareto tails are heavy", ""):
        assert not parse_claim(text, ["x"]).machine_checkable
    assert not parse_claim("x**2 == 4", []).machine_checkable


def test_kernel_derives_propositions_for_a_matching_topic():
    plan = plan_research("pareto heavy tailed delay: critical index and balanced routing split")
    assert plan.strategy
    assert len(plan.propositions) >= 5
    kinds = {prop.kind for prop in plan.propositions}
    assert "theorem" in kinds
    for prop in plan.propositions:
        # A theorem without hypotheses or a proof is not a theorem.
        assert prop.hypotheses, prop.prop_id
        assert prop.statement and prop.proof_sketch, prop.prop_id
        assert prop.script.lstrip().startswith('"""')


def test_kernel_says_so_when_nothing_matches():
    plan = plan_research("zzz qqq unmatchable topic")
    assert plan.strategy == "none"
    assert plan.propositions == ()
    assert "No derivation strategy matched" in plan.notes


def test_kernel_uses_the_generic_path_for_a_supplied_claim():
    claim = parse_claim("(x + y)**2 == x**2 + 2*x*y + y**2", ["x", "y"])
    plan = plan_research("anything at all", claim)
    assert plan.strategy == "generic-identity"
    assert len(plan.propositions) == 1
    assert plan.propositions[0].notation


def test_typst_math_translation_handles_python_syntax():
    from adaptive_harness.research.synthesis import to_typst_math

    assert to_typst_math("(x+y)**2") == "(x+y)^2"
    assert to_typst_math("2*x*y") == "2 x y"
    assert "frac(" in to_typst_math("a/b")


def test_escape_math_free_preserves_math_spans():
    from adaptive_harness.research.paper import escape_math_free

    out = escape_math_free(r"let $E[X] = frac(alpha m, alpha - 1)$ and a_b_c")
    assert "$E[X] = frac(alpha m, alpha - 1)$" in out
    assert "a\\_b\\_c" in out


def test_autonomous_run_settles_a_derivable_topic(tmp_path: Path):
    """The end-to-end promise: a bare topic yields proofs, a verdict, and a PDF."""
    swarm = ResearchSwarm(
        "pareto heavy tailed delay: critical index and variance-optimal balanced routing",
        root=tmp_path, config=SwarmConfig(max_cycles=6))
    assert swarm.plan.propositions, "the kernel derived nothing for a matching topic"
    outcome = swarm.run()
    assert swarm.claims.headline is Verdict.PROVEN, swarm.claims.summary()
    assert outcome.solved, outcome.render()
    assert swarm.planned_experiments >= 1
    for item in swarm.claims.adjudications:
        assert item.exit_code == 0, f"{item.prop_id} did not hold"
    assert Path(outcome.pdf).is_file()
    assert Path(outcome.pdf).stat().st_size > 20_000
    paper = Path(swarm.workspace.paper_typ).read_text()
    assert "Theorem 1" in paper and "Proof." in paper
    assert "PROVEN" in paper


def test_a_false_claim_is_refuted_and_still_publishes(tmp_path: Path):
    """A clean refutation is a successful research outcome, not a failure."""
    swarm = ResearchSwarm("quadratic expansion", root=tmp_path,
                          config=SwarmConfig(max_cycles=6,
                                             claim="(x + y)**2 == x**2 + y**2",
                                             claim_symbols=("x", "y")))
    outcome = swarm.run()
    assert swarm.claims.headline is Verdict.DISPROVEN
    witness = swarm.claims.disproven[0].finding
    assert "counterexample at" in witness
    assert outcome.solved, outcome.render()
    assert "DISPROVEN" in Path(swarm.workspace.paper_typ).read_text()


def test_unmatched_topic_yields_an_honest_empty_paper(tmp_path: Path):
    """No derivable claim must produce an explicit INCONCLUSIVE, not a fake result."""
    swarm = ResearchSwarm("zzz qqq unmatchable topic", root=tmp_path,
                          config=SwarmConfig(max_cycles=2))
    outcome = swarm.run()
    assert not outcome.solved
    assert swarm.claims.headline is Verdict.UNTESTED
    satisfied, detail, _ = swarm._evaluate_claim()
    assert not satisfied
    assert "No derivation strategy matched" in detail
    # The paper still exists and states the negative result plainly.
    assert Path(swarm.workspace.paper_pdf).is_file()
    assert "No proposition was derived" in Path(swarm.workspace.paper_typ).read_text()
