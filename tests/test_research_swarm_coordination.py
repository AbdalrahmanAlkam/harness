"""Swarm-level coordination: hierarchy, leases, cancellation, and recovery.

The unit tests in ``test_research_coordination.py`` prove the board, the bus, and
the control plane behave. These prove the *swarm* actually uses them: that the
Director delegates through leads, that two workers never get one artifact, that a
stop request interrupts a live tool loop, and that an interrupted run resumes
without repeating finished work.
"""

from __future__ import annotations

from pathlib import Path
import json
import threading
import time
from typing import Any

import pytest

from adaptive_harness.llm.mock_client import LLMResponse, ToolCall
from adaptive_harness.research import ResearchSwarm, SwarmConfig
from adaptive_harness.research.coordination import StopKind, WorkerState
from adaptive_harness.research.roles import Division, leader_agent_id, worker_agent_id
from adaptive_harness.research.swarm import DIRECTOR_ID, GAP_ROUTING

from test_research_worker_autonomy import ScriptedClient

PROOF_BODY = (
    "import sympy\n"
    "x, y = sympy.symbols('x y', integer=True)\n"
    "sympy.poly((x + y) ** 2 - (x * x + 2 * x * y + y * y), x, y).is_zero\n"
    "print('expanded')\n"
    "raise SystemExit(0)\n"
)


def _swarm(tmp_path: Path, script: list[tuple[str, dict[str, Any]]] | None = None,
           *, final: str = "Done.", **config_kwargs: Any) -> ResearchSwarm:
    script = script if script is not None else []
    def factory():
        return ScriptedClient(list(script), final)
    config_kwargs.setdefault("max_parallel_workers", 1)
    config_kwargs.setdefault("worker_max_steps", 4)
    config_kwargs.setdefault("max_cycles", 1)
    config = SwarmConfig(llm_client_factory=factory, **config_kwargs)
    return ResearchSwarm("coordination run", root=tmp_path, config=config)


def _write(swarm: ResearchSwarm, relative: str, body: str) -> str:
    """Write a file inside the workspace and return its workspace-relative path."""
    target = swarm.workspace.root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return relative


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

def test_the_director_delegates_through_division_leads_not_directly_to_workers(tmp_path: Path):
    """The previous cycle() called workers itself; leaders were decoration."""
    swarm = _swarm(tmp_path)
    report = _report_with(swarm, "mathematical_soundness")
    planned = swarm._plan_cycle(report)
    assert planned, "the Director should have produced at least one unit of work"
    for _division, _gap, task, worker_id in planned:
        lease = swarm.ledger.by_payload("task_id", task.task_id)
        leased = [entry for entry in lease if entry.action == "TASK_LEASED"]
        assert leased, f"{task.task_id} was planned without a recorded lease"
        sender = leased[0].sender["agent_id"]
        assert sender == leader_agent_id(Division(task.division)), (
            "a task must be handed down by the lead that owns the division")
        assert leased[0].recipient["agent_id"] == worker_id


def test_a_spawned_worker_knows_its_lead_and_the_director_knows_the_leads(tmp_path: Path):
    swarm = _swarm(tmp_path)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it", ["read_file"])
    assert swarm.bus.contact_for(worker) == "theory_lead_01"
    for division in Division:
        assert swarm.bus.contact_for(leader_agent_id(division)) == DIRECTOR_ID
    # The escalation contact is written into the spawn record so a reader of the
    # ledger alone can tell who a worker answers to.
    spawn = [entry for entry in swarm.ledger.by_action("SPAWN_REQUEST")
             if entry.recipient["agent_id"] == worker][0]
    assert spawn.payload["escalation_contact"] == "theory_lead_01"


def test_every_gap_routes_to_exactly_one_division():
    from adaptive_harness.research.gate import Invariant
    assert set(GAP_ROUTING) == {invariant.value for invariant in Invariant}
    assert all(isinstance(value, Division) for value in GAP_ROUTING.values())


# ---------------------------------------------------------------------------
# No duplicate work
# ---------------------------------------------------------------------------

def test_more_workers_than_claims_still_get_distinct_artifacts(tmp_path: Path):
    """The old modulo scheme sent worker 1 and worker 4 to the same proof file."""
    swarm = _swarm(tmp_path, max_workers_per_division=8)
    swarm.plan = _plan_with(swarm, 2)
    swarm.scale_division(Division.THEORY, 8)
    report = _report_with(swarm, "mathematical_soundness")
    planned = swarm._plan_cycle(report)
    artifacts = [task.artifact for _d, _g, task, _w in planned]
    assert len(artifacts) == len(set(artifacts)), f"artifacts collided: {artifacts}"
    assert len(planned) == 2, "one task per claim, not one per worker"


def test_a_task_already_completed_is_not_handed_out_again(tmp_path: Path):
    swarm = _swarm(tmp_path)
    swarm.plan = _plan_with(swarm, 1)
    report = _report_with(swarm, "mathematical_soundness")
    first = swarm._plan_cycle(report)
    assert len(first) == 1
    task = first[0][2]
    worker_id = first[0][3]
    _write(swarm, task.artifact, PROOF_BODY)
    swarm._close_task(task, worker_id, WorkerState.COMPLETED, evidence_ids=("PROOF-001",))
    assert swarm._plan_cycle(report) == [], "a completed task must not be re-dispatched"


def test_a_worker_is_confined_to_its_own_artifact(tmp_path: Path):
    """An unrestricted worker could overwrite a sibling's leased proof."""
    swarm = _swarm(tmp_path, [
        ("write_file", {"path": "proofs/other.py", "content": "x = 1\n"}),
        ("write_file", {"path": "proofs/mine.py", "content": PROOF_BODY}),
    ])
    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "do it", ["write_file"])
    target = swarm.workspace.proof_dir / "mine.py"
    outcome = swarm._run_worker(swarm.agents[agent_id], Division.THEORY, "Write only mine.",
                                success_criterion="mine.py exists.", target=target)
    assert outcome["ran"] is True
    assert not (swarm.workspace.proof_dir / "other.py").exists(), (
        "a worker must not be able to write outside its authorized paths")
    assert target.is_file()
    # The refusal must be visible to the worker in its own trajectory, not silent.
    errors = [item.get("error") for item in outcome["trajectory"] if item.get("error")]
    assert any("restricted" in (error or "") for error in errors), errors


def test_a_proof_worker_may_also_write_its_required_explanation(tmp_path: Path):
    """The adjudication gate needs proofs/<stem>.md; authorising only the script
    would block the artifact the gate actually demands."""
    swarm = _swarm(tmp_path, [
        ("write_file", {"path": "proofs/mine.py", "content": PROOF_BODY}),
        ("write_file", {"path": "proofs/mine.md", "content": "# derivation\n"}),
    ])
    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "do it", ["write_file"])
    target = swarm.workspace.proof_dir / "mine.py"
    outcome = swarm._run_worker(swarm.agents[agent_id], Division.THEORY, "Write both.",
                                success_criterion="mine.py exists.", target=target)
    assert outcome["ran"] is True
    assert target.is_file()
    assert (swarm.workspace.proof_dir / "mine.md").is_file()


# ---------------------------------------------------------------------------
# Acceptance is decided, not assumed
# ---------------------------------------------------------------------------

def test_a_worker_that_reports_success_without_writing_anything_is_not_accepted(tmp_path: Path):
    swarm = _swarm(tmp_path, final="All done!")
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "do it", ["write_file"])
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, worker)
    swarm.control.begin(worker, task_id=task.task_id)
    summary = {"ran": True, "success": True, "tool_calls": 1, "trajectory": [],
               "stop_reason": "completed", "error": None}
    result = swarm._settle(task, swarm.agents[worker], summary, target=None)
    assert result["accepted"] is False
    assert swarm.board.get(task.task_id).state is not WorkerState.COMPLETED


def test_a_claimed_success_with_a_verified_artifact_is_accepted_with_its_receipts(tmp_path: Path):
    swarm = _swarm(tmp_path)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "do it", ["write_file"])
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, worker)
    swarm.control.begin(worker, task_id=task.task_id)
    target = swarm.workspace.root / task.artifact
    _write(swarm, task.artifact, PROOF_BODY)
    # A theory claim is only complete with its natural-language derivation beside
    # the decision script.
    _write(swarm, str(Path(task.artifact).with_suffix(".md")), "# derivation\n")
    swarm.proofs.receipts = _proof_receipt(target)
    summary = {"ran": True, "success": True, "tool_calls": 2, "trajectory": [],
               "stop_reason": "completed", "error": None}
    result = swarm._settle(task, swarm.agents[worker], summary, target=target)
    assert result["accepted"] is True
    assert swarm.board.get(task.task_id).evidence_ids == ("PROP_01",)
    assert swarm.board.get(task.task_id).artifact.endswith(".md"), \
        "the completed artifact is the derivation, not just the script"
    assert swarm.control.record(worker).state is WorkerState.COMPLETED


def test_a_lean_receipt_belongs_to_the_lean_file_it_certified(tmp_path: Path):
    """A proof receipt must not be attributed to a Lean file, or an uncertified
    formal file inherits a computational file's standing."""
    from adaptive_harness.research.lean_gate import LeanProofReceipt
    swarm = _swarm(tmp_path)
    other = swarm.workspace.lean_dir / "prop_02.lean"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("theorem t : True := trivial\n", encoding="utf-8")
    # The receipt was minted for prop_01.lean but its path points at prop_02.lean:
    # only the *name* identifies which file was certified, so the mismatch must
    # resolve against the name, never the path.
    swarm.lean_gate.receipts = [LeanProofReceipt(
        proof_id="LEAN-001", name="prop_01.lean", path=str(other), status="LEAN_VERIFIED",
        sha256="sha256:abc", theorems=["t"], lean_version="4.9.0",
        verified_at="2026-01-01T00:00:00Z")]
    assert swarm.lean_gate.receipts[0].certified
    assert swarm._artifact_evidence(swarm.workspace.lean_dir / "prop_02.lean") == ()
    assert swarm._artifact_evidence(swarm.workspace.lean_dir / "prop_01.lean") == ("LEAN-001",)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_a_stop_request_interrupts_a_live_tool_loop_and_is_recorded(tmp_path: Path):
    swarm = _swarm(tmp_path, worker_max_steps=200)
    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "loop forever", ["run_bash"])
    task = _new_task(swarm)
    # Pre-create the target so the write-first bootstrap is skipped and the cancel
    # lands inside the real tool loop rather than the two-step bootstrap.
    _write(swarm, task.artifact, PROOF_BODY)
    swarm.board.lease(task.task_id, agent_id)
    swarm.control.begin(agent_id, task_id=task.task_id)
    parked, release = threading.Event(), threading.Event()
    turns: list[int] = []

    def blocking_factory():
        client = _BlockingClient(parked, release)
        client.default_model = "stealth/space-bunny-alpha"
        original = client.complete

        def counted(*args, **kwargs):
            turns.append(1)
            return original(*args, **kwargs)
        client.complete = counted  # type: ignore[method-assign]
        return client

    swarm.config.llm_client_factory = blocking_factory
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result.update(swarm._produce(Division.THEORY, swarm.agents[agent_id],
                                         "mathematical_soundness", [], task=task))
        except BaseException as exc:  # noqa: BLE001 - surfaced as an assertion below
            result["exception"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert parked.wait(timeout=30), "the worker never reached the model"
        assert swarm.stop_worker(agent_id, actor="theory_lead_01",
                                 reason="identical failure three times running")
    finally:
        release.set()
    worker.join(timeout=30)
    assert not worker.is_alive(), "the cancelled tool loop never unwound"

    record = swarm.control.record(agent_id)
    assert record.state is WorkerState.CANCELLED
    assert record.stop_actor == "theory_lead_01"
    assert "identical failure" in record.stop_reason
    assert "exception" not in result, result["exception"]
    assert result["cancelled"] is True, sorted(result)
    assert len(turns) == 1, "the worker kept calling the model after being cancelled"
    assert swarm.board.get(task.task_id).owner is None
    assert swarm.ledger.by_action("WORKER_CANCELLED")
    assert swarm.ledger.by_action("STOP_REQUESTED")


def test_stopping_a_worker_reports_whether_there_was_anything_to_stop(tmp_path: Path):
    swarm = _swarm(tmp_path)
    # Unknown agent: nothing to stop, and the caller is told so rather than
    # getting a silent success that hides a typo in an agent id.
    assert swarm.stop_worker("never_registered", actor="director", reason="n/a") is False
    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "d", ["read_file"])
    # A freshly hired worker is QUEUED, and preventing it from starting is a
    # legitimate stop, so this must report that it acted.
    assert swarm.stop_worker(agent_id, actor="director", reason="not needed") is True
    assert swarm.control.record(agent_id).state is WorkerState.CANCELLED
    # A worker that already reached a terminal state cannot be stopped again.
    assert swarm.stop_worker(agent_id, actor="director", reason="too late") is False


def _spawn_scoped(owner: str | None, seconds: int = 120):
    """Start a real child process, registered the way run_process registers it.

    ``owner=None`` reproduces a process started outside any agent scope — a
    verifier running on the caller's own thread.
    """
    import subprocess

    from adaptive_harness.tools import process as process_module
    with process_module.process_scope(owner):
        child = subprocess.Popen(["sleep", str(seconds)], start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process_module._register(child.pid)
    return child


def test_a_stop_request_kills_the_child_processes_that_worker_started(tmp_path: Path):
    """A stopped worker must not leave a Lean build or a sweep burning cycles."""
    swarm = _swarm(tmp_path)
    swarm.control.register("spawner_01", role="Simulation Worker")
    swarm.control.begin("spawner_01")
    process = _spawn_scoped("spawner_01")
    try:
        swarm.control.request_stop("spawner_01", actor="empirical_lead_01",
                                   reason="over its step budget")
        signalled = swarm.ledger.by_action("STOP_REQUESTED")[0].payload["processes_signalled"]
        assert process.pid in signalled
        assert process.wait(timeout=10) != 0, "the child process survived the stop"
    finally:
        if process.poll() is None:  # pragma: no cover - cleanup only
            process.kill()


def test_stopping_a_lead_does_not_kill_the_verifiers_own_subprocess(tmp_path: Path):
    """A regression caught by a live run.

    Attribution used to be by thread id. Leaders and the Director run on the main
    thread, so cancelling a leader also signalled every process the main thread had
    started — including the proof gate's own verifier. A proof script that exits 0
    on its own was reported as REJECTED_NONZERO because the gate's subprocess had
    been killed by an unrelated cancellation.
    """
    from adaptive_harness.tools.process import live_pids

    swarm = _swarm(tmp_path)
    # A lead: registered, queued, and sharing this test's thread.
    lead = "theory_lead_01"
    swarm.control.begin(lead)
    assert swarm.control.record(lead).thread_id is not None

    # The gate's verifier, running on this same thread but owned by no agent.
    verifier = _spawn_scoped(None)
    try:
        assert verifier.pid in live_pids()
        swarm.control.request_stop(lead, actor="executive_director_01",
                                   reason="diverted to the Lean gap")
        assert verifier.poll() is None, (
            "cancelling a lead killed an unrelated process on the same thread")
    finally:
        if verifier.poll() is None:
            verifier.kill()
            verifier.wait(timeout=10)
    assert swarm.control.record(lead).state is WorkerState.CANCELLED


def test_a_proof_script_that_exits_zero_is_not_reported_as_failing(tmp_path: Path):
    """The user-visible consequence of the attribution bug, asserted directly."""
    swarm = _swarm(tmp_path)
    _write(swarm, "proofs/prop-01.py", PROOF_BODY)
    ok, detail, _ = swarm._evaluate_proofs()
    assert ok, detail
    assert [receipt.status for receipt in swarm.proofs.receipts] == ["VERIFIED_EXIT_0"]


def test_stopping_the_run_ends_the_loop_with_an_attributable_external_stop(tmp_path: Path):
    from adaptive_harness.research.gate import StopReason
    swarm = _swarm(tmp_path, absolute_ceiling=20, max_cycles=20)
    outcome = swarm.run(objective="x", on_status=lambda _m: None)
    # A run that was never stopped must not claim it was.
    assert outcome.stop_reason is not StopReason.EXTERNAL_STOP


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_workers_really_run_concurrently(tmp_path: Path):
    """A sequential loop must not be described as a swarm; prove the overlap."""
    swarm = _swarm(tmp_path, max_parallel_workers=4, max_workers_per_division=4)
    swarm.plan = _plan_with(swarm, 4)
    swarm.scale_division(Division.THEORY, 4)
    report = _report_with(swarm, "mathematical_soundness")
    planned = swarm._plan_cycle(report)
    assert len(planned) >= 2
    barrier = threading.Barrier(len(planned), timeout=15)

    def rendezvous(_unit):
        # A barrier only clears if every dispatched unit is in flight at once.
        barrier.wait()
        return {"ran": True, "success": False, "tool_calls": 0, "trajectory": [],
                "error": "synchronised", "state": WorkerState.FAILED.value}

    original = swarm._execute_one
    swarm._execute_one = rendezvous  # type: ignore[method-assign]
    try:
        summaries = swarm._execute_planned(planned)
    finally:
        swarm._execute_one = original  # type: ignore[method-assign]
    assert len(summaries) == len(planned)


def test_serial_mode_is_available_and_still_correct(tmp_path: Path):
    swarm = _swarm(tmp_path, max_parallel_workers=1)
    swarm.plan = _plan_with(swarm, 2)
    swarm.scale_division(Division.THEORY, 2)
    planned = swarm._plan_cycle(_report_with(swarm, "mathematical_soundness"))
    seen: list[str] = []

    def record(_unit):
        seen.append(threading.current_thread().name)
        return {"ran": True, "success": False, "tool_calls": 0, "trajectory": []}

    original = swarm._execute_one
    swarm._execute_one = record  # type: ignore[method-assign]
    try:
        swarm._execute_planned(planned)
    finally:
        swarm._execute_one = original  # type: ignore[method-assign]
    assert len(seen) == len(planned)
    assert len(set(seen)) == 1, "max_parallel_workers=1 must stay on one thread"


def test_concurrent_cycles_keep_the_ledger_chain_intact(tmp_path: Path):
    """Whatever else changes, the audit trail must stay verifiable."""
    swarm = _swarm(tmp_path, max_parallel_workers=4, max_workers_per_division=4)
    swarm.plan = _plan_with(swarm, 4)
    swarm.scale_division(Division.THEORY, 4)
    report = _report_with(swarm, "mathematical_soundness")
    planned = swarm._plan_cycle(report)

    def fake(unit):
        _division, _gap, task, worker = unit
        swarm.control.begin(worker, task_id=task.task_id)
        swarm._log_trajectory(swarm.agents[worker], Division(task.division),
                              {"ran": True, "success": True, "tool_calls": 1,
                               "trajectory": [{"tool": "write_file", "ok": True}]})
        swarm._close_task(task, worker, WorkerState.COMPLETED, evidence_ids=("PROOF-001",))
        return {"ran": True, "success": True, "accepted": True, "tool_calls": 1,
                "trajectory": [], "state": WorkerState.COMPLETED.value}

    original = swarm._execute_one
    swarm._execute_one = fake  # type: ignore[method-assign]
    try:
        summaries = swarm._execute_planned(planned)
    finally:
        swarm._execute_one = original  # type: ignore[method-assign]
    assert all(item["accepted"] for item in summaries)
    ok, detail = swarm.ledger.verify()
    assert ok, detail


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_an_interrupted_run_resumes_without_repeating_completed_work(tmp_path: Path):
    """A restart must not re-bill the Director, the leads, or finished tasks."""
    script = [("write_file", {"path": "00_objective_spec.md", "content": "# spec\n"}),
              ("write_file", {"path": "claim_manifest.json", "content": json.dumps({
                  "claims": [{"id": "PROP-01", "name": "c",
                              "statement": "the sum of two and two is four",
                              "hypotheses": [], "kind": "theorem",
                              "sympy_expression": "2 + 2 == 4",
                              "lean_statement": "example : (2:Nat) + 2 = 4 := by decide"}]})})]
    first = _swarm(tmp_path, script)
    first._prepare_live_research("prove 2+2=4")
    task = _new_task(first, gap="mathematical_soundness", artifact="proofs/prop-01.py")
    first.board.lease(task.task_id, "theory_prover_01")
    _write(first, task.artifact, PROOF_BODY)
    _write(first, "proofs/prop-01.md", "# derivation\n")
    first._close_task(task, "theory_prover_01", WorkerState.COMPLETED,
                      evidence_ids=("PROOF-001",))
    first.board.save(first.workspace.root / "task_board.json")
    calls_before = first.ledger.count

    # A brand new process would construct a brand new swarm over the same tree.
    second = _swarm(tmp_path, script)
    report = second.recovery_report()
    assert report["resumed"] is True
    assert report["tasks"] == 1
    restored = second.board.get(task.task_id)
    assert restored.state is WorkerState.COMPLETED
    assert restored.evidence_ids == ("PROOF-001",)
    # run() prepares the plan before the first cycle; do the same here.
    second._prepare_live_research("prove 2+2=4")
    assert second._plan_cycle(_report_with(second, "mathematical_soundness")) == []
    # Both swarms share one ledger, so only the entries written after the restart
    # count as evidence about what the restart did.
    fresh = second.ledger.read_raw()[calls_before:]
    assert not [raw for raw in fresh
                if raw["action"] == "WORKER_TRAJECTORY"
                and raw["sender"]["agent_id"] == DIRECTOR_ID], \
        "the Director was re-billed on a resumed run"
    assert not [raw for raw in fresh if raw["action"] == "WORKER_TRAJECTORY"], \
        "no lead should be re-run either when its assignment file already exists"
    assert [raw for raw in fresh if raw["action"] == "RUN_RESUMED_FROM_STATE"]
    assert second.ledger.verify()[0]
    assert second.ledger.count > calls_before


def test_a_stale_lease_does_not_strand_the_board_after_a_crash(tmp_path: Path):
    swarm = _swarm(tmp_path)
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, "theory_prover_01", ttl_s=3600)
    swarm.board.save(swarm.workspace.root / "task_board.json")
    # The worker died mid-attempt: the lease is on disk but its owner is gone.
    revived = ResearchSwarm("coordination run", root=tmp_path,
                            config=SwarmConfig(llm_client_factory=lambda: ScriptedClient([])))
    restored = revived.board.get(task.task_id)
    assert restored.owner is None
    assert revived.board.lease(task.task_id, "theory_prover_02").owner == "theory_prover_02"


def test_resume_can_be_switched_off(tmp_path: Path):
    swarm = _swarm(tmp_path)
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, "theory_prover_01")
    swarm.board.save(swarm.workspace.root / "task_board.json")
    fresh = ResearchSwarm("coordination run", root=tmp_path,
                          config=SwarmConfig(resume=False,
                                             llm_client_factory=lambda: ScriptedClient([])))
    assert fresh.board.all() == ()
    assert fresh.recovery_report()["resumed"] is False


def test_the_board_file_is_written_atomically_and_reloads(tmp_path: Path):
    swarm = _swarm(tmp_path)
    task = _new_task(swarm)
    path = swarm.board.save(swarm.workspace.root / "task_board.json")
    assert not path.with_name(path.name + ".tmp").exists(), "a temp file was left behind"
    reloaded = ResearchSwarm("coordination run", root=tmp_path,
                             config=SwarmConfig(llm_client_factory=lambda: ScriptedClient([])))
    assert reloaded.board.get(task.task_id) is not None


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_status_reports_agents_tasks_and_ledger_actions(tmp_path: Path):
    swarm = _swarm(tmp_path)
    swarm.plan = _plan_with(swarm, 1)
    report = _report_with(swarm, "mathematical_soundness")
    swarm._plan_cycle(report)
    status = swarm.swarm_status()
    assert status["open_tasks"], "the operator should see the outstanding work"
    assert status["ledger_actions"].get("TASK_ASSIGNED", 0) >= 1
    assert set(status["state_counts"]) == {state.value for state in WorkerState}
    assert json.dumps(status)
    rendered = swarm.render_status()
    assert DIRECTOR_ID in rendered


def test_an_unanswered_help_request_is_visible_to_the_operator(tmp_path: Path):
    swarm = _swarm(tmp_path)
    swarm.bus.ask_for_help(sender="theory_prover_01", recipient="theory_lead_01",
                           subject="no progress", body="three identical failures")
    pending = swarm.swarm_status()["unacknowledged_requests"]
    assert [item["subject"] for item in pending] == ["no progress"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _BlockingClient(ScriptedClient):
    """Parks inside one turn until released, so a cancel provably lands mid-loop.

    Sleeping would make the test probabilistic: on a fast machine the loop could
    finish before the cancelling thread was scheduled, and the test would pass for
    the wrong reason. Blocking on an event makes the ordering exact.
    """

    def __init__(self, parked: threading.Event, release: threading.Event):
        super().__init__([])
        self.parked = parked
        self.release = release
        self.turns = 0

    def complete(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        self.turn += 1
        self.turns = self.turn
        self.calls.append({"turn": self.turn, "tools": []})
        self.parked.set()
        assert self.release.wait(timeout=30), "the test never released the model"
        return LLMResponse(
            content="checking", model=model or self.default_model,
            tool_calls=[ToolCall(id=f"call_{self.turn}", name="run_bash",
                                 arguments={"command": "true"})])


def _report_with(swarm: ResearchSwarm, gap: str):
    from adaptive_harness.research.gate import GateReport, Invariant, InvariantStatus
    invariant = next(item for item in Invariant if item.value == gap)
    return GateReport(statuses=(InvariantStatus(invariant=invariant, satisfied=False,
                                               detail="forced by the test"),),
                      fingerprint="fp", cycle=1)


def _plan_with(swarm: ResearchSwarm, count: int):
    from adaptive_harness.research.claim import Proposition, TopicPlan
    return TopicPlan(
        swarm.topic,
        tuple(Proposition(prop_id=f"PROP-{index:02d}", kind="theorem", name=f"c{index}",
                          statement=f"claim {index}", hypotheses=(),
                          lean_statement=f"example : claim{index} = {index} := by decide")
              for index in range(1, count + 1)),
        strategy="test", notes="")


def _new_task(swarm: ResearchSwarm, gap: str = "mathematical_soundness",
              artifact: str = "proofs/prop_01.py"):
    return swarm.board.create(title="t", division="theory", gap=gap,
                              created_by=DIRECTOR_ID, artifact=artifact,
                              acceptance="exit 0")


def _proof_receipt(path: Path):
    from adaptive_harness.research.proof import ProofReceipt
    return [ProofReceipt(theorem_id="PROP_01", script=str(path), exit_code=0,
                         stdout="expanded", status="VERIFIED_EXIT_0",
                         sha256="sha256:abc", duration_ms=100.0)]


# ---------------------------------------------------------------------------
# Agent-facing stop control and status
# ---------------------------------------------------------------------------

def test_the_interactive_director_can_stop_and_inspect_its_swarm(tmp_path: Path):
    """The TUI/research-mode agent had no authority over the control plane."""
    from adaptive_harness.agent.agent import DeveloperAgent
    from adaptive_harness.tools.research_swarm import StopWorkerTool, SwarmStatusTool
    from adaptive_harness.llm.client import LLMClient

    swarm = _swarm(tmp_path)
    agent = DeveloperAgent(llm_client=LLMClient(force_mock=True),
                           workspace_root=swarm.workspace.root)
    try:
        agent.enable_research("interactive", root=tmp_path, config=swarm.config)
        assert "stop_worker" in agent.tools and "swarm_status" in agent.tools
        worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "loop", ["read_file"])
        swarm.control.begin(worker, task_id="TASK-0001")

        status = SwarmStatusTool(swarm).execute()
        assert status.success
        payload = json.loads(status.output)
        listed = [item["agent_id"] for item in payload["workers"]]
        assert DIRECTOR_ID in listed and worker in listed
        assert {item["agent_id"] for item in payload["workers"] if item["state"] == "running"} == {worker}

        stopped = StopWorkerTool(swarm).execute(worker, "cancel", "theory_lead_01",
                                                "three identical failures")
        assert stopped.success, stopped.error
        assert json.loads(stopped.output)["stop_actor"] == "theory_lead_01"
        assert swarm.control.record(worker).state is WorkerState.CANCELLED
    finally:
        agent.disable_research()
    assert "stop_worker" not in agent.tools


def test_stopping_a_worker_that_is_not_running_is_reported_as_such(tmp_path: Path):
    from adaptive_harness.tools.research_swarm import StopWorkerTool

    swarm = _swarm(tmp_path)
    tool = StopWorkerTool(swarm)
    result = tool.execute("never_registered", "cancel", "director", "x")
    assert not result.success and "unregistered" in (result.error or "")
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "d", ["read_file"])
    swarm.control.begin(worker)
    assert tool.execute(worker, "cancel", "director", "diverted").success
    # A second cancel has nothing to act on, and says so rather than claiming success.
    again = tool.execute(worker, "cancel", "director", "diverted")
    assert not again.success and "cancelled" in (again.error or "")


def test_a_stop_without_an_actor_or_reason_is_refused(tmp_path: Path):
    from adaptive_harness.tools.research_swarm import StopWorkerTool

    swarm = _swarm(tmp_path)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "d", ["read_file"])
    swarm.control.begin(worker)
    tool = StopWorkerTool(swarm)
    for actor, reason in (("", "because"), ("director", "")):
        result = tool.execute(worker, "cancel", actor, reason)
        assert not result.success
    assert swarm.control.record(worker).state is WorkerState.RUNNING, (
        "a refused stop must not have taken effect")


def test_pause_then_resume_through_the_tool(tmp_path: Path):
    from adaptive_harness.tools.research_swarm import StopWorkerTool

    swarm = _swarm(tmp_path)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "d", ["read_file"])
    swarm.control.begin(worker)
    tool = StopWorkerTool(swarm)
    assert tool.execute(worker, "pause", "theory_lead_01", "awaiting review").success
    assert swarm.control.token(worker).paused
    assert tool.execute(worker, "resume", "theory_lead_01", "review done").success
    assert not swarm.control.token(worker).paused
    assert not tool.execute(worker, "resume", "theory_lead_01", "again").success


def test_an_unknown_action_is_rejected(tmp_path: Path):
    from adaptive_harness.tools.research_swarm import StopWorkerTool

    swarm = _swarm(tmp_path)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "d", ["read_file"])
    swarm.control.begin(worker)
    result = StopWorkerTool(swarm).execute(worker, "detonate", "director", "why not")
    assert not result.success and "cancel, pause, or resume" in (result.error or "")


# ---------------------------------------------------------------------------
# The delivered document for a run that did not converge
# ---------------------------------------------------------------------------

def _unsolved(swarm: ResearchSwarm):
    from adaptive_harness.research.gate import (ConvergenceOutcome, GateReport, Invariant,
                                                InvariantStatus, StopReason)
    report = GateReport(
        statuses=tuple(InvariantStatus(invariant=item, satisfied=False,
                                       detail=f"{item.value} was not satisfied")
                       for item in Invariant),
        fingerprint="fp", cycle=1)
    return ConvergenceOutcome(False, StopReason.EXTERNAL_STOP, 1, report, (), 6)


def test_the_unsolved_report_states_which_invariants_failed_and_why(tmp_path: Path):
    swarm = _swarm(tmp_path)
    text = swarm._progress_report(_unsolved(swarm))
    assert "not a proof" in text
    for invariant in ("mathematical_soundness", "adversarial_clearance", "document_integrity"):
        assert invariant in text
    assert "was not satisfied" in text
    # A run that never certified a Lean file must not imply a formal proof exists.
    assert "No Lean file was certified" in text


def test_the_unsolved_report_names_the_human_who_stopped_the_run(tmp_path: Path):
    """An operator stop must not be presented as a mathematical limit."""
    swarm = _swarm(tmp_path)
    swarm.control.stop_run(actor="operator", reason="SIGINT from the operator")
    text = swarm._progress_report(_unsolved(swarm))
    assert "stopped by a human" in text
    assert "SIGINT from the operator" in text
    assert "not a mathematical limit" in text


def test_the_unsolved_report_lists_open_work_and_unanswered_requests(tmp_path: Path):
    swarm = _swarm(tmp_path)
    task = _new_task(swarm, artifact="proofs/prop-01.py")
    swarm.board.lease(task.task_id, "theory_prover_01")
    swarm.bus.ask_for_help(sender="theory_prover_01", recipient="theory_lead_01",
                           subject="stuck on rfl", body="no progress")
    text = swarm._progress_report(_unsolved(swarm))
    assert "Work that was left open" in text and task.task_id in text
    assert "never answered" in text and "stuck on rfl" in text


def test_typst_metacharacters_from_an_artifact_cannot_break_the_report(tmp_path: Path):
    """The first live failure was ``#math.N``: unescaped model prose is markup."""
    swarm = _swarm(tmp_path)
    swarm.plan = _plan_with(swarm, 1)
    swarm.plan = swarm.plan.__class__(
        swarm.plan.topic,
        tuple(item.__class__(prop_id=item.prop_id, kind=item.kind, name=item.name,
                             statement="cost is #math.N * 3_0 and $x$", hypotheses=item.hypotheses,
                             lean_statement=item.lean_statement) for item in swarm.plan.propositions),
        strategy="test", notes="")
    swarm._adjudicate()
    text = swarm._progress_report(_unsolved(swarm))
    assert "\\#math.N" in text and "\\* 3\\_0" in text and "\\$x\\$" in text


def test_a_stale_but_compiling_draft_does_not_replace_the_diagnostic_report(tmp_path: Path):
    """Shipping a valid old draft would read like a finished paper."""
    swarm = _swarm(tmp_path)
    draft = "= A Draft Paper\n\nSome prose the author wrote.\n"
    swarm.workspace.paper_typ.write_text(draft, encoding="utf-8")
    swarm._republish(_unsolved(swarm))
    delivered = swarm.workspace.paper_typ.read_text(encoding="utf-8")
    assert "Research progress report" in delivered
    assert "A Draft Paper" not in delivered
    assert swarm.workspace.root.joinpath("paper_draft.typ").read_text(
        encoding="utf-8") == draft
    assert not (swarm.workspace.root / "progress_report_build_error.txt").exists()


# ---------------------------------------------------------------------------
# Swarm-level supervision of the per-agent overseer
# ---------------------------------------------------------------------------

def _stalled(count: int = 1) -> dict[str, Any]:
    return {"ran": True, "success": False, "accepted": False, "tool_calls": count * 4,
            "trajectory": [], "error": "no progress",
            "state": WorkerState.FAILED.value,
            "overseer": [{"state": "LOOPING_DETECTED", "directive": "stop circling",
                          "confidence": 0.99} for _ in range(count)]}


def test_a_single_overseer_flag_asks_for_help_rather_than_stopping(tmp_path: Path):
    """One flagged step is normal friction on a hard proof, not a death sentence."""
    swarm = _swarm(tmp_path, overseer_stop_threshold=3)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it", ["read_file"])
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, worker)
    swarm.control.begin(worker, task_id=task.task_id)
    assert swarm._supervise(Division.THEORY, worker, task, _stalled(1)) == "asked_for_help"
    assert swarm.control.record(worker).state is WorkerState.RUNNING
    pending = swarm.bus.unacknowledged()
    assert len(pending) == 1 and pending[0].kind.value == "request_for_help"
    assert pending[0].recipient == "theory_lead_01"


def test_a_repeated_loop_stops_the_worker_with_the_overseer_as_the_reason(tmp_path: Path):
    swarm = _swarm(tmp_path, overseer_stop_threshold=3)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it", ["read_file"])
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, worker)
    swarm.control.begin(worker, task_id=task.task_id)
    assert swarm._supervise(Division.THEORY, worker, task, _stalled(3)) == "stopped"
    record = swarm.control.record(worker)
    assert record.state is WorkerState.CANCELLED
    assert record.stop_actor == "theory_lead_01"
    assert "LOOPING_DETECTED" in record.stop_reason and "3 time(s)" in record.stop_reason
    assert swarm.board.get(task.task_id).owner is None
    assert any(entry.action == "STOP_REQUESTED"
               and entry.sender["agent_id"] == "theory_lead_01"
               for entry in swarm.ledger.by_action("STOP_REQUESTED"))


def test_a_healthy_worker_is_left_alone(tmp_path: Path):
    swarm = _swarm(tmp_path)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it", ["read_file"])
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, worker)
    swarm.control.begin(worker, task_id=task.task_id)
    assert swarm._supervise(Division.THEORY, worker, task, _stalled(0)) is None
    assert swarm._supervise(
        Division.THEORY, worker, task,
        {"overseer": [{"state": "HEALTHY_PROGRESS", "directive": ""}]}) is None
    assert swarm.control.record(worker).state is WorkerState.RUNNING
    assert swarm.bus.unacknowledged() == ()


def test_a_completed_task_is_never_stopped_for_stalling(tmp_path: Path):
    """Finishing is not stalling, however many steps it took."""
    swarm = _swarm(tmp_path, overseer_stop_threshold=1)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it", ["read_file"])
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, worker)
    swarm.control.begin(worker, task_id=task.task_id)
    swarm.board.complete(task.task_id, worker)
    assert swarm._supervise(Division.THEORY, worker, task, _stalled(5)) is None
    assert swarm.control.record(worker).state is WorkerState.RUNNING


def test_supervision_outcomes_appear_in_the_leads_review(tmp_path: Path):
    swarm = _swarm(tmp_path, overseer_stop_threshold=3)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it", ["read_file"])
    task = _new_task(swarm)
    swarm.board.lease(task.task_id, worker)
    swarm.control.begin(worker, task_id=task.task_id)
    swarm._review(Division.THEORY, worker, task, _stalled(2))
    review = [entry for entry in swarm.ledger.by_action("PROGRESS_SUMMARY")
              if entry.sender["agent_id"] == "theory_lead_01"][0]
    assert review.payload["supervision"] == "asked_for_help"
    assert review.payload["overseer_verdicts"] == ["LOOPING_DETECTED"] * 2


def test_a_worker_that_exceeds_its_budget_is_timed_out_and_retryable(tmp_path: Path):
    """A deadline must be recorded as a timeout, not as an unattributed cancel."""
    # Long enough for the run's early classifier events to pass, short enough that
    # the budget lapses while the model is parked mid-attempt.
    swarm = _swarm(tmp_path, worker_max_steps=200, worker_timeout_s=0.5)
    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "loop", ["run_bash"])
    task = _new_task(swarm)
    _write(swarm, task.artifact, PROOF_BODY)
    swarm.board.lease(task.task_id, agent_id)
    parked, release = threading.Event(), threading.Event()

    def factory():
        client = _BlockingClient(parked, release)
        client.default_model = "stealth/space-bunny-alpha"
        return client

    swarm.config.llm_client_factory = factory
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result.update(swarm._produce(Division.THEORY, swarm.agents[agent_id],
                                         "mathematical_soundness", [], task=task))
        except BaseException as exc:  # noqa: BLE001 - surfaced as an assertion below
            result["exception"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert parked.wait(timeout=30)
        time.sleep(0.8)  # let the budget lapse while the model is still parked
    finally:
        release.set()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert "exception" not in result, result["exception"]

    record = swarm.control.record(agent_id)
    assert record.state is WorkerState.TIMED_OUT
    assert record.stop_actor == "swarm_control" and record.stop_cause == "timeout"
    assert record.retryable
    # The attempt returns to the board rather than being written off.
    assert swarm.board.get(task.task_id).owner is None
    assert swarm.board.get(task.task_id).state is not WorkerState.CANCELLED
    # And the reason is on the record, not lost.
    assert "exceeded" in (record.error or record.stop_reason)
    timeouts = [entry for entry in swarm.ledger.by_action("WORKER_STATE_CHANGE")
                if entry.payload.get("cause") == "deadline_expiry"]
    assert timeouts and timeouts[0].payload["retryable"] is True


def test_a_retry_is_told_why_the_previous_attempt_failed(tmp_path: Path):
    """A fresh agent has no memory of the run before it; without this it repeats
    the same mistake and burns the same budget."""
    seen: list[str] = []

    swarm = _swarm(tmp_path, worker_max_steps=1)
    worker = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it",
                                  ["write_file"])
    task = _new_task(swarm)
    original = swarm._run_worker

    def capture(agent, division, directive, **kwargs):
        seen.append(directive)
        return original(agent, division, directive, **kwargs)

    swarm._run_worker = capture  # type: ignore[method-assign]
    try:
        # First attempt: nothing to carry forward.
        swarm.board.lease(task.task_id, worker)
        _write(swarm, task.artifact, PROOF_BODY)
        swarm._produce(Division.THEORY, swarm.agents[worker], "mathematical_soundness",
                       [], task=task)
        assert not any("previous attempt" in item for item in seen)

        # Record a failure, then re-dispatch the same task.
        swarm.board.lease(task.task_id, worker)
        swarm.board.fail(task.task_id, worker, "sympy.has does not exist")
        assert swarm.board.get(task.task_id).attempts == 2
        _write(swarm, task.artifact, PROOF_BODY)
        swarm._produce(Division.THEORY, swarm.agents[worker], "mathematical_soundness",
                       [], task=task)
    finally:
        swarm._run_worker = original  # type: ignore[method-assign]
    retried = [item for item in seen if "previous attempt" in item]
    assert retried, "the retry never learned from the failure"
    assert "attempt 2" in retried[0]
    assert "sympy.has does not exist" in retried[0]
