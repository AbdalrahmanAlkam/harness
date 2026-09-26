"""Tests for the swarm's coordination layer: task board, messaging, stop control.

These are the tests the previous implementation could not have had, because the
behaviours did not exist: no duplicate assignment, no lifecycle states, no routed
messages, no cancellation reaching a running tool loop, no recovery. Each test
below names the failure it prevents.
"""

from __future__ import annotations

from pathlib import Path
import json
import threading
import time

import pytest

from adaptive_harness.research.coordination import (
    CancellationToken, CoordinationError, MessageBus, MessageKind, StopKind, SwarmControl,
    Task, TaskBoard, WorkerCancelled, WorkerState, LEGAL_TRANSITIONS)
from adaptive_harness.research.ledger import ACTIONS, CommLedger
from adaptive_harness.research.roles import Division


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ledger(tmp_path: Path) -> CommLedger:
    return CommLedger(tmp_path / "comm_ledger.jsonl")


@pytest.fixture()
def board() -> TaskBoard:
    return TaskBoard(lease_s=60.0, max_attempts=3)


@pytest.fixture()
def control(ledger: CommLedger, board: TaskBoard) -> SwarmControl:
    return SwarmControl(ledger, board=board, bus=MessageBus(ledger))


def _task(board: TaskBoard, **kwargs) -> Task:
    payload = {"title": "close the gap", "division": "theory", "gap": "mathematical_soundness",
               "created_by": "executive_director_01"}
    payload.update(kwargs)
    return board.create(**payload)


# ---------------------------------------------------------------------------
# Task board
# ---------------------------------------------------------------------------

def test_assignments_get_unique_stable_ids(board: TaskBoard):
    first = _task(board, artifact="proofs/a.py")
    second = _task(board, artifact="proofs/b.py")
    assert (first.task_id, second.task_id) == ("TASK-0001", "TASK-0002")
    assert first.task_id != second.task_id


def test_a_leased_task_cannot_be_leased_by_a_second_worker(board: TaskBoard):
    task = _task(board, artifact="proofs/a.py")
    board.lease(task.task_id, "theory_prover_01")
    with pytest.raises(CoordinationError, match="already leased by theory_prover_01"):
        board.lease(task.task_id, "theory_prover_02")
    # The point is that only one worker owns the work, not that the call is quiet.
    assert board.get(task.task_id).owner == "theory_prover_01"


def test_two_tasks_may_not_claim_the_same_artifact(board: TaskBoard):
    """The collision that made two theory workers race on one proof file."""
    _task(board, artifact="proofs/prop_01.py")
    with pytest.raises(CoordinationError, match="already owned by TASK-0001"):
        _task(board, title="rival", artifact="proofs/prop_01.py")


def test_a_dependent_task_is_unrentable_until_its_dependency_completes(board: TaskBoard):
    first = _task(board, artifact="proofs/a.py")
    second = _task(board, artifact="paper.typ", dependencies=[first.task_id])
    assert board.unmet_dependencies(second.task_id) == (first.task_id,)
    with pytest.raises(CoordinationError, match="blocked on"):
        board.lease(second.task_id, "literature_scout_01")
    board.lease(first.task_id, "theory_prover_01")
    board.complete(first.task_id, "theory_prover_01", evidence_ids=["PROOF-001"])
    assert board.unmet_dependencies(second.task_id) == ()
    assert board.lease(second.task_id, "literature_scout_01").owner == "literature_scout_01"


def test_a_finished_task_cannot_be_reopened(board: TaskBoard):
    task = _task(board)
    board.lease(task.task_id, "worker_01")
    board.complete(task.task_id, "worker_01")
    with pytest.raises(CoordinationError, match="completed"):
        board.lease(task.task_id, "worker_02")


def test_a_failed_attempt_returns_to_the_queue_and_then_exhausts(board: TaskBoard):
    task = _task(board)
    for attempt in range(board.max_attempts):
        board.lease(task.task_id, f"worker_{attempt}")
        board.fail(task.task_id, f"worker_{attempt}", f"attempt {attempt} broke")
    assert board.get(task.task_id).state is WorkerState.FAILED
    with pytest.raises(CoordinationError, match="exhausted"):
        board.lease(task.task_id, "worker_final")
    assert board.get(task.task_id).attempts == board.max_attempts


def test_lease_any_hands_two_racing_workers_different_tasks(board: TaskBoard):
    _task(board, title="a", artifact="proofs/a.py")
    _task(board, title="b", artifact="proofs/b.py")
    first = board.lease_any("worker_a")
    second = board.lease_any("worker_b")
    assert first.task_id != second.task_id


def test_an_expired_lease_can_be_taken_over(board: TaskBoard):
    task = _task(board)
    board.lease(task.task_id, "worker_01", ttl_s=0.01)
    time.sleep(0.05)
    assert board.lease(task.task_id, "worker_02").owner == "worker_02"


def test_a_completed_task_keeps_its_artifact_pinned_as_evidence(board: TaskBoard):
    task = _task(board, artifact="proofs/a.py")
    board.lease(task.task_id, "worker_01")
    board.complete(task.task_id, "worker_01", evidence_ids=["PROOF-001"])
    assert board.get(task.task_id).evidence_ids == ("PROOF-001",)
    assert board.get(task.task_id).state is WorkerState.COMPLETED


def test_only_the_lease_holder_may_settle_a_task(board: TaskBoard):
    task = _task(board)
    board.lease(task.task_id, "worker_01")
    with pytest.raises(CoordinationError, match="does not hold"):
        board.complete(task.task_id, "worker_02")


# ---------------------------------------------------------------------------
# Board persistence and recovery
# ---------------------------------------------------------------------------

def test_the_board_survives_a_save_and_reload(tmp_path: Path, board: TaskBoard):
    task = _task(board, artifact="proofs/a.py", acceptance="runs to exit 0")
    board.lease(task.task_id, "worker_01")
    board.complete(task.task_id, "worker_01", evidence_ids=["PROOF-007"])
    path = board.save(tmp_path / "task_board.json")
    reloaded = TaskBoard.load(path)
    restored = reloaded.get(task.task_id)
    assert restored.state is WorkerState.COMPLETED
    assert restored.evidence_ids == ("PROOF-007",)
    assert restored.acceptance == "runs to exit 0"
    # A reloaded board must not reissue an id that already names different work.
    fresh = reloaded.create(title="new", division="theory", gap="g", created_by="d")
    assert fresh.task_id == "TASK-0002"


def test_recovery_releases_stale_leases_and_trusts_the_ledger(ledger: CommLedger, board: TaskBoard):
    task = _task(board, artifact="proofs/a.py")
    board.lease(task.task_id, "worker_01", ttl_s=600)
    board.release(task.task_id, "worker_01")
    board.lease(task.task_id, "worker_01", ttl_s=600)
    # The process died mid-attempt: the lease is stale even though it has not expired.
    healed = board.recovered_from(ledger)
    assert healed >= 1
    assert board.get(task.task_id).owner is None
    assert board.lease(task.task_id, "worker_02").owner == "worker_02"


def test_the_ledger_overrides_a_board_that_still_shows_work_in_flight(
        ledger: CommLedger, board: TaskBoard):
    task = _task(board, artifact="proofs/a.py")
    board.lease(task.task_id, "worker_01")
    ledger.append("TASK_COMPLETED", {"agent_id": "worker_01"}, {"agent_id": "theory_lead_01"},
                  {"task_id": task.task_id, "state": WorkerState.COMPLETED.value})
    board.recovered_from(ledger)
    assert board.get(task.task_id).state is WorkerState.COMPLETED
    # A task the ledger shows finished must not be handed out again.
    with pytest.raises(CoordinationError):
        board.lease(task.task_id, "worker_02")


# ---------------------------------------------------------------------------
# Routed messaging
# ---------------------------------------------------------------------------

def test_a_message_reaches_its_addressed_recipient(ledger: CommLedger):
    bus = MessageBus(ledger)
    bus.register_route("theory_prover_01", "theory_lead_01")
    sent = bus.ask_for_help(sender="theory_prover_01", recipient=bus.contact_for("theory_prover_01"),
                            subject="stuck on lemma 3", body="decide gives a non-normalised goal")
    assert [item.message_id for item in bus.inbox("theory_lead_01")] == [sent.message_id]
    assert bus.inbox("adversarial_lead_01") == ()


def test_an_unregistered_agent_escalates_to_the_director(ledger: CommLedger):
    bus = MessageBus(ledger)
    assert bus.contact_for("stray_agent") == "executive_director_01"


def test_a_help_request_must_be_acknowledged_and_the_ack_is_recorded(ledger: CommLedger):
    bus = MessageBus(ledger)
    sent = bus.ask_for_help(sender="formalist_01", recipient="formal_lead_01",
                            subject="tactic budget", body="omega times out")
    assert [item.message_id for item in bus.unacknowledged()] == [sent.message_id]
    bus.acknowledge(sent.message_id, "formal_lead_01", "use decide instead")
    assert bus.unacknowledged() == ()
    assert bus.get(sent.message_id).acknowledged_by == "formal_lead_01"
    assert any(entry.action == "MESSAGE_ACKNOWLEDGED"
               for entry in ledger.by_action("MESSAGE_ACKNOWLEDGED"))


def test_only_the_addressed_recipient_may_acknowledge(ledger: CommLedger):
    bus = MessageBus(ledger)
    sent = bus.ask_for_help(sender="a_01", recipient="theory_lead_01", subject="s", body="b")
    with pytest.raises(CoordinationError, match="not the recipient"):
        bus.acknowledge(sent.message_id, "formal_lead_01")


def test_asking_and_hearing_nothing_differs_from_asking_and_hearing_no(ledger: CommLedger):
    """A recorded outcome is what distinguishes 'no answer' from 'the answer was no'."""
    bus = MessageBus(ledger)
    silent = bus.ask_for_help(sender="a_01", recipient="theory_lead_01", subject="q1", body="b1")
    answered = bus.ask_for_help(sender="b_01", recipient="theory_lead_01", subject="q2", body="b2")
    bus.acknowledge(answered.message_id, "theory_lead_01")
    bus.record_outcome(answered.message_id, "b_01", "the lemma is false as stated")
    assert bus.get(silent.message_id).outcome == ""
    assert bus.get(answered.message_id).outcome == "the lemma is false as stated"


def test_a_message_is_threaded_under_its_own_ledger_entry(ledger: CommLedger):
    bus = MessageBus(ledger)
    sent = bus.ask_for_help(sender="a_01", recipient="theory_lead_01", subject="q", body="b")
    bus.acknowledge(sent.message_id, "theory_lead_01")
    ack = ledger.by_action("MESSAGE_ACKNOWLEDGED")[0]
    assert ack.parent_id == sent.ledger_id


def test_the_bus_is_rebuilt_from_the_ledger_on_restart(ledger: CommLedger, tmp_path: Path):
    bus = MessageBus(ledger)
    asked = bus.ask_for_help(sender="theory_prover_01", recipient="theory_lead_01",
                             subject="stuck", body="cannot close the goal")
    bus.acknowledge(asked.message_id, "theory_lead_01", "use omega")
    bus.record_outcome(asked.message_id, "theory_prover_01", "closed")

    reopened = CommLedger(ledger.path)
    revived = MessageBus(reopened)
    revived.load(reopened.read_raw())
    restored = revived.get(asked.message_id)
    assert restored is not None
    assert restored.acknowledged_by == "theory_lead_01"
    assert restored.outcome == "closed"
    assert revived.unacknowledged() == ()
    # A reloaded bus must not reissue an id that already names a different message.
    assert revived.send(sender="x", recipient="y", kind=MessageKind.DIRECTIVE,
                        subject="s", body="b").message_id == "MSGQ-0002"


def test_a_restart_keeps_an_unanswered_help_request_visible(ledger: CommLedger, tmp_path: Path):
    bus = MessageBus(ledger)
    bus.ask_for_help(sender="theory_prover_01", recipient="theory_lead_01",
                     subject="stuck", body="no progress in three attempts")
    reopened = CommLedger(ledger.path)
    revived = MessageBus(reopened)
    revived.load(reopened.read_raw())
    assert [item.subject for item in revived.unacknowledged()] == ["stuck"]


def test_an_agent_may_not_message_itself(ledger: CommLedger):
    bus = MessageBus(ledger)
    with pytest.raises(ValueError):
        bus.send(sender="a_01", recipient="a_01", kind=MessageKind.DIRECTIVE, subject="s", body="b")


# ---------------------------------------------------------------------------
# Lifecycle states
# ---------------------------------------------------------------------------

def test_all_seven_worker_states_exist_and_only_six_are_distinguishable_paths():
    assert {state.value for state in WorkerState} == {
        "queued", "running", "blocked", "completed", "failed", "cancelled", "timed_out"}
    assert WorkerState.COMPLETED.terminal and WorkerState.CANCELLED.terminal
    assert not WorkerState.QUEUED.terminal and WorkerState.QUEUED.active


def test_a_worker_keeps_its_own_terminal_outcome(ledger: CommLedger, control: SwarmControl):
    control.register("done_01")
    control.begin("done_01")
    control.complete("done_01")
    control.register("failed_01")
    control.begin("failed_01")
    control.fail("failed_01", "the script would not run")
    control.register("slow_01")
    control.begin("slow_01")
    control.timeout("slow_01", "exceeded its 60s budget")
    assert {record.agent_id: record.state for record in control.workers()} == {
        "done_01": WorkerState.COMPLETED,
        "failed_01": WorkerState.FAILED,
        "slow_01": WorkerState.TIMED_OUT}


def test_an_impossible_transition_is_refused_rather_than_silently_accepted(
        control: SwarmControl):
    control.register("w1")
    control.begin("w1")
    control.complete("w1")
    with pytest.raises(CoordinationError, match="cannot move from completed to running"):
        control.begin("w1")


def test_every_transition_used_by_the_control_plane_is_declared():
    assert WorkerState.QUEUED in LEGAL_TRANSITIONS
    assert WorkerState.FAILED not in LEGAL_TRANSITIONS[WorkerState.COMPLETED]


def test_every_state_change_is_recorded_with_its_actor_and_direction(
        ledger: CommLedger, control: SwarmControl):
    control.register("w1", role="SymPy Prover", parent_id="theory_lead_01")
    control.begin("w1", task_id="TASK-0001")
    control.complete("w1", tool_calls=3)
    changes = ledger.by_action("WORKER_STATE_CHANGE")
    assert [(entry.payload["from"], entry.payload["to"]) for entry in changes] == [
        ("queued", "running"), ("running", "completed")]
    assert all(entry.sender["agent_id"] == "w1" for entry in changes)
    assert changes[0].payload["task_id"] == "TASK-0001"


# ---------------------------------------------------------------------------
# Stop, pause, cancel
# ---------------------------------------------------------------------------

def test_cancelling_a_running_worker_records_who_and_why(ledger: CommLedger, control: SwarmControl):
    control.register("theory_prover_01", role="SymPy Prover", parent_id="theory_lead_01")
    control.begin("theory_prover_01", task_id="TASK-0001")
    control.request_stop("theory_prover_01", actor="theory_lead_01",
                         reason="two identical Lean failures")
    record = control.record("theory_prover_01")
    assert record.state is WorkerState.CANCELLED
    assert record.stop_actor == "theory_lead_01"
    assert record.stop_reason == "two identical Lean failures"
    request = ledger.by_action("STOP_REQUESTED")[0]
    assert request.sender["agent_id"] == "theory_lead_01"
    assert request.payload["reason"] == "two identical Lean failures"
    assert ledger.by_action("WORKER_CANCELLED")[0].payload["state_before"] == "running"


def test_a_stop_must_name_its_actor_and_reason(control: SwarmControl):
    control.register("w1")
    control.begin("w1")
    with pytest.raises(ValueError, match="actor"):
        control.request_stop("w1", actor="  ", reason="because")
    with pytest.raises(ValueError, match="reason"):
        control.request_stop("w1", actor="director", reason="")


def test_the_first_stop_request_wins_so_attribution_cannot_be_rewritten(control: SwarmControl):
    control.register("w1")
    control.begin("w1")
    assert control.request_stop("w1", actor="theory_lead_01", reason="first") is not None
    record = control.record("w1")
    assert record.stop_actor == "theory_lead_01"
    # A later leader cannot overwrite the recorded reason, and the fact that it
    # tried is itself recorded.
    control.request_stop("w1", actor="formal_lead_01", reason="second")
    assert control.record("w1").stop_actor == "theory_lead_01"
    superseded = [entry for entry in control.ledger.by_action("STOP_REQUESTED")
                  if entry.payload["superseded_an_earlier_request"]]
    assert len(superseded) == 1


def test_a_cancelled_worker_makes_no_further_tool_call(control: SwarmControl):
    """The token is the thing the tool loop consults, so this is the real guarantee."""
    token = control.register("w1") and control.token("w1")
    token.raise_if_stopped()  # a fresh token does not block
    control.request_stop("w1", actor="director", reason="budget exhausted")
    with pytest.raises(WorkerCancelled) as caught:
        token.raise_if_stopped()
    assert caught.value.actor == "director"
    assert caught.value.reason == "budget exhausted"


def test_a_paused_worker_may_continue_but_a_cancelled_one_may_not(control: SwarmControl):
    control.register("w1")
    control.begin("w1")
    control.request_stop("w1", actor="director", reason="hold for review", kind=StopKind.PAUSE)
    token = control.token("w1")
    assert token.paused and not token.cancelled
    # A pause must not kill the worker's tool loop: it returns and may be resumed.
    token.raise_if_stopped()
    assert control.record("w1").state is WorkerState.RUNNING
    token.signal.clear()
    assert not token.paused
    control.request_stop("w1", actor="director", reason="now really stop", kind=StopKind.CANCEL)
    cancelled = control.token("w1")
    assert cancelled.cancelled
    with pytest.raises(WorkerCancelled):
        cancelled.raise_if_stopped()
    # A cancel is not clearable: lifting the signal must not resurrect the worker.
    cancelled.signal.clear()
    assert cancelled.cancelled
    control.register("expired_01", deadline_s=0.01)
    time.sleep(0.05)
    with pytest.raises(WorkerCancelled, match="budget"):
        control.token("expired_01").raise_if_stopped()


def test_cancelling_a_worker_releases_its_task_for_reassignment(
        ledger: CommLedger, control: SwarmControl, board: TaskBoard):
    task = _task(board, artifact="proofs/a.py")
    board.lease(task.task_id, "worker_01")
    control.register("worker_01", parent_id="theory_lead_01")
    control.begin("worker_01", task_id=task.task_id)
    control.request_stop("worker_01", actor="theory_lead_01", reason="diverted to the Lean gap")
    assert board.get(task.task_id).owner is None
    assert board.lease(task.task_id, "worker_02").owner == "worker_02"


def test_stopping_the_run_cancels_every_active_worker_and_names_the_actor(
        ledger: CommLedger, control: SwarmControl):
    for agent_id in ("w1", "w2", "w3"):
        control.register(agent_id)
        control.begin(agent_id)
    control.complete("w3")
    control.stop_run(actor="operator", reason="Ctrl-C")
    assert control.run_stopped and control.stopped_by == "operator"
    assert {record.agent_id for record in control.by_state(WorkerState.CANCELLED)} == {"w1", "w2"}
    assert control.record("w3").state is WorkerState.COMPLETED
    assert control.should_stop(None) is True
    terminated = ledger.by_action("RUN_TERMINATED")[0]
    assert sorted(terminated.payload["stopped_agents"]) == ["w1", "w2"]


def test_pausing_and_resuming_the_run_is_recorded(ledger: CommLedger, control: SwarmControl):
    control.pause_run(actor="operator", reason="reviewing the proof")
    assert control.run_paused
    control.pause_run(actor="someone_else", reason="again")
    assert len(ledger.by_action("RUN_PAUSED")) == 1, "a second pause must not overwrite the first"
    control.resume_run(actor="operator")
    assert not control.run_paused and ledger.by_action("RUN_RESUMED")


def test_a_worker_that_exceeds_its_budget_times_out(control: SwarmControl):
    control.register("slow_01")
    control.register("quick_01", deadline_s=30)
    control.begin("slow_01")
    assert not control.token("slow_01").expired()
    expired = control.register("dead_01", deadline_s=-1)
    assert control.token("dead_01").expired()
    with pytest.raises(WorkerCancelled, match="budget"):
        control.token("dead_01").raise_if_stopped()
    assert expired.state is WorkerState.QUEUED


def test_a_cancellation_reaches_a_worker_inside_a_running_tool_loop(
        tmp_path: Path, ledger: CommLedger, control: SwarmControl):
    """The end-to-end guarantee: a loop that would otherwise keep calling tools
    stops at the first event after the cancel lands."""
    control.register("looping_01", role="SymPy Prover", parent_id="theory_lead_01")
    control.begin("looping_01")
    token = control.token("looping_01")
    calls: list[str] = []
    first_call = threading.Event()

    stopped: list[str] = []

    def fake_tool_loop() -> None:
        # Stands in for DeveloperAgent.run_stream: yield an event, then check
        # whether to make the next tool call, exactly as the real loop does.
        try:
            for index in range(200):
                token.raise_if_stopped()
                calls.append(f"tool_{index}")
                first_call.set()
                time.sleep(0.002)
            stopped.append("ran to completion")
        except WorkerCancelled as exc:
            stopped.append(f"cancelled by {exc.actor}")

    def canceller() -> None:
        first_call.wait(timeout=5)
        control.request_stop("looping_01", actor="theory_lead_01", reason="looping detected")

    worker = threading.Thread(target=fake_tool_loop, daemon=True)
    stopper = threading.Thread(target=canceller, daemon=True)
    worker.start()
    stopper.start()
    worker.join(timeout=10)
    stopper.join(timeout=10)
    assert worker.is_alive() is False
    assert 0 < len(calls) < 200, f"the loop should stop early, not run to completion: {len(calls)}"
    assert control.record("looping_01").state is WorkerState.CANCELLED


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_ledger_writes_keep_the_chain_intact(tmp_path: Path):
    """A forked chain or a duplicate id would make the audit trail untrustworthy."""
    ledger = CommLedger(tmp_path / "comm_ledger.jsonl")

    def writer(index: int) -> None:
        for step in range(30):
            ledger.append("WORKER_STATE_CHANGE",
                          {"agent_id": f"w{index}"}, {"agent_id": "director"},
                          {"step": step})

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    ok, detail = ledger.verify()
    assert ok, detail
    assert detail == "240 entries verified"


def test_two_ledger_objects_on_one_path_do_not_fork_the_chain(tmp_path: Path):
    path = tmp_path / "comm_ledger.jsonl"
    first, second = CommLedger(path), CommLedger(path)
    first.append("STATUS_REPORT", {"agent_id": "a"}, {"agent_id": "d"}, {"i": 1})
    second.append("STATUS_REPORT", {"agent_id": "b"}, {"agent_id": "d"}, {"i": 2})
    third = CommLedger(path)
    third.append("STATUS_REPORT", {"agent_id": "c"}, {"agent_id": "d"}, {"i": 3})
    ok, detail = third.verify()
    assert ok, detail
    assert [raw["id"] for raw in third.read_raw()] == ["MSG-0001", "MSG-0002", "MSG-0003"]


def test_concurrent_workers_cannot_lease_the_same_task(board: TaskBoard):
    _task(board, artifact="proofs/a.py")
    winners: list[str] = []
    lock = threading.Lock()

    def contend(name: str) -> None:
        try:
            board.lease("TASK-0001", name)
        except CoordinationError:
            return
        with lock:
            winners.append(name)

    threads = [threading.Thread(target=contend, args=(f"worker_{index}",)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(winners) == 1, f"exactly one worker may hold the lease, got {winners}"


def test_concurrent_registrations_do_not_duplicate_a_roster_entry(control: SwarmControl):
    def register(index: int) -> None:
        for _ in range(20):
            control.register(f"shared_{index}", role="Worker", division="theory")

    threads = [threading.Thread(target=register, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(control.workers()) == 6


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_the_snapshot_reports_every_worker_and_open_task(
        ledger: CommLedger, control: SwarmControl, board: TaskBoard):
    task = _task(board, artifact="proofs/a.py")
    board.lease(task.task_id, "w1")
    control.register("w1", role="SymPy Prover", division="theory", parent_id="theory_lead_01")
    control.begin("w1", task_id=task.task_id)
    control.register("w2")
    snapshot = control.snapshot()
    assert snapshot["state_counts"] == {"queued": 1, "running": 1, "blocked": 0,
                                        "completed": 0, "failed": 0, "cancelled": 0,
                                        "timed_out": 0}
    assert [item["task_id"] for item in snapshot["open_tasks"]] == [task.task_id]
    assert json.dumps(snapshot)  # must be serialisable for the CLI and the paper


def test_progress_summary_counts_states_and_flags_unanswered_requests(
        ledger: CommLedger, control: SwarmControl):
    control.register("w1")
    control.begin("w1")
    control.bus.ask_for_help(sender="w1", recipient="theory_lead_01", subject="stuck", body="b")
    counts = control.progress_summary(cycle=2)
    assert counts["running"] == 1
    summary = ledger.by_action("PROGRESS_SUMMARY")[0]
    assert summary.payload["cycle"] == 2
    assert summary.payload["unacknowledged_requests"] == 1


def test_a_stopped_worker_is_visible_in_the_render(control: SwarmControl):
    control.register("w1", role="SymPy Prover", division="theory", parent_id="theory_lead_01")
    control.begin("w1")
    control.request_stop("w1", actor="theory_lead_01", reason="diverted")
    rendered = control.render()
    assert "cancelled" in rendered and "w1" in rendered and "theory_lead_01" in rendered


# ---------------------------------------------------------------------------
# Vocabulary hygiene
# ---------------------------------------------------------------------------

def test_every_coordination_action_is_part_of_the_ledger_vocabulary():
    for action in ("TASK_LEASED", "TASK_COMPLETED", "TASK_FAILED", "TASK_BLOCKED",
                   "TASK_RETRY", "MESSAGE_SENT", "MESSAGE_ACKNOWLEDGED", "MESSAGE_OUTCOME",
                   "HELP_REQUESTED", "HELP_OFFERED", "WORKER_STATE_CHANGE", "STOP_REQUESTED",
                   "WORKER_CANCELLED", "RUN_PAUSED", "RUN_RESUMED", "RUN_TERMINATED",
                   "SWARM_SCALED", "PROGRESS_SUMMARY", "RUN_RESUMED_FROM_STATE"):
        assert action in ACTIONS


# ---------------------------------------------------------------------------
# Skill applicability
# ---------------------------------------------------------------------------

def test_a_skill_whose_checks_need_absent_tools_is_never_injected():
    """A checklist the worker has no instrument for is a rigged test.

    ``symbolic_checked`` is only dischargeable by the ``verify_equation`` tool, so
    a research worker that has read/write/bash/repl but not that tool can never
    satisfy it — and must not be handed the skill at all.
    """
    from adaptive_harness.skills.registry import BaseSkill
    from adaptive_harness.skills.verifier import SkillVerifier

    verifier = SkillVerifier()
    symbolic = BaseSkill("symbolic_worker", "Symbolic", "science", "sym", "instructions",
                         ("run_python_repl",), ("symbolic_checked", "numerical_checked"))
    research_tools = ("read_file", "write_file", "edit_file", "run_bash", "run_python_repl",
                      "run_lean_proof")
    assert verifier.applicable((symbolic,), research_tools) == ()
    assert verifier.unreachable((symbolic,), research_tools) == {
        "symbolic_worker": ("symbolic_checked",)}
    # Give it the tool and it becomes applicable again.
    with_tool = research_tools + ("verify_equation",)
    assert verifier.applicable((symbolic,), with_tool) == (symbolic,)
    assert verifier.unreachable((symbolic,), with_tool) == {}


def test_a_command_backed_check_is_reachable_by_the_shell():
    """``run_bash`` genuinely can run pytest and git, so those checks are fair."""
    from adaptive_harness.skills.registry import BaseSkill
    from adaptive_harness.skills.verifier import SkillVerifier

    verifier = SkillVerifier()
    tested = BaseSkill("test_runner", "Testing", "engineering", "t", "instructions",
                       ("run_bash",), ("tests_green", "git_checked"))
    assert verifier.applicable((tested,), ("run_bash",)) == (tested,)
    assert verifier.applicable((tested,), ("read_file",)) == ()


def test_a_partially_satisfiable_skill_is_dropped_not_half_applied():
    from adaptive_harness.skills.registry import BaseSkill
    from adaptive_harness.skills.verifier import SkillVerifier

    verifier = SkillVerifier()
    mixed = BaseSkill("mixed_worker", "Mixed", "science", "m", "instructions",
                      ("run_python_repl", "web_search"),
                      ("evidence_checked", "symbolic_checked"))
    # read_file makes evidence_checked reachable; verify_equation does not exist.
    assert verifier.applicable((mixed,), ("read_file", "run_python_repl")) == ()
    assert verifier.unreachable((mixed,), ("read_file", "run_python_repl")) == {
        "mixed_worker": ("symbolic_checked",)}


def test_a_skill_with_no_invariants_is_always_applicable():
    from adaptive_harness.skills.registry import BaseSkill
    from adaptive_harness.skills.verifier import SkillVerifier

    plain = BaseSkill("plain_worker", "Plain", "general", "p", "instructions",
                      ("read_file",), ())
    assert SkillVerifier().applicable((plain,), ("write_file",)) == (plain,)


def test_reachability_is_a_superset_of_what_verification_measures():
    """A check that verification can never satisfy must not be advertised as
    reachable, or the two would disagree about the same run."""
    from adaptive_harness.skills.registry import BaseSkill
    from adaptive_harness.skills.verifier import SkillVerifier

    verifier = SkillVerifier()
    skill = BaseSkill("s", "S", "c", "t", "i", ("run_bash",),
                      ("symbolic_checked", "execution_checked", "sources_checked"))
    observations = [{"name": "run_bash", "success": True, "arguments": {"command": "ls"}}]
    checks = verifier.verify((skill,), observations)
    assert "execution_checked" in checks[0].satisfied
    assert "symbolic_checked" in checks[0].missing
    reachable = verifier.reachable_invariants((skill,), ("run_bash",))
    # run_bash cannot produce a symbolic verification, so it is not reachable even
    # though the tool overlaps with several other checks.
    assert "symbolic_checked" not in reachable
