"""Task board, routed messaging, and leader-initiated stop control.

The research swarm's own convergence loop decides *what* is missing. This module
supplies the three things it lacked for a Director to actually *coordinate*
that work:

* :class:`TaskBoard` — a shared board of uniquely identified assignments with a
  single owner, explicit dependencies, an artifact path, an evidence list, and a
  time-boxed lease. The lease is what makes duplicate work impossible rather
  than merely discouraged: a second worker asking for a task that is already
  leased, or for an artifact another live lease already owns, is refused.
* :class:`MessageBus` — addressed messages between a worker and the nearest
  relevant lead or peer, with acknowledgements and recorded outcomes. A message
  nobody answered is visible as such instead of vanishing into the log.
* :class:`SwarmControl` — lifecycle states and leader-initiated stop, pause and
  cancel that *reach* a running worker: the worker's tool loop checks a
  cancellation token before each tool call, and its child processes are killed so
  a stopped worker leaves nothing running behind it.

Two honesty constraints shape the API:

* Nothing here records hidden model reasoning. The ledger captures observable
  actions — a lease granted, a tool that failed, a transition that occurred — and
  progress summaries an agent chose to report. It cannot attest to a thought.
* A cancellation is a *request to stop*, not a guarantee of immediate
  preemption. A tool call already in flight is allowed to return, and the loop
  then breaks before the next call. Killing the process group is what stops the
  work itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from adaptive_harness.research.ledger import CommLedger

TASK_BOARD_FILENAME = "task_board.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _monotonic() -> float:
    return time.monotonic()


class WorkerState(str, Enum):
    """Lifecycle of a single worker, from queue to a terminal outcome.

    The distinction between these values is what makes a stalled swarm legible.
    A worker that never ran, one that is waiting on a dependency, one that ran
    and succeeded, and one that was deliberately stopped all leave different
    evidence, and collapsing them into a boolean "busy" hides exactly the
    failures an operator needs to see.
    """

    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL_STATES

    @property
    def active(self) -> bool:
        return self in _ACTIVE_STATES


_TERMINAL_STATES = frozenset({
    WorkerState.COMPLETED, WorkerState.FAILED, WorkerState.CANCELLED, WorkerState.TIMED_OUT})
_ACTIVE_STATES = frozenset({WorkerState.QUEUED, WorkerState.RUNNING, WorkerState.BLOCKED})

#: State transitions the control plane permits. An unlisted transition is a bug
#: in a caller, not a recoverable condition, so it raises rather than being
#: silently coerced — a recorded audit trail that quietly accepts impossible
#: sequences is worse than no audit trail.
LEGAL_TRANSITIONS: Mapping[WorkerState, frozenset[WorkerState]] = {
    WorkerState.QUEUED: frozenset({WorkerState.RUNNING, WorkerState.BLOCKED,
                                   WorkerState.CANCELLED, WorkerState.TIMED_OUT}),
    WorkerState.RUNNING: frozenset({WorkerState.COMPLETED, WorkerState.FAILED,
                                    WorkerState.BLOCKED, WorkerState.CANCELLED,
                                    WorkerState.TIMED_OUT}),
    WorkerState.BLOCKED: frozenset({WorkerState.QUEUED, WorkerState.RUNNING,
                                    WorkerState.CANCELLED, WorkerState.TIMED_OUT}),
    WorkerState.COMPLETED: frozenset(),
    WorkerState.FAILED: frozenset({WorkerState.QUEUED}),
    WorkerState.CANCELLED: frozenset(),
    WorkerState.TIMED_OUT: frozenset({WorkerState.QUEUED}),
}


class StopKind(str, Enum):
    """What a leader asked for. ``PAUSE`` resumes; ``CANCEL`` does not."""

    PAUSE = "pause"
    CANCEL = "cancel"


class CoordinationError(RuntimeError):
    """An operation violated the coordination contract."""


class WorkerCancelled(CoordinationError):
    """Raised inside a worker's tool loop to unwind it without further tools."""

    #: Tells the agent layer that this is an abort, not a failure. A cancelled
    #: worker must never be retried, and must never be reported as having failed
    #: on its own merits. The flag is the whole contract, so this module stays
    #: independent of the agent layer that honours it.
    abort_subagent = True

    def __init__(self, agent_id: str, reason: str, actor: str):
        super().__init__(f"{agent_id} was stopped by {actor}: {reason}")
        self.agent_id = agent_id
        self.reason = reason
        self.actor = actor


# ---------------------------------------------------------------------------
# Stop authority
# ---------------------------------------------------------------------------

@dataclass
class StopSignal:
    """A standing instruction that reaches a worker wherever it is running.

    The signal is a plain object guarded by its own lock rather than a
    ``threading.Event`` alone, because the *who* and *why* must survive to the
    ledger: "the Formal Proof Lead cancelled theory_prover_01 after two
    identical Lean failures" is auditable, "stopped" is not.
    """

    kind: StopKind = StopKind.PAUSE
    actor: str = ""
    reason: str = ""
    at: str = field(default_factory=_utc_now)
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def fire(self, kind: StopKind, actor: str, reason: str) -> bool:
        """Record a stop request. The first request wins; later ones are ignored.

        First-writer-wins matters: a second leader cannot overwrite the recorded
        attribution of an earlier stop, so the audit trail cannot be rewritten by
        whoever happened to cancel last.
        """
        with self._lock:
            if self._event.is_set():
                return False
            self.kind = kind
            self.actor = actor
            self.reason = reason
            self.at = _utc_now()
            self._event.set()
            return True

    def clear(self) -> None:
        """Lift a pause so a paused worker may continue. A cancel is not clearable."""
        with self._lock:
            if self._event.is_set() and self.kind is StopKind.PAUSE:
                self.kind = StopKind.PAUSE
                self.actor = ""
                self.reason = ""
                self.at = _utc_now()
                self._event.clear()


class CancellationToken:
    """What a worker's tool loop consults before every tool call."""

    def __init__(self, agent_id: str, deadline_s: float | None = None):
        self.agent_id = agent_id
        self.deadline_s = deadline_s
        self.signal = StopSignal()
        self._started = _monotonic()

    @property
    def cancelled(self) -> bool:
        return self.signal.requested and self.signal.kind is StopKind.CANCEL

    @property
    def paused(self) -> bool:
        return self.signal.requested and self.signal.kind is StopKind.PAUSE

    @property
    def elapsed_s(self) -> float:
        return _monotonic() - self._started

    def expired(self) -> bool:
        return self.deadline_s is not None and self.elapsed_s > self.deadline_s

    def request(self, kind: StopKind, actor: str, reason: str) -> bool:
        return self.signal.fire(kind, actor, reason)

    def raise_if_stopped(self) -> None:
        """Unwind the worker when cancelled. Raise before each tool call.

        A paused worker returns instead of raising, so the tool loop can wait out
        the pause rather than die of it.
        """
        if self.cancelled:
            raise WorkerCancelled(self.agent_id, self.signal.reason, self.signal.actor)
        if self.expired():
            raise WorkerCancelled(self.agent_id, f"exceeded its {self.deadline_s}s budget",
                                 "swarm_control")


# ---------------------------------------------------------------------------
# Task board
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """One uniquely identified unit of work on the shared board."""

    task_id: str
    title: str
    division: str
    gap: str
    created_by: str
    owner: str | None = None
    state: WorkerState = WorkerState.QUEUED
    dependencies: tuple[str, ...] = ()
    artifact: str = ""
    evidence_ids: tuple[str, ...] = ()
    acceptance: str = ""
    contacts: tuple[str, ...] = ()
    # A simulation's hypothesis, sampling model, seed, and expected bound, written
    # down *before* the script runs. Recording it on the board is what makes a
    # post-hoc rationalisation detectable: a prediction invented after seeing the
    # data has no earlier entry to match.
    hypothesis: str = ""
    attempts: int = 0
    lease_expires_at: float = 0.0
    leased_at: str = ""
    finished_at: str = ""
    note: str = ""

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    @property
    def leased(self) -> bool:
        return self.owner is not None and self.lease_expires_at > _monotonic()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["dependencies"] = list(self.dependencies)
        payload["evidence_ids"] = list(self.evidence_ids)
        payload["contacts"] = list(self.contacts)
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Task":
        return cls(
            task_id=str(raw.get("task_id", "")),
            title=str(raw.get("title", "")),
            division=str(raw.get("division", "")),
            gap=str(raw.get("gap", "")),
            created_by=str(raw.get("created_by", "")),
            owner=raw.get("owner") or None,
            state=WorkerState(str(raw.get("state", WorkerState.QUEUED.value))),
            dependencies=tuple(raw.get("dependencies", ()) or ()),
            artifact=str(raw.get("artifact", "")),
            evidence_ids=tuple(raw.get("evidence_ids", ()) or ()),
            acceptance=str(raw.get("acceptance", "")),
            contacts=tuple(raw.get("contacts", ()) or ()),
            hypothesis=str(raw.get("hypothesis", "")),
            attempts=int(raw.get("attempts", 0) or 0),
            lease_expires_at=float(raw.get("lease_expires_at", 0.0) or 0.0),
            leased_at=str(raw.get("leased_at", "")),
            finished_at=str(raw.get("finished_at", "")),
            note=str(raw.get("note", "")),
        )


class TaskBoard:
    """The shared, lease-protected assignment board.

    The board is the only sanctioned way to claim work. Because a lease is both
    owner-bound and artifact-bound, two consequences follow that the previous
    round-robin assignment could not offer:

    * a task already leased by a live worker cannot be leased again;
    * a task whose *artifact* is already leased by another live task cannot be
      leased either, even if the task ids differ — which is precisely the
      collision that used to send two theory workers to the same proof file.
    """

    def __init__(self, *, lease_s: float = 900.0, max_attempts: int = 3):
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.lease_s = lease_s
        self.max_attempts = max_attempts
        self._tasks: dict[str, Task] = {}
        self._artifacts: dict[str, str] = {}  # artifact path -> task_id
        self._lock = threading.RLock()
        self._counter = 0

    # -- ids and lookup -----------------------------------------------------
    def _next_id(self) -> str:
        self._counter += 1
        return f"TASK-{self._counter:04d}"

    def adopt_id(self, task_id: str) -> None:
        """Raise the id counter past ``task_id`` after a reload.

        Recovery must not reissue an identifier that already appears in the
        ledger: a repeated ``TASK-0007`` pointing at different work would make
        the audit trail ambiguous.
        """
        digits = "".join(character for character in str(task_id) if character.isdigit())
        if digits:
            self._counter = max(self._counter, int(digits))

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def all(self) -> tuple[Task, ...]:
        with self._lock:
            return tuple(self._tasks[key] for key in sorted(self._tasks))

    def by_owner(self, owner: str) -> tuple[Task, ...]:
        with self._lock:
            return tuple(task for task in self.all() if task.owner == owner)

    def open_tasks(self) -> tuple[Task, ...]:
        with self._lock:
            return tuple(task for task in self.all() if not task.terminal)

    def artifact_owner(self, artifact: str) -> str | None:
        with self._lock:
            task_id = self._artifacts.get(artifact)
            return task_id if task_id and not self._released(task_id) else None

    def _released(self, task_id: str) -> bool:
        """Whether a task has given up its artifact claim.

        A *completed* task still pins its artifact: the file is the evidence and
        must stay attributable to the task that produced it. A task that is merely
        queued does not pin anything — the Director's reservation is a plan, and
        insisting on it would make a planned task un-replannable.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return True
        return task.state.terminal and task.state is not WorkerState.COMPLETED

    def _expired(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        return task is None or task.owner is None or task.lease_expires_at <= _monotonic()

    # -- creation -----------------------------------------------------------
    def create(self, *, title: str, division: str, gap: str, created_by: str,
               artifact: str = "", dependencies: Sequence[str] = (),
               acceptance: str = "", contacts: Sequence[str] = (),
               evidence_ids: Sequence[str] = (), hypothesis: str = "") -> Task:
        """Register a new task. An artifact already claimed cannot be re-claimed."""
        if not title or not title.strip():
            raise ValueError("A task needs a title")
        with self._lock:
            if artifact and artifact in self._artifacts and not self._released(self._artifacts[artifact]):
                raise CoordinationError(
                    f"artifact {artifact!r} is already owned by {self._artifacts[artifact]}")
            for dependency in dependencies:
                if dependency not in self._tasks:
                    raise CoordinationError(f"unknown dependency {dependency!r}")
            task = Task(task_id=self._next_id(), title=title.strip(), division=division,
                        gap=gap, created_by=created_by, dependencies=tuple(dependencies),
                        artifact=artifact, acceptance=acceptance, contacts=tuple(contacts),
                        evidence_ids=tuple(evidence_ids), hypothesis=hypothesis)
            self._tasks[task.task_id] = task
            if artifact:
                self._artifacts[artifact] = task.task_id
            return task

    # -- leasing ------------------------------------------------------------
    def unmet_dependencies(self, task_id: str) -> tuple[str, ...]:
        """Dependencies of ``task_id`` that have not reached a successful state."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return ()
            return tuple(dependency for dependency in task.dependencies
                         if (self._tasks.get(dependency) is not None
                             and self._tasks[dependency].state is not WorkerState.COMPLETED))

    def lease(self, task_id: str, owner: str, *, ttl_s: float | None = None) -> Task:
        """Take exclusive ownership of a task, or refuse with a reason.

        Refusal is raised rather than returned as ``None`` because every refusal
        is a coordination event the Director needs to see: a worker that quietly
        got nothing looks identical to a worker that found nothing to do.
        """
        if not owner or not owner.strip():
            raise ValueError("A lease needs an owner")
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise CoordinationError(f"unknown task {task_id!r}")
            if task.attempts >= self.max_attempts:
                raise CoordinationError(
                    f"{task_id} exhausted its {self.max_attempts} attempts")
            if task.terminal:
                raise CoordinationError(
                    f"{task_id} is {task.state.value}; a finished task cannot be leased")
            if task.leased and task.owner != owner:
                raise CoordinationError(
                    f"{task_id} is already leased by {task.owner} until it expires")
            unmet = self.unmet_dependencies(task_id)
            if unmet:
                raise CoordinationError(
                    f"{task_id} is blocked on {', '.join(unmet)}")
            task.owner = owner
            task.state = WorkerState.QUEUED
            task.attempts += 1
            task.lease_expires_at = _monotonic() + (ttl_s if ttl_s is not None else self.lease_s)
            task.leased_at = _utc_now()
            task.finished_at = ""
            if task.artifact:
                self._artifacts[task.artifact] = task_id
            return task

    def lease_any(self, owner: str, *, division: str | None = None,
                  gap: str | None = None) -> Task:
        """Lease the first unclaimed runnable task for ``owner``.

        Selection is deterministic — lowest task id first — so two workers racing
        for the same board take different tasks instead of both taking the first.
        """
        with self._lock:
            for task in self.all():
                if task.terminal or task.state is WorkerState.BLOCKED:
                    continue
                if division and task.division != division:
                    continue
                if gap and task.gap != gap:
                    continue
                if task.leased and task.owner != owner:
                    continue
                if task.attempts >= self.max_attempts:
                    continue
                if self.unmet_dependencies(task.task_id):
                    continue
                return self.lease(task.task_id, owner)
        raise CoordinationError(f"no runnable task is available for {owner}")

    def renew(self, task_id: str, owner: str, *, ttl_s: float | None = None) -> Task:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.owner != owner:
                raise CoordinationError(f"{owner} does not hold {task_id}")
            task.lease_expires_at = _monotonic() + (ttl_s if ttl_s is not None else self.lease_s)
            return task

    def release(self, task_id: str, owner: str, *, reason: str = "") -> Task:
        """Give a task back without completing it, so another worker may take it."""
        with self._lock:
            task = self._tasks[task_id]
            if task.owner not in (None, owner):
                raise CoordinationError(f"{owner} does not hold {task_id}")
            task.owner = None
            task.state = WorkerState.QUEUED
            task.lease_expires_at = 0.0
            if reason:
                task.note = reason
            self._maybe_free_artifact(task)
            return task

    def block(self, task_id: str, owner: str, reason: str) -> Task:
        with self._lock:
            task = self._tasks[task_id]
            task.state = WorkerState.BLOCKED
            task.note = reason
            task.lease_expires_at = 0.0
            task.owner = owner
            self._maybe_free_artifact(task)
            return task

    def complete(self, task_id: str, owner: str, *,
                 evidence_ids: Sequence[str] = (), artifact: str = "",
                 note: str = "") -> Task:
        with self._lock:
            task = self._tasks[task_id]
            if task.owner not in (None, owner):
                raise CoordinationError(f"{owner} does not hold {task_id}")
            task.state = WorkerState.COMPLETED
            task.evidence_ids = tuple(dict.fromkeys((*task.evidence_ids, *evidence_ids)))
            if artifact:
                task.artifact = artifact
                self._artifacts[artifact] = task_id
            task.finished_at = _utc_now()
            task.lease_expires_at = 0.0
            if note:
                task.note = note
            return task

    def fail(self, task_id: str, owner: str, reason: str, *, retry: bool = True) -> Task:
        """Record a failed attempt, returning the task to the queue if retryable."""
        with self._lock:
            task = self._tasks[task_id]
            task.state = WorkerState.QUEUED if retry and task.attempts < self.max_attempts \
                else WorkerState.FAILED
            task.finished_at = _utc_now() if task.state is WorkerState.FAILED else ""
            task.owner = None
            task.lease_expires_at = 0.0
            task.note = reason[:400]
            self._maybe_free_artifact(task)
            return task

    def _maybe_free_artifact(self, task: Task) -> None:
        """Release an artifact claim once the task is no longer holding it.

        A completed task keeps its artifact, because the artifact is the evidence
        and must stay pinned to the task that produced it. Only a relinquished or
        abandoned task frees the path for another worker.
        """
        if task.artifact and self._released(task.task_id):
            if self._artifacts.get(task.artifact) == task.task_id:
                del self._artifacts[task.artifact]

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"counter": self._counter, "lease_s": self.lease_s,
                    "max_attempts": self.max_attempts,
                    "tasks": [task.to_dict() for task in self.all()]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskBoard":
        board = cls(lease_s=float(raw.get("lease_s", 900.0) or 900.0),
                    max_attempts=int(raw.get("max_attempts", 3) or 3))
        for entry in raw.get("tasks", []) or []:
            task = Task.from_dict(entry)
            if not task.task_id:
                continue
            board._tasks[task.task_id] = task
            board.adopt_id(task.task_id)
            if task.artifact and not board._released(task.task_id):
                board._artifacts[task.artifact] = task.task_id
        board._counter = max(board._counter, int(raw.get("counter", 0) or 0))
        return board

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written to a sibling then renamed, so a crash mid-write cannot leave a
        # half-serialised board that recovery would read as authoritative.
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                             encoding="utf-8")
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TaskBoard | None":
        target = Path(path)
        if not target.is_file():
            return None
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        return cls.from_dict(raw)

    def recovered_from(self, ledger: CommLedger) -> int:
        """Reconcile the board with the ledger after an interrupted run.

        The ledger is the tamper-evident record and the board is a derived
        convenience, so a task the ledger shows finished wins over a task the
        board still shows as running. This is what stops a restart from blindly
        repeating completed work.
        """
        healed = 0
        with self._lock:
            for task in self.all():
                if task.state in (WorkerState.RUNNING, WorkerState.QUEUED,
                                  WorkerState.BLOCKED) and task.leased:
                    task.owner = None
                    task.lease_expires_at = 0.0
                    task.state = WorkerState.QUEUED
                    healed += 1
                for action, state in (("TASK_COMPLETED", WorkerState.COMPLETED),
                                      ("TASK_FAILED", WorkerState.FAILED),
                                      ("WORKER_CANCELLED", WorkerState.CANCELLED),
                                      ("TASK_BLOCKED", WorkerState.BLOCKED)):
                    for entry in ledger.by_payload("task_id", task.task_id):
                        if entry.action == action and str(entry.payload.get("state", state.value)) == state.value:
                            if task.state is not state:
                                task.state = state
                                healed += 1
                            break
            for task in self.all():
                if task.artifact and task.state is not WorkerState.COMPLETED:
                    self._artifacts.setdefault(task.artifact, task.task_id)
        return healed


# ---------------------------------------------------------------------------
# Routed messaging
# ---------------------------------------------------------------------------

class MessageKind(str, Enum):
    DIRECTIVE = "directive"
    STATUS = "status"
    REQUEST_FOR_HELP = "request_for_help"
    HELP_OFFER = "help_offer"
    RESPONSE = "response"
    ESCALATION = "escalation"
    REVIEW = "review"
    STOP = "stop"


@dataclass
class Message:
    """One addressed inter-agent message with an auditable disposition."""

    message_id: str
    kind: MessageKind
    sender: str
    recipient: str
    subject: str
    body: str
    created_at: str = field(default_factory=_utc_now)
    requires_ack: bool = False
    acknowledged_by: str = ""
    acknowledged_at: str = ""
    outcome: str = ""
    ledger_id: str = ""
    parent_id: str | None = None

    @property
    def acknowledged(self) -> bool:
        return bool(self.acknowledged_by)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


class MessageBus:
    """Addressed messaging between workers, leads, and the Director.

    Every send lands in the ledger with a message id, so an unanswered request is
    a queryable fact (``unacknowledged()``) rather than a gap in a reader's
    memory. Help requests are a distinct kind on purpose: they are the only
    messages that are *required* to be acknowledged, so a worker that asked for
    help and was ignored is visible instead of lost.
    """

    def __init__(self, ledger: CommLedger, *, parent_id: str = "",
                 routes: Mapping[str, str] | None = None):
        self.ledger = ledger
        self.default_parent_id = parent_id or None
        self._routes: dict[str, str] = dict(routes or {})
        self._messages: dict[str, Message] = {}
        self._inbox: dict[str, list[str]] = {}
        self._lock = threading.RLock()
        self._counter = 0

    def adopt_id(self, message_id: str) -> None:
        digits = "".join(character for character in str(message_id) if character.isdigit())
        if digits:
            self._counter = max(self._counter, int(digits))

    def register_route(self, agent_id: str, contact: str) -> None:
        """Record who ``agent_id`` should escalate to when it is stuck."""
        self._routes[agent_id] = contact

    def contact_for(self, agent_id: str) -> str:
        """The nearest relevant lead for ``agent_id``.

        The route table is authoritative; the Director is the terminal fallback so
        a message from an unregistered agent still reaches a decision-maker rather
        than dying in an empty inbox.
        """
        return self._routes.get(agent_id, "executive_director_01")

    def send(self, *, sender: str, recipient: str, kind: MessageKind, subject: str,
             body: str, requires_ack: bool | None = None,
             parent_id: str | None = None) -> Message:
        if not sender.strip() or not recipient.strip():
            raise ValueError("a message needs a sender and a recipient")
        if sender == recipient:
            raise ValueError("an agent cannot address a message to itself")
        with self._lock:
            self._counter += 1
            message = Message(
                message_id=f"MSGQ-{self._counter:04d}", kind=kind, sender=sender,
                recipient=recipient, subject=subject, body=body[:2000],
                requires_ack=(kind in (MessageKind.REQUEST_FOR_HELP, MessageKind.ESCALATION)
                              if requires_ack is None else bool(requires_ack)),
                parent_id=parent_id if parent_id is not None else self.default_parent_id)
            action = {"request_for_help": "HELP_REQUESTED",
                      "help_offer": "HELP_OFFERED"}.get(kind.value, "MESSAGE_SENT")
            entry = self.ledger.append(
                action, {"agent_id": sender, "role": sender}, {"agent_id": recipient, "role": recipient},
                {"message_id": message.message_id, "kind": kind.value, "subject": subject,
                 "body": message.body, "requires_ack": message.requires_ack},
                parent_id=message.parent_id)
            message.ledger_id = entry.id
            self._messages[message.message_id] = message
            self._inbox.setdefault(recipient, []).append(message.message_id)
        return message

    def ask_for_help(self, *, sender: str, recipient: str, subject: str, body: str) -> Message:
        return self.send(sender=sender, recipient=recipient, kind=MessageKind.REQUEST_FOR_HELP,
                         subject=subject, body=body)

    def acknowledge(self, message_id: str, agent_id: str, note: str = "") -> Message:
        """Confirm receipt. The first acknowledgement wins, so credit is unambiguous."""
        with self._lock:
            message = self._require(message_id)
            if message.recipient != agent_id:
                raise CoordinationError(
                    f"{agent_id} is not the recipient of {message_id} "
                    f"(addressed to {message.recipient})")
            if message.acknowledged:
                return message
            message.acknowledged_by = agent_id
            message.acknowledged_at = _utc_now()
            self.ledger.append(
                "MESSAGE_ACKNOWLEDGED", {"agent_id": agent_id, "role": agent_id},
                {"agent_id": message.sender, "role": message.sender},
                {"message_id": message_id, "in_reply_to": message.ledger_id,
                 "kind": message.kind.value, "subject": message.subject,
                 "note": note[:400]}, parent_id=message.ledger_id)
        return message

    def record_outcome(self, message_id: str, agent_id: str, outcome: str) -> Message:
        """Record what came of a message, so 'asked and got no answer' is distinct
        from 'asked and the answer was no'."""
        with self._lock:
            message = self._require(message_id)
            if agent_id not in (message.sender, message.recipient):
                raise CoordinationError(f"{agent_id} is not a party to {message_id}")
            message.outcome = outcome[:600]
            self.ledger.append(
                "MESSAGE_OUTCOME", {"agent_id": agent_id, "role": agent_id},
                {"agent_id": message.sender if agent_id == message.recipient else message.recipient,
                 "role": message.sender if agent_id == message.recipient else message.recipient},
                {"message_id": message_id, "in_reply_to": message.ledger_id,
                 "kind": message.kind.value, "subject": message.subject,
                 "outcome": message.outcome}, parent_id=message.ledger_id)
        return message

    def _require(self, message_id: str) -> Message:
        message = self._messages.get(message_id)
        if message is None:
            raise KeyError(f"unknown message {message_id!r}")
        return message

    def inbox(self, agent_id: str, *, unread_only: bool = True) -> tuple[Message, ...]:
        with self._lock:
            ids = self._inbox.get(agent_id, [])
            messages = [self._messages[key] for key in ids if key in self._messages]
            return tuple(item for item in messages if not (unread_only and item.acknowledged))

    def unacknowledged(self) -> tuple[Message, ...]:
        """Messages that required an answer and never got one."""
        with self._lock:
            return tuple(message for message in self._messages.values()
                         if message.requires_ack and not message.acknowledged)

    def get(self, message_id: str) -> Message | None:
        with self._lock:
            return self._messages.get(message_id)

    def all(self) -> tuple[Message, ...]:
        with self._lock:
            return tuple(self._messages[key] for key in sorted(self._messages))

    def load(self, entries: Iterable[Any]) -> None:
        """Rebuild the bus from ledger entries during recovery.

        Accepts either :class:`LedgerEntry` objects or the raw dicts
        ``CommLedger.read_raw`` returns, because a caller recovering from a fresh
        process naturally has the raw form and should not have to re-wrap it.

        Only coordination messages are replayed — proof receipts and gate
        evaluations are not conversations and must not appear in an inbox.
        """
        interesting = {"MESSAGE_SENT", "MESSAGE_ACKNOWLEDGED", "MESSAGE_OUTCOME",
                       "HELP_REQUESTED", "HELP_OFFERED"}
        with self._lock:
            for raw in entries:
                action = raw.get("action") if isinstance(raw, Mapping) \
                    else getattr(raw, "action", "")
                if action not in interesting:
                    continue
                payload = raw.get("payload", {}) if isinstance(raw, Mapping) \
                    else getattr(raw, "payload", {})
                sender = raw.get("sender", {}) if isinstance(raw, Mapping) \
                    else getattr(raw, "sender", {})
                recipient = raw.get("recipient", {}) if isinstance(raw, Mapping) \
                    else getattr(raw, "recipient", {})
                timestamp = raw.get("timestamp", "") if isinstance(raw, Mapping) \
                    else getattr(raw, "timestamp", "")
                parent_id = raw.get("parent_id") if isinstance(raw, Mapping) \
                    else getattr(raw, "parent_id", None)
                entry_id = raw.get("id", "") if isinstance(raw, Mapping) \
                    else getattr(raw, "id", "")
                payload = payload or {}
                message_id = str(payload.get("message_id", ""))
                if not message_id:
                    continue
                self.adopt_id(message_id)
                message = self._messages.get(message_id)
                if message is None:
                    kind = str(payload.get("kind", MessageKind.DIRECTIVE.value))
                    message = Message(
                        message_id=message_id,
                        kind=MessageKind(kind) if kind in MessageKind.__members__.values()
                        else MessageKind.DIRECTIVE,
                        sender=str(sender.get("agent_id", "")),
                        recipient=str(recipient.get("agent_id", "")),
                        subject=str(payload.get("subject", "")),
                        body=str(payload.get("body", "")),
                        requires_ack=bool(payload.get("requires_ack", False)),
                        parent_id=parent_id, ledger_id=str(entry_id))
                    self._messages[message_id] = message
                    self._inbox.setdefault(message.recipient, []).append(message_id)
                    if message.recipient:
                        self._routes.setdefault(message.recipient, message.sender)
                if action == "MESSAGE_ACKNOWLEDGED":
                    message.acknowledged_by = str(sender.get("agent_id", ""))
                    message.acknowledged_at = str(timestamp)
                if action == "MESSAGE_OUTCOME":
                    message.outcome = str(payload.get("outcome", ""))


# ---------------------------------------------------------------------------
# Lifecycle and stop control
# ---------------------------------------------------------------------------

@dataclass
class WorkerRecord:
    """What the control plane knows about one agent."""

    agent_id: str
    role: str = ""
    division: str = ""
    parent_id: str = ""
    task_id: str = ""
    state: WorkerState = WorkerState.QUEUED
    thread_id: int | None = None
    registered_at: str = field(default_factory=_utc_now)
    started_at: str = ""
    finished_at: str = ""
    stop_kind: str = ""
    stop_actor: str = ""
    stop_reason: str = ""
    tool_calls: int = 0
    error: str = ""
    artifact: str = ""
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["evidence_ids"] = list(self.evidence_ids)
        payload.pop("thread_id", None)
        return payload


class SwarmControl:
    """Lifecycle bookkeeping and leader-initiated stop for the whole swarm.

    One :class:`SwarmControl` per run owns three things: the roster with each
    agent's state, the cancellation token each running worker consults, and the
    run-level pause/stop switch. Every transition is written to the ledger with
    its actor and reason, so the published record shows who stopped what.
    """

    def __init__(self, ledger: CommLedger, *, board: TaskBoard | None = None,
                 bus: MessageBus | None = None, on_status: Callable[[str], None] | None = None):
        self.ledger = ledger
        self.board = board
        self.bus = bus
        self.on_status = on_status
        self._workers: dict[str, WorkerRecord] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = threading.RLock()
        self._run_pause = StopSignal()
        self._run_stop = StopSignal()
        self.stopped_by = ""

    # -- roster -------------------------------------------------------------
    def register(self, agent_id: str, *, role: str = "", division: str = "",
                 parent_id: str = "", deadline_s: float | None = None) -> WorkerRecord:
        """Add an agent to the roster, idempotently.

        A repeat registration must not replace a live cancellation token. The
        Director re-registers its leaders on every construction, and handing a
        worker a fresh token on re-registration would silently discard a pending
        stop request — the worker would carry on after being told to stop.
        """
        with self._lock:
            existing = self._workers.get(agent_id)
            if existing is not None:
                return existing
            record = WorkerRecord(agent_id=agent_id, role=role, division=division,
                                  parent_id=parent_id)
            self._workers[agent_id] = record
            if agent_id not in self._tokens:
                self._tokens[agent_id] = CancellationToken(agent_id, deadline_s=deadline_s)
            elif deadline_s is not None:
                self._tokens[agent_id].deadline_s = deadline_s
            return record

    def record(self, agent_id: str) -> WorkerRecord | None:
        with self._lock:
            return self._workers.get(agent_id)

    def token(self, agent_id: str) -> CancellationToken:
        with self._lock:
            token = self._tokens.get(agent_id)
            if token is None:
                token = CancellationToken(agent_id)
                self._tokens[agent_id] = token
            return token

    def workers(self) -> tuple[WorkerRecord, ...]:
        with self._lock:
            return tuple(self._workers[key] for key in sorted(self._workers))

    def by_state(self, state: WorkerState) -> tuple[WorkerRecord, ...]:
        return tuple(record for record in self.workers() if record.state is state)

    def active(self) -> tuple[WorkerRecord, ...]:
        return tuple(record for record in self.workers() if record.state.active)

    def running(self) -> tuple[WorkerRecord, ...]:
        return self.by_state(WorkerState.RUNNING)

    # -- transitions --------------------------------------------------------
    def _transition(self, agent_id: str, state: WorkerState, *, task_id: str = "",
                    error: str = "", tool_calls: int | None = None,
                    artifact: str = "", evidence_ids: Sequence[str] = ()) -> WorkerRecord:
        with self._lock:
            record = self._workers.get(agent_id)
            if record is None:
                record = self.register(agent_id)
            if state is not record.state:
                allowed = LEGAL_TRANSITIONS.get(record.state, frozenset())
                if state not in allowed:
                    raise CoordinationError(
                        f"{agent_id} cannot move from {record.state.value} to {state.value}")
            previous = record.state
            record.state = state
            if task_id:
                record.task_id = task_id
            if tool_calls is not None:
                record.tool_calls = tool_calls
            if artifact:
                record.artifact = artifact
            if evidence_ids:
                record.evidence_ids = tuple(dict.fromkeys((*record.evidence_ids, *evidence_ids)))
            if state is WorkerState.RUNNING and not record.started_at:
                record.started_at = _utc_now()
                record.thread_id = threading.get_ident()
            if state.terminal:
                record.finished_at = _utc_now()
                record.thread_id = None
            if error:
                record.error = error[:400]
        if previous is not state:
            self.ledger.append(
                "WORKER_STATE_CHANGE", {"agent_id": agent_id, "role": record.role},
                {"agent_id": record.parent_id or "executive_director_01",
                 "role": record.parent_id or "Chief Scientist"},
                {"from": previous.value, "to": state.value, "task_id": record.task_id,
                 "division": record.division, "tool_calls": record.tool_calls,
                 "error": record.error or None})
        return record

    def begin(self, agent_id: str, *, task_id: str = "", tool_calls: int = 0) -> WorkerRecord:
        """Start a worker's tool loop, requeueing it first if it is retryable.

        A worker whose last attempt failed or timed out is dispatched again on the
        next cycle, so ``begin`` has to move it out of its terminal state before
        starting. That is done as an explicit, recorded requeue rather than by
        quietly allowing FAILED to RUNNING, because the requeue is the moment a
        retry begins and an audit needs to see it.
        """
        with self._lock:
            record = self._workers.get(agent_id)
        if record is not None and record.state in (WorkerState.FAILED, WorkerState.TIMED_OUT):
            self._transition(agent_id, WorkerState.QUEUED)
        return self._transition(agent_id, WorkerState.RUNNING, task_id=task_id,
                                tool_calls=tool_calls)

    def complete(self, agent_id: str, *, artifact: str = "",
                 evidence_ids: Sequence[str] = (), tool_calls: int | None = None) -> WorkerRecord:
        return self._transition(agent_id, WorkerState.COMPLETED, artifact=artifact,
                                evidence_ids=evidence_ids, tool_calls=tool_calls)

    def fail(self, agent_id: str, reason: str, *, tool_calls: int | None = None) -> WorkerRecord:
        return self._transition(agent_id, WorkerState.FAILED, error=reason, tool_calls=tool_calls)

    def timeout(self, agent_id: str, reason: str) -> WorkerRecord:
        return self._transition(agent_id, WorkerState.TIMED_OUT, error=reason)

    def block(self, agent_id: str, reason: str) -> WorkerRecord:
        return self._transition(agent_id, WorkerState.BLOCKED, error=reason)

    def settle_worker(self, agent_id: str, state: WorkerState, *, reason: str = "",
                      artifact: str = "", evidence_ids: Sequence[str] = (),
                      tool_calls: int | None = None) -> WorkerRecord:
        """Record a worker's outcome without insisting the state machine allow it.

        A worker's lifecycle state and its task's state are different facts, and
        a caller that learned the task failed may be looking at a worker already
        cancelled by a leader. Raising there would discard a real task transition
        to protect a bookkeeping nicety. The mismatch is therefore *recorded* —
        the ledger shows both what happened to the worker and what happened to the
        task — instead of being raised or silently papered over.
        """
        with self._lock:
            record = self._workers.get(agent_id)
            current = record.state if record else None
        if current is not None and state not in LEGAL_TRANSITIONS.get(current, frozenset()):
            self.ledger.append(
                "STATUS_REPORT", {"agent_id": agent_id, "role": record.role if record else ""},
                {"agent_id": record.parent_id if record and record.parent_id else "executive_director_01",
                 "role": "Chief Scientist"},
                {"status": "outcome_recorded_outside_lifecycle", "worker_state": current.value,
                 "reported_outcome": state.value, "reason": reason[:300]})
            if record is not None and tool_calls is not None:
                record.tool_calls = tool_calls
            return record if record is not None else self.register(agent_id)
        return self._transition(agent_id, state, error=reason, tool_calls=tool_calls,
                                artifact=artifact, evidence_ids=evidence_ids)

    # -- stop authority -----------------------------------------------------
    def request_stop(self, agent_id: str, *, actor: str, reason: str,
                     kind: StopKind = StopKind.CANCEL) -> WorkerRecord:
        """Stop, pause, or release one worker, and kill what it started.

        The token is set *before* the ledger write so a worker that is between
        tool calls stops as early as possible; the record follows immediately
        after, and the process group is killed last because that is the only step
        that can block. ``actor`` is not optional: an unattributable stop is not
        auditable, and the requirement is the point of the feature.
        """
        if not actor.strip():
            raise ValueError("a stop must name the actor that issued it")
        if not reason.strip():
            raise ValueError("a stop must state a reason")
        with self._lock:
            record = self._workers.get(agent_id)
            if record is None:
                record = self.register(agent_id)
        token = self.token(agent_id)
        first = token.request(kind, actor, reason)
        killed: tuple[int, ...] = ()
        if kind is StopKind.CANCEL:
            killed = self._terminate_agent_processes(agent_id)
        self.ledger.append(
            "STOP_REQUESTED", {"agent_id": actor, "role": actor},
            {"agent_id": agent_id, "role": record.role},
            {"agent_id_target": agent_id, "kind": kind.value, "reason": reason[:400],
             "state_at_request": record.state.value, "task_id": record.task_id,
             "superseded_an_earlier_request": not first,
             "processes_signalled": list(killed)})
        if kind is StopKind.CANCEL and record.state.active:
            state_before = record.state.value
            record = self._transition(agent_id, WorkerState.CANCELLED)
            # Attribution lives on the record as well as the ledger, so a status
            # render can answer "who stopped this and why" without replaying it.
            record.stop_kind = kind.value
            record.stop_actor = actor
            record.stop_reason = reason[:400]
            self.ledger.append(
                "WORKER_CANCELLED", {"agent_id": actor, "role": actor},
                {"agent_id": agent_id, "role": record.role},
                {"task_id": record.task_id, "reason": reason[:400], "actor": actor,
                 "kind": kind.value, "state_before": state_before})
            if record.task_id and self.board is not None:
                try:
                    self.board.release(record.task_id, agent_id, reason=f"stopped by {actor}")
                except (CoordinationError, KeyError):
                    # The task may already be settled, or may never have existed on
                    # this board. Either way there is nothing left to release, and
                    # losing the release must not lose the recorded cancellation.
                    pass
        self._emit(f"{actor} {'cancelled' if kind is StopKind.CANCEL else 'paused'} "
                   f"{agent_id}: {reason}")
        return record

    def _terminate_agent_processes(self, agent_id: str) -> tuple[int, ...]:
        """SIGKILL the process groups belonging to one specific worker.

        Attribution is by *scope*, not by thread: leaders and the Director run on
        the main thread, so a thread-keyed kill would also take down the
        verifiers' own subprocesses running there — which is how a passing proof
        script came to be reported as a non-zero exit during a live stop.
        """
        from adaptive_harness.tools.process import terminate_owned
        return terminate_owned(agent_id)

    def pause_run(self, *, actor: str, reason: str) -> None:
        """Pause the whole swarm. Workers finish their current tool call, then wait."""
        if not self._run_pause.fire(StopKind.PAUSE, actor, reason):
            return
        self.ledger.append("RUN_PAUSED", {"agent_id": actor, "role": actor},
                           {"agent_id": "all_agents", "role": "Swarm"},
                           {"reason": reason[:400], "actor": actor})
        self._emit(f"run paused by {actor}: {reason}")

    def resume_run(self, *, actor: str) -> None:
        if not self._run_pause.requested:
            return
        self._run_pause.clear()
        self.ledger.append("RUN_RESUMED", {"agent_id": actor, "role": actor},
                           {"agent_id": "all_agents", "role": "Swarm"}, {"actor": actor})
        self._emit(f"run resumed by {actor}")

    def stop_run(self, *, actor: str, reason: str) -> None:
        """Stop the run: signal every active worker, then latch the run closed."""
        if not self._run_stop.fire(StopKind.CANCEL, actor, reason):
            return
        self.stopped_by = actor
        targets = [record.agent_id for record in self.active()]
        for agent_id in targets:
            try:
                self.request_stop(agent_id, actor=actor,
                                  reason=f"run stopped: {reason}", kind=StopKind.CANCEL)
            except (CoordinationError, ValueError):  # pragma: no cover - defensive
                continue
        self.ledger.append("RUN_TERMINATED", {"agent_id": actor, "role": actor},
                           {"agent_id": "all_agents", "role": "Swarm"},
                           {"reason": reason[:400], "actor": actor,
                            "stopped_agents": targets})
        self._emit(f"run stopped by {actor}: {reason}")

    @property
    def run_stopped(self) -> bool:
        return self._run_stop.requested

    @property
    def run_paused(self) -> bool:
        return self._run_pause.requested

    def wait_while_paused(self, timeout_s: float = 30.0) -> bool:
        """Block a worker while the run is paused. False means 'stopped, not paused'."""
        if self.run_stopped:
            return False
        if not self.run_paused:
            return True
        return self._run_pause._event.wait(timeout_s)

    def should_stop(self, _report: Any = None) -> bool:
        """The ``should_stop`` hook the convergence loop calls between cycles."""
        return self.run_stopped

    # -- reporting ----------------------------------------------------------
    def _emit(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(message)

    def progress_summary(self, cycle: int = 0) -> dict[str, int]:
        counts = {state.value: 0 for state in WorkerState}
        for record in self.workers():
            counts[record.state.value] += 1
        self.ledger.append("PROGRESS_SUMMARY",
                           {"agent_id": "executive_director_01", "role": "Chief Scientist"},
                           {"agent_id": "invariants", "role": "Invariant Gate"},
                           {"cycle": cycle, "state_counts": counts,
                            "open_tasks": len(self.board.open_tasks()) if self.board else 0,
                            "unacknowledged_requests": len(self.bus.unacknowledged())
                            if self.bus else 0})
        return counts

    def snapshot(self) -> dict[str, Any]:
        """A JSON-ready view of the swarm for the CLI, the TUI, and the paper."""
        with self._lock:
            workers = [record.to_dict() for record in self.workers()]
        return {
            "run_stopped": self.run_stopped,
            "run_paused": self.run_paused,
            "stopped_by": self.stopped_by,
            "workers": workers,
            "state_counts": {state.value: sum(1 for item in workers
                                              if item["state"] == state.value)
                             for state in WorkerState},
            "open_tasks": [task.to_dict() for task in self.board.open_tasks()]
            if self.board else [],
            "unacknowledged_requests": [
                {"message_id": message.message_id, "sender": message.sender,
                 "recipient": message.recipient, "subject": message.subject}
                for message in self.bus.unacknowledged()] if self.bus else [],
        }

    def render(self) -> str:
        """One-screen status for an operator watching a run."""
        lines = []
        for record in self.workers():
            detail = f" task={record.task_id}" if record.task_id else ""
            if record.state is WorkerState.CANCELLED and record.stop_actor:
                detail += f" stopped_by={record.stop_actor} ({record.stop_reason})"
            lines.append(f"  {record.state.value:<10} {record.agent_id:<28} "
                         f"{record.role}{detail}")
        counts = self.snapshot()["state_counts"]
        lines.append("  " + "  ".join(f"{key}={value}" for key, value in counts.items()
                                      if value))
        if self.bus is not None:
            pending = self.bus.unacknowledged()
            if pending:
                lines.append(f"  unanswered help requests: {len(pending)}")
        return "\n".join(lines)
