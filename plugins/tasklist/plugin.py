"""tasklist — live task state for a run, in which a ``done`` has to be earned.

The harness already scans source for TODO markers. That finds intentions written
down *before* the work. It says nothing about the work a run actually did, and
nothing about a model that says "done" without having done anything. This plugin
is the other half: the state of the tasks the run is holding right now, with a
rule that a task cannot be closed on the strength of an assertion.

The rule is the point, and it is borrowed from :class:`SkillVerifier`, which
already refuses to accept an LLM claim that a check passed unless a *tool* was
observed doing it. Here the same principle is applied to a status transition: a
task reaches ``done`` only when the plugin has seen a successful call to a real
tool since that task entered ``in_progress``. The evidence is **observed**, not
asserted, because it arrives through the ``on_agent_event`` hook reading the
agent's own event stream rather than through anything the model says about
itself. A model that calls ``task_update(id, "done")`` having run nothing gets a
refusal naming what is missing, and a ``pre_tool`` hook denies the same call
before it is dispatched — a guard that only the handler enforces is a guard the
agent loop could route around.

State lives in a small JSON file so it survives a restart, and every mutation
is a read-modify-write under one lock. That lock is not decoration: the TUI
worker thread and the main thread both reach this state, and a read-modify-write
that interleaves loses an update silently. The write is atomic (a temp file
plus ``os.replace``) so a half-written file can never be read back as truth.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from adaptive_harness.plugins.types import ALLOW, HookVerdict

#: The plugin's own tools, and the tools that only observe. Neither can be
#: evidence that work happened: the first because a model could discharge its
#: own evidence by listing its own list, the second because reading a file
#: changes nothing. Everything else that succeeds counts, which is deliberate —
#: a new built-in or an MCP tool discharges a task without this file needing to
#: learn its name.
OBSERVER_TOOLS = frozenset({
    "read_file", "list_directory", "search_files", "read_full_output",
    "calculate", "task_update", "task_list", "task_clear",
})

PENDING = "pending"
IN_PROGRESS = "in_progress"
DONE = "done"
BLOCKED = "blocked"
STATUSES = (PENDING, IN_PROGRESS, DONE, BLOCKED)

#: Where a task may move. A strictly downward move is refused: a task that was
#: finished does not quietly become un-started, because the list is what a later
#: turn (or a person) reads to decide whether the work is real.
RANK = {PENDING: 0, IN_PROGRESS: 1, BLOCKED: 2, DONE: 2}
TERMINAL = frozenset({DONE, BLOCKED})

#: Retained evidence and history are bounded so the state file cannot grow
#: without limit over a long run. The bound is generous relative to a turn.
MAX_EVIDENCE = 200
MAX_HISTORY = 20
MAX_NOTES = 8

#: One lock for the whole module. The agent's worker thread calls the tools
#: while the main thread may be listing or clearing, and every path that touches
#: state takes this first. The state is a file, so the lock also serialises the
#: read-modify-write: two module instances pointed at one file still cannot
#: lose each other's updates.
_lock = threading.RLock()
_cache: Optional[dict[str, Any]] = None
_cache_path: str = ""
#: Set when the state file exists but cannot be read. Reported by ``task_list``
#: rather than swallowed, because a silently empty list reads as "no work was
#: ever tracked" when the truth is that the file is damaged.
_load_error: str = ""
#: Whether any task is currently open. The event hook runs on every step, so in
#: the common case it must do nothing at all — not even take a lock or touch
#: the disk — because no open task could be affected by a tool result.
_has_open_task = False


# --- state file -------------------------------------------------------------


def _state_path() -> Path:
    """Where the state lives.

    The override exists so a run, a test or a second workspace can keep its own
    list; the default is a plugin-owned directory under the harness config dir
    rather than a file in the workspace, so a repository the agent is pointed
    at never accumulates state it did not ask for.
    """
    override = os.environ.get("ADAPTIVE_HARNESS_TASKLIST_STATE", "").strip()
    if override:
        return Path(override).expanduser()
    from adaptive_harness.data.config import DEFAULT_CONFIG_DIR

    return Path(DEFAULT_CONFIG_DIR).expanduser() / "tasklist" / "state.json"


def _blank_state() -> dict[str, Any]:
    return {"version": 1, "seq": 0, "tasks": {}, "evidence": []}


def _load_locked() -> dict[str, Any]:
    """Read the state, reusing the in-memory copy when the path is unchanged.

    Callers must hold ``_lock``. A missing file is an empty list, not an error;
    a damaged one is reported, because the difference is the difference between
    "nothing was tracked" and "the tracking was lost".
    """
    global _cache, _cache_path, _load_error, _has_open_task
    path = str(_state_path())
    if _cache is not None and _cache_path == path:
        return _cache
    state = _blank_state()
    error = ""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raw = ""
    except OSError as exc:
        raw = ""
        error = f"The task state at {path} could not be read ({exc}); starting empty."
    if raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            parsed = None
            error = f"The task state at {path} is not valid JSON ({exc}); starting empty."
        if isinstance(parsed, dict) and isinstance(parsed.get("tasks"), dict):
            state = {
                "version": 1,
                "seq": int(parsed.get("seq") or 0),
                "tasks": {str(k): v for k, v in parsed["tasks"].items()
                          if isinstance(v, dict)},
                "evidence": [e for e in (parsed.get("evidence") or [])
                             if isinstance(e, dict)],
            }
    _cache, _cache_path, _load_error = state, path, error
    _has_open_task = any(
        str(task.get("status")) == IN_PROGRESS for task in state["tasks"].values())
    return state


def _write_locked(state: dict[str, Any]) -> None:
    """Persist the state atomically. Callers must hold ``_lock``."""
    global _cache, _cache_path
    path = Path(_state_path())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=".tasklist-", suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
        os.replace(temp_name, path)
    except OSError as exc:
        raise RuntimeError(f"Could not write task state to {path}: {exc}") from exc
    _cache, _cache_path = state, str(path)


def _now() -> float:
    return round(time.time(), 3)


# --- the rules --------------------------------------------------------------


def _evidence_since(state: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    """Successful real-tool calls observed after this task was started."""
    since = task.get("since")
    if since is None:
        return []
    return [entry for entry in state.get("evidence", [])
            if int(entry.get("seq", 0)) > int(since)]


def _transition_refusal(task: dict[str, Any], status: str, note: str) -> str:
    """Why this status change is not allowed, or ``""`` if it is.

    Two rules, both about the list meaning something. A task that was finished
    does not quietly become un-started, and a task that is only terminal while
    blocked says so in a note before it is re-opened — the one regression that
    is allowed has to be explicit and recorded, which is the difference between
    a change of mind and an erasure.
    """
    current = str(task.get("status", PENDING))
    if status == current:
        return ""
    if RANK[status] < RANK[current]:
        if current == BLOCKED and status == IN_PROGRESS:
            if not note.strip():
                return (
                    f"Task {task['id']!r} is blocked. Re-opening it is allowed, but it "
                    f"has to say why in `note`, so the block and its resolution are "
                    f"both in the history. Nothing was changed."
                )
            return ""
        return (
            f"Task {task['id']!r} is {current} and cannot go back to {status}: a task "
            f"that was marked {current} stays {current}. A finished task is not "
            f"silently un-finished. To redo the work, add a new task id. "
            f"Nothing was changed."
        )
    return ""


def _missing_evidence(state: dict[str, Any], task: dict[str, Any]) -> str:
    """Why this task may not be closed, or ``""`` if it may."""
    if str(task.get("status")) == DONE:
        return ""  # already closed; re-asserting is a no-op, not a new claim
    if str(task.get("status")) != IN_PROGRESS:
        return (
            f"Task {task['id']!r} is {task.get('status', PENDING)}, not in_progress, so "
            f"there is nothing yet that could be evidence of the work. Call "
            f"task_update(id={task['id']!r}, status='in_progress') first, do the work, "
            f"and only then close it."
        )
    evidence = _evidence_since(state, task)
    if evidence:
        return ""
    # The tools seen since the task started, if any, are named in the refusal so
    # the model can tell "nothing happened" from "only reads happened".
    since = int(task.get("since") or 0)
    seen = [str(entry.get("tool", "?")) for entry in state.get("evidence", [])
            if int(entry.get("seq", 0)) > since]
    detail = ", ".join(seen[-3:]) if seen else "none"
    return (
        f"Task {task['id']!r} cannot be marked done: no successful call to a real tool "
        f"has been observed since it entered in_progress. Saying it is finished is not "
        f"the same as having finished it -- this check is made from the agent's own "
        f"event stream, not from the request. Run the work with a tool such as "
        f"run_bash, run_pytest, write_file or edit_file, and the transition will be "
        f"allowed once one has succeeded. (Reading files does not count. Real tool "
        f"calls observed for this task so far: {detail}.) Nothing was changed."
    )


def _open_task_ids_locked(state: dict[str, Any]) -> list[str]:
    return [task_id for task_id, task in state["tasks"].items()
            if str(task.get("status")) == IN_PROGRESS]


def _record_evidence_locked(state: dict[str, Any], tool: str) -> None:
    """Stamp one observed success against every open task."""
    state["seq"] = int(state.get("seq", 0)) + 1
    state.setdefault("evidence", []).append(
        {"seq": state["seq"], "tool": tool, "at": _now()})
    del state["evidence"][:-MAX_EVIDENCE]
    for task_id in _open_task_ids_locked(state):
        task = state["tasks"][task_id]
        task["evidence"] = len(_evidence_since(state, task))


# --- the tools --------------------------------------------------------------


def task_update(id, status, note=""):
    """Create or update a task.

    ``status`` is one of pending, in_progress, done, blocked. A task reaching
    ``done`` needs observed evidence: a successful real tool call since it
    entered in_progress. Without one this refuses and says what is missing.
    """
    task_id = str(id or "").strip()
    if not task_id:
        return {"success": False, "error": "A task needs an id."}
    if len(task_id) > 120:
        return {"success": False, "error": "A task id may be at most 120 characters."}
    status = str(status or "").strip()
    if status not in STATUSES:
        return {"success": False,
                "error": f"Unknown status {status!r}. Use one of: {', '.join(STATUSES)}."}
    note = str(note or "").strip()

    global _has_open_task
    with _lock:
        state = _load_locked()
        task = state["tasks"].get(task_id)
        creating = task is None
        if creating:
            task = {"id": task_id, "status": PENDING, "note": "", "notes": [],
                    "evidence": 0, "since": None, "created": _now(), "history": []}
            state["tasks"][task_id] = task
        previous = str(task["status"])
        refusal = _transition_refusal(task, status, note) or (
            _missing_evidence(state, task) if status == DONE else "")
        if refusal:
            if creating:
                state["tasks"].pop(task_id, None)
            return {"success": False, "error": refusal}
        if status == IN_PROGRESS and previous != IN_PROGRESS:
            # Entering progress is what opens the evidence window. Re-asserting
            # in_progress must not reopen it, or a model could reset the clock
            # between doing the work and claiming it.
            task["since"] = int(state.get("seq", 0))
            task["evidence"] = 0
        if note:
            task["notes"] = (task.get("notes") or [])[-MAX_NOTES + 1:] + \
                [{"at": _now(), "text": note}]
            task["note"] = note
        task["status"] = status
        task["updated"] = _now()
        if previous != status:
            task["history"] = (task.get("history") or [])[-MAX_HISTORY + 1:] + \
                [{"at": _now(), "from": previous, "to": status, "note": note}]
        _has_open_task = any(
            str(other.get("status")) == IN_PROGRESS for other in state["tasks"].values())
        _write_locked(state)

    evidence = _evidence_since(state, task)
    if creating:
        return {"success": True,
                "output": f"Created task {task_id!r} as {status}."}
    return {"success": True,
            "output": f"Task {task_id!r}: {previous} -> {status}"
                       + (f" ({len(evidence)} observed tool call(s) since it started)."
                          if status == DONE else ".")}


def task_list(status=""):
    """List the tasks, optionally only those in one status."""
    wanted = str(status or "").strip()
    if wanted and wanted not in STATUSES:
        return {"success": False,
                "error": f"Unknown status {wanted!r}. Use one of: {', '.join(STATUSES)}, "
                         f"or an empty string for all."}
    with _lock:
        state = _load_locked()
        rows = []
        for task_id in sorted(state["tasks"]):
            task = state["tasks"][task_id]
            if wanted and str(task.get("status")) != wanted:
                continue
            rows.append(_render(state, task))
        warning = _load_error
    if not rows:
        return {"success": True,
                "output": f"No tasks{' with status ' + wanted if wanted else ''}."
                          + (f" ({warning})" if warning else "")}
    header = f"{len(rows)} task(s)" + (f" with status {wanted}" if wanted else "") + ":"
    tail = f"\n({warning})" if warning else ""
    return {"success": True, "output": header + "\n" + "\n".join(rows) + tail}


def _render(state: dict[str, Any], task: dict[str, Any]) -> str:
    evidence = len(_evidence_since(state, task))
    note = task.get("note") or ""
    detail = f"ev {evidence}" if str(task.get("status")) == IN_PROGRESS else \
        (f"ev {task.get('evidence', 0)}" if str(task.get("status")) == DONE else "")
    parts = [f"  {str(task.get('id', '')):<28} {str(task.get('status', '')):<12}"]
    if detail:
        parts.append(f"{detail:<7}")
    return "".join(parts) + (f"  {note}" if note else "")


def task_clear():
    """Reset the list. Trivially resettable by design: one call, no arguments."""
    global _has_open_task
    with _lock:
        state = _blank_state()
        _has_open_task = False
        _write_locked(state)
    return {"success": True, "output": "Task list cleared."}


# --- the hooks --------------------------------------------------------------


def on_agent_event(event):
    """Record successful real-tool calls as evidence. Never raises.

    This is the difference between evidence and a claim: the event stream is
    what the agent did, and it arrives whether or not the model narrates it. A
    call to ``task_list`` is not evidence, and neither is a call that failed.
    """
    global _has_open_task
    if not _has_open_task:
        return  # nothing open: no lock, no disk, no work. This runs every step.
    if getattr(event, "event_type", "") != "tool_result":
        return
    payload = getattr(event, "payload", None)
    if not isinstance(payload, dict) or not payload.get("success"):
        return
    tool = str(payload.get("name", ""))
    if not tool or tool in OBSERVER_TOOLS:
        return
    with _lock:
        state = _load_locked()
        # Re-checked under the lock: the fast path above can be stale by the
        # time this thread gets here, and a task may have been closed in between.
        open_tasks = _open_task_ids_locked(state)
        if not open_tasks:
            _has_open_task = False
            return
        _record_evidence_locked(state, tool)
        _has_open_task = True
        _write_locked(state)


def guard_pre_tool(tool_name, arguments):
    """Deny a ``done`` that no observed tool call supports.

    The handler refuses the same transition; having it here as well is what
    makes the rule un-routable. A check that only the thing it is checking
    enforces is a suggestion, and this one runs before the risk classifier and
    before the safety profile, so no profile can wave it through.
    """
    if tool_name != "task_update":
        return ALLOW
    if str((arguments or {}).get("status", "")).strip() != DONE:
        return ALLOW
    task_id = str((arguments or {}).get("id", "")).strip()
    if not task_id:
        return ALLOW  # the handler reports a missing id, with a better message
    with _lock:
        state = _load_locked()
        task = state["tasks"].get(task_id)
        if task is None:
            return HookVerdict(
                action="deny", plugin="tasklist",
                reason=(f"Task {task_id!r} does not exist, so it cannot be marked done. "
                        f"Create it with task_update(id={task_id!r}, "
                        f"status='in_progress'), do the work, then close it."))
        reason = _missing_evidence(state, task)
    if reason:
        return HookVerdict(action="deny", plugin="tasklist", reason=reason)
    return ALLOW
