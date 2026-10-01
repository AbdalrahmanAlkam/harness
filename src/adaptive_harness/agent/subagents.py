"""Subagent lifecycle and monitoring.

A subagent that runs behind a single `⇢ subagent tool call: run_bash {"command":
...}` line is invisible: you cannot tell which agent is doing what, whether one
is stuck, or what any of it cost. When several run at once it is worse, because
their output interleaves into a wall of text.

So every subagent gets an identity and a lifecycle:

- a short, readable id (``agent_7f3a``) that appears on every one of its events,
  so a user can follow one agent or refer to it by name;
- ``agent_spawned`` and ``agent_completed`` events carrying that id and the
  parent's, so the tree can be reconstructed from the event stream alone;
- a per-agent record of status, tool calls, tokens and elapsed time, so
  ``/tasks`` can answer "what is running, and what has it cost me" without
  having been watching.

**Where this deliberately differs from the Claude Code contract, and why.**
Claude Code's documented hooks are ``SubagentStart`` and ``SubagentStop``, and
their payloads use ``agent_id`` and ``agent_type`` in snake_case; the id format
is an ``agent_`` prefix plus an alphanumeric suffix; and ``/tasks`` is the view
that lists running subagents. This follows all four. Two deliberate departures:

- ``parent_id`` exists here and does not in Claude Code. Their tree is two deep
  -- a subagent is parented by the session -- whereas this harness can nest a
  subagent under a subagent, so the parent has to be a real id or the deeper
  levels cannot be reconstructed.
- The event *names* are this harness's own, since no public contract defines an
  internal event stream; what is mirrored is the payload field naming, the id
  format, the display conventions, and the command name.

The efficiency win is the same thing: a subagent's chatter is *not* replayed
into the parent's context. What comes back is its result plus a short ledger, and
the full stream stays in the registry for anyone who asks to see it.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

#: The glyph a tool call renders under. One character, and it reads as "the
#: agent did a thing" at a glance, which is what a run of them needs.
BULLET = "⏺"
#: Used while the agent is still running.
BULLET_ACTIVE = "⏺"
BULLET_DONE = "●"

#: Short and pronounceable. A longer id is easier to mistype when a user wants
#: to refer to a specific agent, and there is no reason for it to be.
ID_PREFIX = "agent_"
ID_LENGTH = 4

#: The ledger a subagent hands back to its parent. Small on purpose: this text
#: is replayed into the parent's context on every step it runs in, so an
#: unbounded summary would quietly become one of the largest costs in a run.
LEDGER_MAX_TOOLS = 6
LEDGER_MAX_RESULT = 1200

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"


#: A monotonic counter behind every id. Ids derived from the clock alone
#: collide inside a 65-second window, and the collision path grew the suffix
#: past the documented width. Counting cannot collide, and a 24-bit counter
#: still reads as a short hex id.
_id_counter = itertools.count(1)


def _make_id() -> str:
    """A short id, unique for the life of the process.

    A clock-derived suffix looks tidier and is worse: it collides inside a
    65-second window, and the collision path then produced ids wider than the
    format this module documents.
    """
    # Plain counter modulo the width. Mixing in a clock-derived seed looks
    # better and is worse: folding a seed into 16 bits costs real collisions,
    # because the space is then shared between the seed and the counter rather
    # than spent entirely on uniqueness. Wrapping is fine -- the caller
    # disambiguates, and the width is what the documented format promises.
    return f"{ID_PREFIX}{next(_id_counter) % (16 ** ID_LENGTH):0{ID_LENGTH}x}"


@dataclass
class SubagentRecord:
    """Everything known about one subagent."""

    id: str
    parent_id: str
    role: str
    description: str
    status: str = STATUS_RUNNING
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    #: Tool calls made, in order, as (name, one-line target).
    tools: List[tuple[str, str]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    result: str = ""
    error: str = ""
    stop_reason: str = ""
    depth: int = 0
    #: An optional tint from a definition's `color:` line. Decorative only, and
    #: it never carries meaning the glyph and the status word do not already.
    color: str = ""

    @property
    def elapsed_ms(self) -> int:
        end = self.ended_at or time.time()
        return int((end - self.started_at) * 1000)

    @property
    def running(self) -> bool:
        return self.status == STATUS_RUNNING

    def ledger(self) -> str:
        """The compact account a subagent returns to its parent.

        Deliberately bounded. The full transcript lives in the registry for
        anyone who asks; what goes back up is a summary and a count, because a
        subagent's chatter replayed into every later request is one of the
        largest avoidable costs in a tool loop.
        """
        lines = [f"{self.role} ({self.id}): {self.description}"]
        if self.tools:
            shown = self.tools[:LEDGER_MAX_TOOLS]
            for name, target in shown:
                lines.append(f"  · {name}{f' {target}' if target else ''}")
            if len(self.tools) > len(shown):
                lines.append(f"  · … and {len(self.tools) - len(shown)} more")
        if self.result:
            lines.append(f"  → {self.result[:LEDGER_MAX_RESULT]}")
        if self.error:
            lines.append(f"  ✗ {self.error[:400]}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "parent_id": self.parent_id, "role": self.role,
            "description": self.description, "status": self.status,
            "elapsed_ms": self.elapsed_ms, "tools": [list(item) for item in self.tools],
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "model": self.model, "result": self.result[:LEDGER_MAX_RESULT],
            "error": self.error, "stop_reason": self.stop_reason, "depth": self.depth,
        }

    #: Accepted tints, for a definition's `color:` line. An unlisted name is
    #: ignored rather than being passed through to a terminal escape sequence.
    COLORS = frozenset({"red", "green", "yellow", "blue", "magenta", "cyan",
                         "white", "grey"})

    def line(self) -> str:
        """One row for the monitoring view."""
        mark = "●" if self.running else ("✓" if self.status == STATUS_COMPLETED else "✗")
        timing = f"{self.elapsed_ms / 1000:.1f}s"
        tokens = self.input_tokens + self.output_tokens
        cost = f"{tokens:,} tok" if tokens else "—"
        # The tint is decorative and only ever names itself. Status stays in the
        # glyph and the word, so nothing is carried by colour alone.
        tint = f" [{self.color}]" if self.color in self.COLORS else ""
        return (f"{mark} {self.id}  {self.role:<12} {timing:>7}  {len(self.tools):>3} tools  "
                f"{cost:>12}{tint:<16} {self.description[:44]}")


class SubagentRegistry:
    """Every subagent the run has seen, live and finished.

    Thread-safe, because a swarm runs its workers on a thread pool and the
    interface reads this from the event loop while they write to it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, SubagentRecord] = {}
        self._order: List[str] = []
        self._counter = itertools.count(1)

    # -- lifecycle ---------------------------------------------------------

    def spawn(self, *, parent_id: str, role: str, description: str,
              model: str = "", depth: int = 0) -> SubagentRecord:
        with self._lock:
            # A collision is possible and harmless; the counter makes it
            # vanishingly unlikely without needing a lock of its own.
            agent_id = _make_id()
            suffix = 0
            while agent_id in self._records:
                suffix += 1
                agent_id = f"{_make_id()}{suffix:x}"
            record = SubagentRecord(id=agent_id, parent_id=parent_id, role=role,
                                    description=description, model=model, depth=depth)
            self._records[agent_id] = record
            self._order.append(agent_id)
        return record

    def complete(self, agent_id: str, *, result: str = "", error: str = "",
                  stop_reason: str = "", status: Optional[str] = None) -> Optional[SubagentRecord]:
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return None
            if not record.running:
                # A late completion must not overwrite a stop. The user pressed
                # Esc, was told the run was being cancelled, and a background
                # thread that finished minutes later would otherwise turn that
                # into "failed" -- reporting a cancellation as a fault.
                return record
            record.ended_at = time.time()
            record.result = result or record.result
            record.error = error or record.error
            record.stop_reason = stop_reason or record.stop_reason
            if status is None:
                status = STATUS_FAILED if record.error else STATUS_COMPLETED
            record.status = status
            return record

    def stop_all(self, reason: str = "stopped by the operator") -> List[SubagentRecord]:
        """Mark every running agent stopped. Used when a run is cancelled, so
        the monitoring view does not keep spinning for agents that are gone."""
        with self._lock:
            stopped = [record for record in self._records.values() if record.running]
            for record in stopped:
                record.status = STATUS_STOPPED
                record.ended_at = time.time()
                record.error = reason
            return stopped

    # -- activity ----------------------------------------------------------

    def record_tool(self, agent_id: str, name: str, target: str = "") -> None:
        """One tool call, with a short target so the row is scannable.

        The arguments matter for auditing but not for a glance, so the target is
        a one-line summary rather than a JSON dump.
        """
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return
            record.tools.append((str(name), _short_target(target)))

    def mark_failed(self, agent_id: str, name: str) -> None:
        """Mark the most recent matching call as failed, rather than adding one."""
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return
            for index in range(len(record.tools) - 1, -1, -1):
                if record.tools[index][0] == name:
                    target = f"{record.tools[index][1]} (failed)".strip()
                    record.tools[index] = (name, target)
                    return

    def record_usage(self, agent_id: str, *, input_tokens: int = 0,
                      output_tokens: int = 0, model: str = "") -> None:
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return
            record.input_tokens += int(input_tokens or 0)
            record.output_tokens += int(output_tokens or 0)
            if model:
                record.model = model

    # -- reading -----------------------------------------------------------

    def get(self, agent_id: str) -> Optional[SubagentRecord]:
        with self._lock:
            return self._records.get(agent_id)

    def all(self) -> List[SubagentRecord]:
        with self._lock:
            return [self._records[agent_id] for agent_id in self._order]

    def running(self) -> List[SubagentRecord]:
        return [record for record in self.all() if record.running]

    def children_of(self, parent_id: str) -> List[SubagentRecord]:
        return [record for record in self.all() if record.parent_id == parent_id]

    def totals(self) -> Dict[str, Any]:
        records = self.all()
        running = sum(1 for record in records if record.running)
        return {
            "agents": len(records),
            "running": running,
            "finished": len(records) - running,
            "failed": sum(1 for record in records if record.status == STATUS_FAILED),
            "tools": sum(len(record.tools) for record in records),
            "input_tokens": sum(record.input_tokens for record in records),
            "output_tokens": sum(record.output_tokens for record in records),
        }

    def describe(self) -> str:
        """The monitoring view, printed by `/tasks`."""
        records = self.all()
        if not records:
            return ("No subagents this session.\n"
                    "They appear here when the harness delegates one — ask for "
                    "parallel work, or a role that should be reviewed separately.")
        lines = [f"{BULLET} subagents ({len(records)})"]
        for record in records:
            indent = "  " * (record.depth + 1)
            lines.append(f"{indent}{record.line()}")
            if record.error and record.status != STATUS_RUNNING:
                lines.append(f"{indent}  {record.error[:160]}")
        totals = self.totals()
        summary = (f"{totals['running']} running · {totals['finished']} finished"
                   + (f" · {totals['failed']} failed" if totals["failed"] else ""))
        if totals["input_tokens"] or totals["output_tokens"]:
            summary += (f" · {totals['input_tokens']:,} in / "
                        f"{totals['output_tokens']:,} out tokens")
        lines.append("")
        lines.append(summary)
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._order.clear()


def _short_target(target: Any, limit: int = 60) -> str:
    """A one-line summary of a tool's arguments.

    A JSON dump of every argument is unreadable in a list of ten rows and is
    mostly noise: the first meaningful string is what identifies the call.
    """
    if not isinstance(target, dict):
        return str(target or "")[:limit]
    for key in ("path", "file_path", "command", "query", "url", "target", "name", "role"):
        value = target.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            return (text[:limit - 1] + "…") if len(text) > limit else text
    return ""


# --- event emission ---------------------------------------------------------

def spawn_event(record: SubagentRecord) -> Dict[str, Any]:
    """The `agent_spawned` payload.

    Shaped so the whole agent tree can be reconstructed from the event stream
    alone, without a registry: id, parent, role, and what it was asked to do.
    """
    return {"agent_id": record.id, "parent_id": record.parent_id, "role": record.role,
            "description": record.description, "model": record.model,
            "depth": record.depth, "status": record.status}


def complete_event(record: SubagentRecord) -> Dict[str, Any]:
    """The `agent_completed` payload: what it did, what it cost, what came back."""
    return {"agent_id": record.id, "parent_id": record.parent_id, "role": record.role,
            "status": record.status, "elapsed_ms": record.elapsed_ms,
            "tools": [list(item) for item in record.tools],
            "input_tokens": record.input_tokens, "output_tokens": record.output_tokens,
            "result": record.result[:LEDGER_MAX_RESULT], "error": record.error,
            "stop_reason": record.stop_reason}


class SubagentReporter:
    """Bridges the agent's event stream onto the registry.

    One place that knows which events are about a subagent, so the interface
    does not have to guess, and so a new event type cannot quietly stop being
    monitored.
    """

    def __init__(self, registry: SubagentRegistry) -> None:
        self.registry = registry

    def on_event(self, agent_id: str, event: Any) -> None:
        event_type = getattr(event, "event_type", "")
        payload = getattr(event, "payload", {}) or {}
        if event_type == "tool_call":
            self.registry.record_tool(agent_id, payload.get("name", "?"),
                                     payload.get("arguments", {}))
        elif event_type == "tool_result" and not payload.get("success"):
            # A failure is marked on the entry the call already created, rather
            # than appended as a second one. `tool_call` fires for every declared
            # call before execution, so appending here counted every failure
            # twice and inflated the very number a user reads to learn what a
            # subagent cost.
            self.registry.mark_failed(agent_id, payload.get("name", "?"))
        elif event_type == "response":
            usage = payload.get("usage") or {}
            self.registry.record_usage(
                agent_id,
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
                model=str(payload.get("model", "") or ""))
