"""Subagent identity, lifecycle, and monitoring.

A subagent that is one anonymous burst of tool calls cannot be watched, stopped,
or accounted for -- and with several running at once their output interleaves
into a wall of text where you cannot tell whose work you are reading.

These tests pin the properties that fix that:

- every subagent has a short, readable id, and that id is on every event it
  produces, so the tree can be rebuilt from the stream alone;
- the lifecycle is `agent_spawned` then `agent_completed`, carrying a parent;
- a run is answered without watching: `/agents` reports status, tool counts,
  tokens, and elapsed time for every agent;
- **the parent gets a bounded ledger, not the transcript.** A subagent's chatter
  replayed into every later request is one of the largest avoidable costs in a
  tool loop, so what comes back up is capped.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from adaptive_harness.agent.subagents import (
    BULLET,
    ID_PREFIX,
    LEDGER_MAX_RESULT,
    LEDGER_MAX_TOOLS,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    SubagentRecord,
    SubagentRegistry,
    SubagentReporter,
    complete_event,
    spawn_event,
)


# --- identity ---------------------------------------------------------------


def test_every_subagent_gets_a_short_readable_id():
    registry = SubagentRegistry()
    first = registry.spawn(parent_id="main", role="Explore", description="map the code")
    second = registry.spawn(parent_id="main", role="coder", description="write the fix")

    for record in (first, second):
        assert record.id.startswith(ID_PREFIX)
        # Short enough to type when a user wants to refer to one agent.
        assert len(record.id) <= len(ID_PREFIX) + 6
    assert first.id != second.id, "two agents were given the same id"


def test_ids_are_stable_for_the_life_of_the_agent():
    """An id that changes is worse than no id: it cannot be referred to."""
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="Explore", description="x")
    registry.record_tool(record.id, "Read", {"path": "a.py"})
    registry.complete(record.id, result="done")
    assert registry.get(record.id).id == record.id
    assert registry.get(record.id) is not None


def test_a_parent_child_relationship_is_recorded():
    registry = SubagentRegistry()
    parent = registry.spawn(parent_id="main", role="architect", description="plan")
    child = registry.spawn(parent_id=parent.id, role="coder", description="build",
                           depth=1)
    assert [record.id for record in registry.children_of(parent.id)] == [child.id]
    assert registry.children_of("main") == [parent]
    assert child.depth == 1


# --- lifecycle --------------------------------------------------------------


def test_spawn_then_complete_is_the_lifecycle():
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="write it")
    assert record.status == STATUS_RUNNING
    assert record.running

    registry.complete(record.id, result="wrote the file", stop_reason="completed")
    finished = registry.get(record.id)
    assert finished.status == STATUS_COMPLETED
    assert not finished.running
    assert finished.result == "wrote the file"
    assert finished.ended_at >= finished.started_at


def test_a_failure_is_distinguished_from_a_completion():
    """A run that broke is not a run that finished, and the interface has to be
    able to tell them apart."""
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    registry.complete(record.id, error="refused by policy")
    assert registry.get(record.id).status == STATUS_FAILED
    assert registry.get(record.id).error == "refused by policy"


def test_completing_an_unknown_agent_is_not_an_error():
    """A late event from a cancelled worker must not raise."""
    assert SubagentRegistry().complete("agent_ffff", result="late") is None


def test_stopping_a_run_marks_every_running_agent_stopped():
    """Otherwise the monitoring view keeps spinning for agents that are gone."""
    registry = SubagentRegistry()
    running = registry.spawn(parent_id="main", role="a", description="x")
    registry.spawn(parent_id="main", role="b", description="y")
    finished = registry.spawn(parent_id="main", role="c", description="z")
    registry.complete(finished.id, result="done")

    stopped = registry.stop_all("cancelled")
    assert len(stopped) == 2
    assert not registry.running()
    assert registry.get(running.id).status == STATUS_STOPPED


# --- the events -------------------------------------------------------------


def test_the_spawn_event_carries_everything_needed_to_rebuild_the_tree():
    record = SubagentRecord(id="agent_1a2b", parent_id="main", role="coder",
                            description="write the guard", model="sonnet", depth=1)
    payload = spawn_event(record)
    assert payload["agent_id"] == "agent_1a2b"
    assert payload["parent_id"] == "main"
    assert payload["role"] == "coder"
    assert payload["description"] == "write the guard"
    assert payload["depth"] == 1


def test_the_payload_uses_claude_codes_field_naming():
    """`agent_spawned` and `agent_completed` are this harness's own event
    names -- there is no public contract for an internal event stream -- but the
    *fields* follow the documented SubagentStart/SubagentStop naming exactly:
    snake_case `agent_id`, and an `agent_` prefix on the id.

    Getting this wrong is not cosmetic: a plugin written against the documented
    shape would read none of it.
    """
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    for payload in (spawn_event(record), complete_event(record)):
        assert "agent_id" in payload
        assert "agentId" not in payload, "camelCase leaks in; Claude Code's is snake_case"
        assert payload["agent_id"].startswith("agent_")


def test_the_complete_event_carries_the_cost():
    """What a user needs at the end: did it work, what did it do, what did it
    cost."""
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    registry.record_tool(record.id, "Write", {"file_path": "a.py"})
    registry.record_usage(record.id, input_tokens=900, output_tokens=120, model="m")
    registry.complete(record.id, result="done", stop_reason="completed")

    payload = complete_event(registry.get(record.id))
    assert payload["status"] == STATUS_COMPLETED
    assert payload["elapsed_ms"] >= 0
    assert payload["tools"] == [["Write", "a.py"]]
    assert payload["input_tokens"] == 900 and payload["output_tokens"] == 120


# --- the reporter -----------------------------------------------------------


class _Event:
    def __init__(self, event_type, payload):
        self.event_type = event_type
        self.payload = payload


def test_the_reporter_tracks_tool_calls_and_usage():
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    reporter = SubagentReporter(registry)

    reporter.on_event(record.id, _Event("tool_call", {"name": "Read", "arguments": {"path": "a.py"}}))
    reporter.on_event(record.id, _Event("tool_result", {"name": "Read", "success": True}))
    reporter.on_event(record.id, _Event("response", {"usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": "m"}))

    tracked = registry.get(record.id)
    assert tracked.tools == [("Read", "a.py")]
    assert tracked.input_tokens == 10 and tracked.output_tokens == 5
    assert tracked.model == "m"


def test_a_failed_tool_call_is_marked_not_appended():
    """A failure marks the entry the call already made.

    `tool_call` fires for every declared call *before* execution, so appending
    the failure as a second entry counted it twice -- and the count is what a
    user reads to learn what a subagent cost.
    """
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    reporter = SubagentReporter(registry)
    reporter.on_event(record.id,
                      _Event("tool_call", {"name": "Edit", "arguments": {"path": "a.py"}}))
    reporter.on_event(record.id, _Event("tool_result", {"name": "Edit", "success": False}))

    tools = registry.get(record.id).tools
    assert len(tools) == 1, f"one failed call was recorded {len(tools)} times"
    assert tools[0][0] == "Edit"
    assert "failed" in tools[0][1]


def test_a_failure_with_no_matching_call_does_not_invent_one():
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    SubagentReporter(registry).on_event(
        record.id, _Event("tool_result", {"name": "Ghost", "success": False}))
    assert registry.get(record.id).tools == [], (
        "a result for a call that was never declared invented an entry")


def test_a_successful_result_does_not_double_count_the_tool():
    """`tool_call` already recorded it; recording the result too would show two
    of every call in the ledger."""
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    reporter = SubagentReporter(registry)
    reporter.on_event(record.id, _Event("tool_call", {"name": "Read", "arguments": {}}))
    reporter.on_event(record.id, _Event("tool_result", {"name": "Read", "success": True}))
    assert len(registry.get(record.id).tools) == 1


def test_an_event_for_an_unknown_agent_is_ignored():
    registry = SubagentRegistry()
    SubagentReporter(registry).on_event("agent_ffff", _Event("tool_call", {"name": "x"}))
    assert registry.all() == []


# --- the efficiency property ------------------------------------------------


def test_the_ledger_is_bounded():
    """The parent replays this on every later request, so it must not grow with
    the subagent's chatter."""
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    for index in range(200):
        registry.record_tool(record.id, "Bash", {"command": f"step {index}"})
    registry.record_usage(record.id, input_tokens=1, output_tokens=1)
    registry.complete(record.id, result="R" * 50_000)

    ledger = registry.get(record.id).ledger()
    assert len(ledger) < 3000, "the ledger is unbounded; it will be replayed forever"
    assert f"and {200 - LEDGER_MAX_TOOLS} more" in ledger
    assert len(ledger) < len("R" * 50_000)


def test_the_ledger_names_the_agent_and_its_outcome():
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="write the guard")
    registry.record_tool(record.id, "Write", {"file_path": "src/guard.py"})
    registry.complete(record.id, result="Added the guard.")

    ledger = registry.get(record.id).ledger()
    assert record.id in ledger
    assert "Write src/guard.py" in ledger
    assert "Added the guard." in ledger


def test_a_failed_agent_says_why_in_its_ledger():
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="security", description="check it")
    registry.complete(record.id, error="could not reach the network")
    assert "could not reach the network" in registry.get(record.id).ledger()


def test_a_long_result_is_truncated_in_the_ledger():
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    registry.complete(record.id, result="Z" * 100_000)
    assert len(registry.get(record.id).ledger()) < LEDGER_MAX_RESULT + 500


# --- the monitoring view ----------------------------------------------------


def test_the_view_answers_what_is_running_and_what_it_cost():
    """A run should not have to be watched to be understood."""
    registry = SubagentRegistry()
    running = registry.spawn(parent_id="main", role="Explore", description="map the code")
    done = registry.spawn(parent_id="main", role="coder", description="write the fix")
    registry.record_usage(done.id, input_tokens=1200, output_tokens=340)
    registry.complete(done.id, result="done")

    view = registry.describe()
    assert running.id in view and done.id in view
    assert "1,540" in view or "1,540 tok" in view
    assert "1 running" in view
    assert "1 finished" in view


def test_the_view_says_so_when_there_were_none():
    """An empty view must still explain itself, not just print nothing."""
    described = SubagentRegistry().describe()
    assert "No subagents" in described
    assert "parallel" in described.lower() or "delegate" in described.lower(), (
        "the empty view should say how to get one, not just report none")


def test_a_failure_shows_its_reason_in_the_view():
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    registry.complete(record.id, error="the workspace was not writable")
    assert "the workspace was not writable" in registry.describe()


def test_nested_agents_are_indented():
    registry = SubagentRegistry()
    parent = registry.spawn(parent_id="main", role="architect", description="plan")
    child = registry.spawn(parent_id=parent.id, role="coder", description="build", depth=1)
    lines = registry.describe().splitlines()
    child_line = next(line for line in lines if child.id in line)
    parent_line = next(line for line in lines if parent.id in line)
    assert parent_line.startswith("  ") and not parent_line.startswith("   ")
    assert child_line.startswith("    "), (
        "a nested agent is not indented under its parent")
    assert len(child_line) - len(child_line.lstrip()) > len(parent_line) - len(parent_line.lstrip())


def test_the_totals_add_up():
    registry = SubagentRegistry()
    a = registry.spawn(parent_id="main", role="x", description="1")
    b = registry.spawn(parent_id="main", role="y", description="2")
    registry.record_tool(a.id, "Read", {"path": "x.py"})
    registry.record_usage(a.id, input_tokens=10, output_tokens=5)
    registry.complete(a.id, result="ok")
    registry.complete(b.id, error="broke")

    totals = registry.totals()
    assert totals["agents"] == 2 and totals["finished"] == 2
    assert totals["failed"] == 1
    assert totals["tools"] == 1
    assert totals["input_tokens"] == 10 and totals["output_tokens"] == 5


def test_the_bullet_is_a_single_glyph():
    """It sits in front of every tool call; anything wider and the column
    stops aligning."""
    assert len(BULLET) == 1


# --- concurrency ------------------------------------------------------------


def test_the_registry_is_safe_under_concurrent_writers():
    """Swarm workers run on a thread pool while the interface reads this."""
    registry = SubagentRegistry()
    errors: list[Exception] = []

    def spawn_many(index: int) -> None:
        try:
            for _ in range(25):
                record = registry.spawn(parent_id="main", role="w", description=f"{index}")
                registry.record_tool(record.id, "Read", {"path": "x.py"})
                registry.record_usage(record.id, input_tokens=1, output_tokens=1)
                registry.complete(record.id, result="ok")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=spawn_many, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    totals = registry.totals()
    assert totals["agents"] == 150
    assert totals["input_tokens"] == 150
    assert len({record.id for record in registry.all()}) == 150, "two agents shared an id"


def test_reading_while_writing_does_not_raise():
    registry = SubagentRegistry()
    stop = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for index in range(300):
                record = registry.spawn(parent_id="main", role="w", description=str(index))
                registry.record_tool(record.id, "Read", {"path": "x.py"})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            while not stop.is_set():
                registry.describe()
                registry.totals()
                registry.running()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors


def test_reset_clears_everything():
    registry = SubagentRegistry()
    registry.spawn(parent_id="main", role="x", description="1")
    registry.reset()
    assert registry.all() == [] and registry.totals()["agents"] == 0


# --- a capability that exists but nothing calls is not a capability ----------


def test_cancelling_a_run_marks_its_subagents_stopped():
    """The bug this pins: the monitoring view said agents were still running for
    a run that had already been cancelled, which is exactly the state a user
    checks `/agents` in order to avoid trusting."""
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    assert record.running

    stopped = registry.stop_all("the run was cancelled")

    assert stopped, "cancelling reported nothing, so a running agent stays 'running' forever"
    assert not registry.running()
    assert registry.get(record.id).status == STATUS_STOPPED
    assert "cancelled" in registry.get(record.id).error


def test_the_tui_cancels_subagents_too():
    """Wiring, checked against the source: a method that exists but is never
    called is the failure mode this whole module is about."""
    import inspect

    from adaptive_harness.tui.app import AdaptiveHarnessApp

    source = inspect.getsource(AdaptiveHarnessApp.action_cancel_task)
    assert "stop_all" in source, (
        "Esc cancels the agent but not its subagents, so /agents keeps spinning")
