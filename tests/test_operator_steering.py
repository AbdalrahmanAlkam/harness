"""Mid-turn operator steering.

A user who types while the agent is working wants their instruction to reach
the model inside the running turn, not after it ends. That is only safe if the
transcript has exactly one writer, and the tests below hold that line:

- a producer may only ever put text in the inbox, never in ``self.messages``;
- the injection happens at a step boundary, never inside a tool batch, so a
  note can never land between an assistant ``tool_calls`` message and its
  ``tool`` result and orphan the pair.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from adaptive_harness.agent.agent import (
    AgentEvent,
    DeveloperAgent,
    OperatorInbox,
    OperatorMessage,
)
from adaptive_harness.agent.compaction import OPERATOR_PREFIX
from adaptive_harness.llm.client import LLMResponse, ToolCall
from adaptive_harness.tools.base import ToolResult


class ScriptedClient:
    """Replays one step per entry and records what the model was actually sent."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.requests: list[list[dict]] = []
        self.default_model = "mock/model"
        self.provider = "openrouter"

    def complete(self, messages, **kwargs):
        self.requests.append([dict(message) for message in messages])
        return self.steps[min(len(self.requests) - 1, len(self.steps) - 1)]


def _text(content: str) -> LLMResponse:
    return LLMResponse(model="mock/model", content=content)


def _tool(name: str, call_id: str, **arguments) -> LLMResponse:
    return LLMResponse(model="mock/model", content="",
                       tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def _agent(tmp_path: Path, steps) -> DeveloperAgent:
    return DeveloperAgent(workspace_root=tmp_path, llm_client=ScriptedClient(steps),
                          max_steps=6, step_policy="unbounded")


def _drain(agent: DeveloperAgent) -> list[AgentEvent]:
    return list(agent.run_stream("do the thing"))


# --- the inbox itself ------------------------------------------------------


def test_the_inbox_holds_messages_in_arrival_order():
    inbox = OperatorInbox()
    assert inbox.depth == 0
    assert inbox.queue(OperatorMessage("first")) == 1
    assert inbox.queue(OperatorMessage("second")) == 2
    assert [item.text for item in inbox.drain()] == ["first", "second"]
    assert inbox.depth == 0, "draining must empty the inbox"


def test_an_unknown_operator_message_kind_is_rejected():
    with pytest.raises(ValueError):
        OperatorMessage("x", kind="shout")


def test_the_inbox_is_thread_safe_under_contention():
    """Eight producers, one consumer. Nothing may be lost or duplicated."""
    inbox = OperatorInbox()
    per_thread = 200

    def produce(index: int) -> None:
        for i in range(per_thread):
            inbox.queue(OperatorMessage(f"t{index}-{i}"))

    threads = [threading.Thread(target=produce, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    drained = inbox.drain()
    assert len(drained) == 8 * per_thread
    assert len({item.text for item in drained}) == 8 * per_thread


# --- a steer reaches the model inside the running turn ---------------------


def test_a_steer_is_injected_before_the_next_model_request(tmp_path: Path):
    agent = _agent(tmp_path, [_tool("list_directory", "c1", path="."),
                              _text("All done.")])
    events: list[AgentEvent] = []
    for event in agent.run_stream("do the thing"):
        events.append(event)
        if event.event_type == "tool_result":
            # Typed while the first step is still resolving.
            agent.queue_operator_message("use ripgrep instead")

    delivered = [event for event in events
                 if event.event_type == "operator_message" and not event.payload.get("late")]
    assert delivered, "the steer was never injected"
    assert "use ripgrep" in delivered[0].payload["raw"]
    # It must be in the second request, not the first.
    second_request = agent.llm_client.requests[1]
    assert any("use ripgrep" in str(message.get("content"))
               for message in second_request), "the model never saw the steer"


def test_a_steer_lands_after_the_whole_tool_batch(tmp_path: Path):
    """The note must not land between an assistant tool_calls message and its
    tool result, or the provider rejects the whole request."""
    agent = _agent(tmp_path, [_tool("list_directory", "c1", path="."),
                              _tool("list_directory", "c2", path="."),
                              _text("All done.")])
    events = []
    for event in agent.run_stream("do the thing"):
        events.append(event)
        if event.event_type == "tool_result" and event.payload.get("name") == "list_directory":
            agent.queue_operator_message("stop listing, read the file")

    def _pairs_are_paired(messages) -> bool:
        for index, message in enumerate(messages):
            for call in message.get("tool_calls") or []:
                following = messages[index + 1:index + 2]
                if not following or following[0].get("tool_call_id") != call["id"]:
                    return False
        return True

    assert _pairs_are_paired(agent.messages), "a tool_call was orphaned"


def test_the_steer_is_a_user_message_not_a_system_message(tmp_path: Path):
    """A system message in this transcript means the harness is speaking. The
    TUI renders the two differently, and user text must not borrow the
    authority of a harness intervention."""
    agent = _agent(tmp_path, [_tool("list_directory", "c1", path="."), _text("done")])
    for event in agent.run_stream("do the thing"):
        if event.event_type == "tool_result":
            agent.queue_operator_message("actually, use pytest")
    injected = [m for m in agent.messages
                if str(m.get("content", "")).startswith(OPERATOR_PREFIX)]
    assert injected, "the steer was not appended"
    assert injected[0]["role"] == "user"
    assert "actually, use pytest" in injected[0]["content"]


# --- an interrupt redirects rather than notes -------------------------------


def test_an_interrupt_stops_the_run_and_becomes_the_next_task(tmp_path: Path):
    agent = _agent(tmp_path, [_tool("list_directory", "c1", path="."), _text("done")])
    # Queue the interrupt from a thread that fires while the first step resolves.
    def inject() -> None:
        time.sleep(0.01)
        agent.queue_operator_message("start over and write tests", kind="interrupt")

    thread = threading.Thread(target=inject)
    thread.start()
    events = []
    for event in agent.run_stream_with_followup("do the thing"):
        events.append(event)
    thread.join()

    interrupts = [event for event in events
                  if event.event_type == "operator_message"
                  and event.payload.get("kind") == "interrupt"]
    assert interrupts, "the interrupt was never reported"
    assert any(event.payload.get("state") == "interrupting" for event in interrupts)


def test_an_interrupt_is_not_written_into_the_transcript_twice(tmp_path: Path):
    """It is the next turn's task, not a note to this one. Appending it as both
    would put the same instruction in the history twice."""
    agent = _agent(tmp_path, [_tool("list_directory", "c1", path="."), _text("done")])

    def inject() -> None:
        time.sleep(0.01)
        agent.queue_operator_message("switch to writing tests", kind="interrupt")

    thread = threading.Thread(target=inject)
    thread.start()
    list(agent.run_stream_with_followup("do the thing"))
    thread.join()

    occurrences = [m for m in agent.messages
                   if "switch to writing tests" in str(m.get("content", ""))]
    # A redirect belongs in the transcript exactly once: as the next turn's task.
    # It must never also appear as an injected steer note, which is the shape
    # that would repeat the instruction to the model.
    injected_notes = [m for m in occurrences
                      if str(m.get("content", "")).startswith(OPERATOR_PREFIX)]
    assert not injected_notes, "the redirect was injected as a steer as well as used as a task"
    assert occurrences, "the redirect never reached the transcript"


def test_a_redirect_loop_is_bounded(tmp_path: Path):
    """A user who keeps interrupting must not be able to loop forever.

    The client interrupts on every step, so the loop is driven by the redirect
    itself rather than by a racing thread.
    """
    class InterruptingClient(ScriptedClient):
        def complete(self, messages, **kwargs):
            agent.queue_operator_message("again", kind="interrupt")
            return _text("done")

    agent = _agent(tmp_path, [_text("done")])
    agent.llm_client = InterruptingClient([_text("done")])

    events = list(agent.run_stream_with_followup("do the thing", followup_limit=3))

    notices = [event for event in events if event.event_type == "llm_notice"]
    assert notices, "the bounded loop did not report that it stopped"
    assert "3 consecutive operator redirects" in notices[0].payload["message"]
    assert len(agent.llm_client.requests) <= 6, "the redirect loop was not bounded"


# --- edge cases ------------------------------------------------------------


def test_a_late_steer_is_kept_but_reported_as_late(tmp_path: Path):
    """Typed during the final step: the run is over, so it must not be claimed
    to have steered anything -- but it is not discarded either."""
    agent = _agent(tmp_path, [_text("done")])

    class LateClient(ScriptedClient):
        def complete(self, messages, **kwargs):
            agent.queue_operator_message("one more thing")
            return super().complete(messages, **kwargs)

    agent.llm_client = LateClient([_text("done")])
    events = _drain(agent)
    late = [event for event in events
            if event.event_type == "operator_message" and event.payload.get("late")]
    assert late, "a note typed during the final step was silently dropped"
    assert any("one more thing" in str(m.get("content")) for m in agent.messages)


def test_two_messages_typed_quickly_are_both_delivered_in_order(tmp_path: Path):
    agent = _agent(tmp_path, [_tool("list_directory", "c1", path="."), _text("done")])
    events = []
    for event in agent.run_stream("do the thing"):
        events.append(event)
        if event.event_type == "tool_result":
            agent.queue_operator_message("first note")
            agent.queue_operator_message("second note")

    delivered = [event.payload["raw"] for event in events
                 if event.event_type == "operator_message" and not event.payload.get("late")]
    assert delivered == ["first note", "second note"]


def test_an_injected_steer_survives_compaction(tmp_path: Path):
    """A correction must not be summarised away a few steps after it is typed."""
    from adaptive_harness.agent.context_window import prepare_context

    history = [{"role": "system", "content": "system prompt"}]
    history += [{"role": "user", "content": f"request {i}"} for i in range(6)]
    history += [{"role": "assistant", "content": f"answer {i}"} for i in range(6)]
    history += [{"role": "user", "content": f"{OPERATOR_PREFIX} · sent at step 7\n"
                                                 "do not use the cached build"}]
    history += [{"role": "assistant", "content": f"later {i}"} for i in range(10)]

    prepared, _info = prepare_context(history, "mock/model", None, limit=200)
    assert any("do not use the cached build" in str(m.get("content")) for m in prepared), (
        "the operator steer was compacted away")


def test_queueing_does_not_touch_the_transcript_from_another_thread(tmp_path: Path):
    """The producer's only writable surface is the inbox."""
    agent = _agent(tmp_path, [_tool("list_directory", "c1", path="."), _text("done")])
    before = list(agent.messages)

    def produce() -> None:
        for i in range(50):
            agent.queue_operator_message(f"note {i}")

    thread = threading.Thread(target=produce)
    thread.start()
    thread.join()

    assert agent.messages == before, "a foreign thread wrote to the transcript"
    assert agent.operator_inbox.depth == 50


def test_a_concurrent_run_and_steer_keeps_the_transcript_valid(tmp_path: Path):
    """The real case: one thread driving the run, another steering it."""
    agent = _agent(tmp_path, [_tool("list_directory", "c1", path="."), _text("done")])

    def produce() -> None:
        for i in range(20):
            agent.queue_operator_message(f"note-{i}")
            time.sleep(0.001)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    list(agent.run_stream_with_followup("do the thing", followup_limit=2))
    thread.join()

    # Every declared tool call has a result, in order.
    for index, message in enumerate(agent.messages):
        for call in message.get("tool_calls") or []:
            following = agent.messages[index + 1:index + 2]
            assert following and following[0].get("tool_call_id") == call["id"], (
                "a tool call was orphaned by steering")

    # Each note appears at most once. The separator matters: "note-1" is a
    # substring of "note-10" through "note-19", and counting on the bare
    # number would blame the code for the test's own substring collision.
    for i in range(20):
        needle = f"note-{i}\n"
        count = sum(1 for m in agent.messages if needle in str(m.get("content", "")))
        assert count <= 1, f"note-{i} was injected {count} times"
