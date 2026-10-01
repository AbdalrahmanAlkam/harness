"""A run you cannot stop is not shippable.

Cancellation is cooperative: a tool call already in flight is allowed to finish,
because pre-empting arbitrary synchronous code is not something the agent loop
can promise. Everything after it is skipped, so the run stops spending tokens
the user asked it not to spend, and it says plainly that it stopped and why.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.llm.client import LLMResponse, ToolCall


class LoopingClient:
    """Always asks for another tool call, so only cancellation ends the run."""

    default_model = "mock/model"
    provider = "openrouter"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(model=self.default_model, content="",
                           tool_calls=[ToolCall(id=f"c{self.calls}", name="list_directory",
                                                arguments={"path": "."})])


def _agent(tmp_path: Path) -> DeveloperAgent:
    return DeveloperAgent(workspace_root=tmp_path, llm_client=LoopingClient(),
                          max_steps=50, step_policy="unbounded")


def _response(events) -> dict:
    responses = [event for event in events if event.event_type == "response"]
    assert responses, "the run never produced a response event"
    return responses[-1].payload


def test_a_run_stops_when_cancellation_is_requested(tmp_path: Path):
    agent = _agent(tmp_path)
    agent.request_cancel("user asked")

    response = _response(list(agent.run_stream("keep going")))

    assert response["stop_reason"] == "cancelled"
    assert response["success"] is False


def test_cancelling_does_not_claim_success(tmp_path: Path):
    """A cancelled run must never be reported as a completed task, because the
    TUI takes `success` from this flag and the CLI exits non-zero on it."""
    agent = _agent(tmp_path)
    agent.request_cancel()
    assert _response(list(agent.run_stream("keep going")))["success"] is False


def test_the_cancellation_is_attributed(tmp_path: Path):
    """A silent stop looks like a crash. The reason the operator gave has to
    reach them."""
    agent = _agent(tmp_path)
    agent.request_cancel("the command was the wrong one")
    content = _response(list(agent.run_stream("keep going")))["content"]
    assert "the command was the wrong one" in content


def test_cancellation_can_be_requested_from_another_thread(tmp_path: Path):
    """The Esc key arrives on the event loop while the run is on a worker."""
    agent = _agent(tmp_path)
    assert not agent.cancel_requested

    def cancel() -> None:
        agent.request_cancel("operator pressed Esc")

    thread = threading.Thread(target=cancel)
    thread.start()
    thread.join()

    assert agent.cancel_requested
    assert _response(list(agent.run_stream("keep going")))["stop_reason"] == "cancelled"


def test_an_uncancelled_run_is_not_affected(tmp_path: Path):
    """The default must stay a full run, not a run that stops one step early."""
    client = LoopingClient()
    agent = DeveloperAgent(workspace_root=tmp_path, llm_client=client,
                           max_steps=4, step_policy="unbounded")
    assert not agent.cancel_requested
    response = _response(list(agent.run_stream("keep going")))
    assert response["stop_reason"] == "step_limit"
    assert client.calls >= 3, "the run stopped far earlier than its budget"


def test_the_transcript_stays_valid_when_cancelled(tmp_path: Path):
    """Cancelling at a step boundary must not leave a declared tool call without
    a result, or the next request in the session is rejected."""
    agent = _agent(tmp_path)

    def cancel_midway() -> None:
        agent.request_cancel("enough")

    thread = threading.Thread(target=cancel_midway)
    thread.start()
    list(agent.run_stream("keep going"))
    thread.join()

    for index, message in enumerate(agent.messages):
        for call in message.get("tool_calls") or []:
            following = agent.messages[index + 1:index + 2]
            assert following and following[0].get("tool_call_id") == call["id"], (
                "cancellation orphaned a tool call")
