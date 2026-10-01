"""Phase 2.1 — plan mode.

Plan mode is a mode, not a personality. The agent may read, search and delegate;
every mutating tool is intercepted and the run ends with a plan for the operator
to approve, edit, or reject.

The properties worth holding:
- a mutating call in plan mode never reaches the filesystem;
- the refusal is *visible* to the model, so it learns the boundary rather than
  guessing at it, and the transcript stays well-formed;
- a plan run never reports success. Reporting "completed" would claim the work
  was done — precisely what plan mode refused to do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.classifiers.domain_classifier import DomainMode, parse_domain_mode
from adaptive_harness.llm.client import LLMResponse, ToolCall


class Scripted:
    """Replays one tool call, then a final answer."""

    default_model = "mock/model"
    provider = "openrouter"

    def __init__(self, first_call, final: str = "## Plan\n\n1. Do the thing."):
        self.first_call = first_call
        self.final = final
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1 and self.first_call is not None:
            return LLMResponse(model=self.default_model, content="",
                               tool_calls=[self.first_call])
        return LLMResponse(model=self.default_model, content=self.final)


def _agent(tmp_path: Path, client) -> DeveloperAgent:
    return DeveloperAgent(workspace_root=tmp_path, llm_client=client,
                          forced_mode="plan", max_steps=4, step_policy="unbounded")


def _response(events) -> dict:
    return [event for event in events if event.event_type == "response"][-1].payload


# --- the mode exists and is a real mode -------------------------------------


def test_plan_parses_as_a_mode():
    assert parse_domain_mode("plan") is DomainMode.PLAN


def test_an_unknown_mode_is_still_refused():
    with pytest.raises(ValueError):
        parse_domain_mode("planning")


def test_plan_has_its_own_guidance():
    from adaptive_harness.classifiers.domain_classifier import DOMAIN_GUIDANCE

    guidance = DOMAIN_GUIDANCE[DomainMode.PLAN]
    assert "PLAN MODE" in guidance
    assert "do not modify" in guidance.lower()


# --- mutating tools are refused ---------------------------------------------


def test_a_write_is_refused_and_never_reaches_the_filesystem(tmp_path: Path):
    client = Scripted(ToolCall(id="c1", name="write_file",
                               arguments={"path": "a.py", "content": "x"}))
    events = list(_agent(tmp_path, client).run_stream("plan the change"))

    assert not (tmp_path / "a.py").exists(), "plan mode wrote a file"
    assert any(event.event_type == "plan_mode_blocked" for event in events)


def test_an_edit_is_refused(tmp_path: Path):
    (tmp_path / "a.py").write_text("original", encoding="utf-8")
    client = Scripted(ToolCall(id="c1", name="edit_file",
                               arguments={"path": "a.py", "target_text": "original",
                                          "replacement_text": "changed"}))
    list(_agent(tmp_path, client).run_stream("plan the change"))
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "original"


def test_a_mutating_shell_command_is_refused(tmp_path: Path):
    client = Scripted(ToolCall(id="c1", name="run_bash",
                               arguments={"command": "echo pwned > created.txt"}))
    list(_agent(tmp_path, client).run_stream("plan the change"))
    assert not (tmp_path / "created.txt").exists()


def test_a_read_only_shell_command_is_allowed(tmp_path: Path):
    """Plan mode is read-and-investigate; refusing `git status` would make it
    useless for the investigation it exists to support."""
    client = Scripted(ToolCall(id="c1", name="run_bash",
                               arguments={"command": "ls -la"}))
    events = list(_agent(tmp_path, client).run_stream("plan the change"))
    assert not any(event.event_type == "plan_mode_blocked" for event in events)


def test_reading_is_allowed(tmp_path: Path):
    (tmp_path / "a.py").write_text("content", encoding="utf-8")
    client = Scripted(ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}))
    events = list(_agent(tmp_path, client).run_stream("plan the change"))
    assert not any(event.event_type == "plan_mode_blocked" for event in events)


# --- the refusal is visible and the transcript stays valid ------------------


def test_the_refusal_tells_the_model_why(tmp_path: Path):
    client = Scripted(ToolCall(id="c1", name="write_file",
                               arguments={"path": "a.py", "content": "x"}))
    agent = _agent(tmp_path, client)
    events = list(agent.run_stream("plan the change"))
    blocked = [event for event in events if event.event_type == "plan_mode_blocked"]
    assert blocked
    assert "read-only" in blocked[0].payload["reason"]
    # And the model was actually told, in the transcript it will re-read.
    assert any("read-only" in str(message.get("content", ""))
               for message in agent.messages)


def test_a_refused_call_still_gets_a_result(tmp_path: Path):
    """Otherwise the declared tool_call is left unanswered and the provider
    rejects every later request in the session."""
    client = Scripted(ToolCall(id="c1", name="write_file",
                               arguments={"path": "a.py", "content": "x"}))
    agent = _agent(tmp_path, client)
    list(agent.run_stream("plan the change"))
    for index, message in enumerate(agent.messages):
        for call in message.get("tool_calls") or []:
            following = agent.messages[index + 1:index + 2]
            assert following and following[0].get("tool_call_id") == call["id"]


# --- the deliverable is a plan, not a completed task ------------------------


def test_a_plan_run_reports_plan_ready_not_completed(tmp_path: Path):
    client = Scripted(None)
    response = _response(list(_agent(tmp_path, client).run_stream("plan the change")))
    assert response["stop_reason"] == "plan_ready"
    assert response["success"] is False, (
        "a plan mode run reported success; it refused to do the work")


def test_the_plan_is_emitted_as_its_own_event(tmp_path: Path):
    client = Scripted(None)
    events = list(_agent(tmp_path, client).run_stream("plan the change"))
    ready = [event for event in events if event.event_type == "plan_ready"]
    assert ready, "the plan was not surfaced as an event"
    assert "## Plan" in ready[0].payload["content"]


def test_a_plan_run_with_no_steps_is_still_a_plan(tmp_path: Path):
    client = Scripted(None, final="I need more information before planning.")
    response = _response(list(_agent(tmp_path, client).run_stream("plan the change")))
    assert response["stop_reason"] == "plan_ready"
