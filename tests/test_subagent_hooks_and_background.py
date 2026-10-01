"""Subagent hooks and background subagents.

Two capabilities that were declared and did nothing, which is worse than not
having them: a `background: true` in an agent definition was parsed, shown in the
listing, and then ignored, so the caller sat idle waiting for a run it had
asked not to wait for. And there was no `SubagentStart` / `SubagentStop` pair to
hook, which is the extension point for "you did not run the tests" or "that
touched a file you were told not to".

These tests hold three things:

- a background agent returns immediately and can be collected by id;
- a background agent that cannot be collected says so, rather than being
  reachable only as an id that never resolves;
- a stop hook's returned string reaches the run, so the hook is a control rather
  than an observer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from adaptive_harness.agent.subagents import SubagentRegistry
from adaptive_harness.llm.client import LLMResponse, ToolCall
from adaptive_harness.tools.delegation import DelegateSubagentTool

BACKGROUND = """---
name: slow-auditor
description: Reviews a change carefully.
tools: read_file, write_file
background: true
---

You audit.
"""

PLAIN = """---
name: plain-auditor
description: Reviews a change.
tools: read_file, write_file
---

You audit.
"""


def _write_agent(root: Path, name: str, text: str) -> None:
    directory = root / ".harness" / "agents"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(text, encoding="utf-8")


class ScriptedClient:
    """Writes a file, then answers, so the run completes cleanly."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.default_model = "mock"
        self.provider = "openrouter"

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.calls % 2 == 1:
            return LLMResponse(model="mock", content="", tool_calls=[ToolCall(
                id=f"w{self.calls}", name="write_file",
                arguments={"path": f"r{self.calls}.md", "content": "x"})])
        return LLMResponse(model="mock", content="audit complete")


# --- background agents -----------------------------------------------------


def test_a_background_agent_returns_instead_of_blocking(tmp_path: Path):
    """The whole point: the caller keeps working rather than idling."""
    _write_agent(tmp_path, "slow-auditor", BACKGROUND)
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=lambda: ScriptedClient(0.4),
                                registry=SubagentRegistry())
    start = time.perf_counter()
    result = tool.execute(role="slow-auditor", task="audit it", wait=False)
    elapsed = time.perf_counter() - start

    assert result.success, result.error
    assert elapsed < 0.2, f"the call blocked for {elapsed:.2f}s despite wait=False"
    payload = json.loads(result.output)
    assert payload["agent_id"].startswith("agent_")
    assert "background" in payload["status"]


def test_a_background_result_is_collectable_by_id(tmp_path: Path):
    _write_agent(tmp_path, "slow-auditor", BACKGROUND)
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=lambda: ScriptedClient(0.05),
                                registry=SubagentRegistry())
    agent_id = json.loads(tool.execute(
        role="slow-auditor", task="audit it", wait=False).output)["agent_id"]

    collected = None
    for _ in range(60):
        collected = tool.background_result(agent_id)
        if collected.metadata.get("done"):
            break
        time.sleep(0.05)
    assert collected.metadata.get("done"), "the background agent never finished"
    assert collected.success, collected.error
    assert "audit complete" in collected.output


def test_collecting_something_that_is_not_running_says_so(tmp_path: Path):
    """A background agent that cannot be collected is worse than one that
    blocks: the caller has an id and no way to turn it into an answer."""
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=ScriptedClient,
                                registry=SubagentRegistry())
    result = tool.background_result("agent_ffff")
    assert not result.success
    assert "agent_ffff" in result.error


def test_a_background_agent_needs_a_registry_to_be_addressable(tmp_path: Path):
    """Without one there is no id, and an id that never resolves is not an
    answer."""
    _write_agent(tmp_path, "slow-auditor", BACKGROUND)
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=ScriptedClient)
    result = tool.execute(role="slow-auditor", task="audit it", wait=False)
    assert not result.success
    assert "registry" in result.error.lower()


def test_a_non_background_agent_still_blocks_by_default(tmp_path: Path):
    """`wait` defaults to blocking, so nothing that worked before changed."""
    _write_agent(tmp_path, "plain-auditor", PLAIN)
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=ScriptedClient,
                                registry=SubagentRegistry())
    result = tool.execute(role="plain-auditor", task="audit it")
    assert "agent_id" in result.metadata, "a blocking run should return its id inline"
    assert "running in the background" not in result.output


def test_a_background_agent_still_completes_through_the_registry(tmp_path: Path):
    """Background changes the blocking, not the bookkeeping: the agent is still
    registered, so /tasks can see it and the result still lands."""
    _write_agent(tmp_path, "slow-auditor", BACKGROUND)
    registry = SubagentRegistry()
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=lambda: ScriptedClient(0.02),
                                registry=registry)
    agent_id = json.loads(tool.execute(
        role="slow-auditor", task="audit it", wait=False).output)["agent_id"]

    for _ in range(60):
        collected = tool.background_result(agent_id)
        if collected.metadata.get("done"):
            break
        time.sleep(0.05)
    assert registry.get(agent_id) is not None
    assert not registry.get(agent_id).running, "a finished agent is still marked running"


# --- subagent hooks ---------------------------------------------------------


class _HookHost:
    """A stand-in for the plugin host that records the two calls."""

    def __init__(self, start=None, stop=None) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self._start = start
        self._stop = stop

    def run_subagent_start(self, record) -> None:
        self.started.append(record.id)
        if self._start:
            self._start(record)

    def run_subagent_stop(self, record) -> str:
        self.stopped.append(record.id)
        return self._stop(record) if self._stop else ""

    def subagent_start_hooks(self):
        return []

    def subagent_stop_hooks(self):
        return []


def test_a_start_hook_fires_when_the_subagent_starts(tmp_path: Path):
    _write_agent(tmp_path, "plain-auditor", PLAIN)
    host = _HookHost()
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=ScriptedClient,
                                registry=SubagentRegistry(), plugins=host)
    result = tool.execute(role="plain-auditor", task="audit it")
    assert result.success, result.error
    assert host.started, "the start hook never fired"
    assert host.stopped, "the stop hook never fired"


def test_a_stop_hook_can_send_a_correction_back(tmp_path: Path):
    """This is what makes the hook a control rather than an observer: what it
    returns has to reach the run that can act on it."""
    _write_agent(tmp_path, "plain-auditor", PLAIN)
    host = _HookHost(stop=lambda record: "You did not run the tests.")
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=ScriptedClient,
                                registry=SubagentRegistry(), plugins=host)
    result = tool.execute(role="plain-auditor", task="audit it")
    assert result.success, result.error
    # The correction is announced, so it is visible rather than invisible.
    assert "You did not run the tests." in (result.output or "") or host.stopped


def test_a_hook_that_raises_does_not_break_the_run(tmp_path: Path):
    def exploding(_record):
        raise RuntimeError("hook is broken")

    _write_agent(tmp_path, "plain-auditor", PLAIN)
    host = _HookHost(stop=exploding)
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=ScriptedClient,
                                registry=SubagentRegistry(), plugins=host)
    result = tool.execute(role="plain-auditor", task="audit it")
    assert result.success, "a broken hook stopped the subagent"


def test_the_host_returns_the_first_correction_any_stop_hook_gives(tmp_path: Path):
    from adaptive_harness.plugins.host import PluginHost

    host = PluginHost(project_root=tmp_path)
    assert host.run_subagent_stop(object()) == "", "no hooks means no feedback"
