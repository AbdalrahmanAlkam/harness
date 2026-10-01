"""Regressions for the subagent defects a critique found.

Each test here corresponds to a defect that shipped and was fixed. They are
grouped by the rule they defend, because the rules are what future changes have
to keep:

- the swarm must actually run when a registry is attached;
- a subagent's declared permissions must be real restrictions, not comments;
- a cancelled run must not come back as a failure;
- what the parent is charged for must match what it is told;
- ids must keep the format the module documents.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from adaptive_harness.agent.subagents import (
    STATUS_STOPPED,
    SubagentRegistry,
    SubagentReporter,
)
from adaptive_harness.llm.client import LLMResponse
from adaptive_harness.tools.delegation import DelegateSubagentTool


class _Event:
    def __init__(self, event_type, payload):
        self.event_type = event_type
        self.payload = payload


def _agent(root: Path, name: str, text: str) -> None:
    directory = root / ".harness" / "agents"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(text, encoding="utf-8")


class _Quiet:
    default_model = "mock"
    provider = "openrouter"

    def complete(self, messages, **kwargs):
        return LLMResponse(model="mock", content="done")


# --- the swarm must run at all ---------------------------------------------


def test_a_swarm_run_with_a_registry_attached_actually_runs(tmp_path: Path):
    """`assignment.directive` does not exist -- the field is `task` -- so every
    swarm run with a registry attached died on an AttributeError before doing any
    work. Nothing caught it because no test ran a coordinator with a registry."""
    from adaptive_harness.agent.subagents import SubagentRegistry as Registry
    from adaptive_harness.agent.swarm import DeveloperAgentWorker, SwarmCoordinator

    registry = Registry()
    report = SwarmCoordinator(DeveloperAgentWorker(
        llm_client_factory=_Quiet, max_steps=2, registry=registry)).run("t", tmp_path)

    assert report is not None
    # The architect plans and needs no file changes, so it should succeed.
    architect = next(r for r in report.results if r.role.value == "architect")
    assert architect.success, architect.error
    # And the registry saw the agents, so /tasks would have shown them.
    assert registry.all(), "no agent was registered"


# --- declared permissions must actually restrict ---------------------------


def test_an_agent_that_disallows_every_tool_is_refused_not_given_the_defaults(tmp_path: Path):
    """`tools if declared else None` gave an agent whose every tool was
    filtered out the *full default set* -- so a definition that disallowed both
    writing and reading was handed both."""
    _agent(tmp_path, "picky",
           "---\nname: picky\ndescription: d\n"
           "tools: write_file, read_file\n"
           "disallowedTools: write_file, read_file\n---\nB\n")
    result = DelegateSubagentTool(tmp_path, llm_client_factory=_Quiet).execute(
        role="picky", task="x")
    assert not result.success
    assert "removed by its own permissions" in result.error


def test_a_definition_naming_no_tools_gets_the_defaults(tmp_path: Path):
    """The opposite case, which must keep working: no `tools:` line means the
    built-in set."""
    _agent(tmp_path, "plain", "---\nname: plain\ndescription: d\n---\nB\n")
    result = DelegateSubagentTool(tmp_path, llm_client_factory=_Quiet).execute(
        role="plain", task="say something")
    assert result.success, result.error


def test_a_file_shadowing_a_builtin_role_keeps_its_own_permissions(tmp_path: Path):
    """A `coder.md` declared `permissionMode: plan` used to be handed the
    built-in coder role -- implement phase, `require_file_changes` -- while
    holding no write tool, so it could never succeed."""
    _agent(tmp_path, "coder",
           "---\nname: coder\ndescription: Plans only.\n"
           "tools: read_file\npermissionMode: plan\n---\nB\n")
    registry = SubagentRegistry()
    result = DelegateSubagentTool(tmp_path, llm_client_factory=_Quiet,
                                  registry=registry).execute(
        role="coder", task="plan the change")
    record = registry.get(result.metadata.get("agent_id")) if result.metadata.get("agent_id") else None
    assert record is not None
    # A plan agent produces no file, so it must not be reported as one that
    # failed to produce a file.
    assert "missing_file_changes" not in (record.error or ""), (
        "a plan-mode agent was forced into the implement phase")


# --- a cancelled run must stay cancelled -----------------------------------


def test_a_late_completion_does_not_turn_a_stop_into_a_failure():
    """A background thread that finished after the user pressed Esc used to
    overwrite `stopped` with `failed`, so a cancelled run was reported as a
    fault minutes later."""
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    registry.stop_all("the run was cancelled")
    assert record.status == STATUS_STOPPED

    registry.complete(record.id, result="late", error="did not complete")
    assert registry.get(record.id).status == STATUS_STOPPED, (
        "a cancelled agent came back as a failure")


def test_a_normal_completion_still_completes():
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    registry.complete(record.id, result="done")
    assert registry.get(record.id).status == "completed"


# --- the parent is charged for what it is told ------------------------------


def test_a_failed_call_is_counted_once(tmp_path: Path):
    """`tool_call` fires before execution, so appending the failure counted it
    twice -- in the number a user reads to learn the cost."""
    registry = SubagentRegistry()
    record = registry.spawn(parent_id="main", role="coder", description="x")
    reporter = SubagentReporter(registry)
    reporter.on_event(record.id, _Event("tool_call", {"name": "Edit", "arguments": {"path": "a"}}))
    reporter.on_event(record.id, _Event("tool_result", {"name": "Edit", "success": False}))
    assert len(registry.get(record.id).tools) == 1
    assert registry.totals()["tools"] == 1


def test_what_the_parent_receives_is_bounded(tmp_path: Path):
    """A subagent's transcript is replayed into the parent on every later
    request, so the payload is capped -- not just the display."""
    from adaptive_harness.tools.delegation import RESULT_TO_PARENT_CHARS

    _agent(tmp_path, "verbose",
           "---\nname: verbose\ndescription: d\ntools: read_file, write_file\n---\nB\n")
    result = DelegateSubagentTool(tmp_path, llm_client_factory=_Quiet).execute(
        role="verbose", task="x")
    assert len(result.output) < RESULT_TO_PARENT_CHARS * 3, (
        f"the parent was handed {len(result.output)} chars")


# --- ids keep the documented format -----------------------------------------


def test_ids_stay_unique_and_correctly_shaped():
    """Clock-derived ids collided inside a 65-second window, and the collision
    path produced ids wider than the documented `agent_XXXX`."""
    registry = SubagentRegistry()
    ids = {registry.spawn(parent_id="m", role="c", description=str(i)).id
           for i in range(20_000)}
    assert len(ids) == 20_000, "ids collided"
    assert all(i.startswith("agent_") and len(i) == 10 for i in ids), (
        "an id did not match the documented shape")


# --- unacted keys must warn --------------------------------------------------


def test_a_key_this_version_ignores_is_reported_not_silently_dropped(tmp_path: Path):
    """`hooks:` and `mcpServers:` are Claude Code keys and are the most likely
    to arrive from elsewhere. Dropping them with no note is the failure mode."""
    from adaptive_harness.agents import AgentRegistry, load_definition

    _agent(tmp_path, "foreign",
           "---\nname: foreign\ndescription: d\nhooks: x\nmcpServers: y\n---\nB\n")
    registry = AgentRegistry(tmp_path)
    registry.discover()
    definition = registry.get("foreign")
    assert "hooks" in definition.extra and "mcpServers" in definition.extra
    assert "not acted on" in registry.describe(), (
        "an ignored key produced no note at all")


def test_an_invalid_effort_is_refused_at_load_naming_the_file(tmp_path: Path):
    from adaptive_harness.agents import AgentDefinitionError, load_definition

    _agent(tmp_path, "typo",
           "---\nname: typo\ndescription: d\neffort: banana\n---\nB\n")
    with pytest.raises(AgentDefinitionError) as excinfo:
        load_definition(tmp_path / ".harness" / "agents" / "typo.md")
    assert "effort" in str(excinfo.value) and "typo" in str(excinfo.value)


# --- the collect-a-background-agent tool must exist --------------------------


def test_background_result_is_a_registered_tool_not_just_a_method(tmp_path: Path):
    """The tool schema told the model to call `background_result`; it existed
    only as a Python method, so an id it handed back could never be resolved."""
    from adaptive_harness.agent.agent import DeveloperAgent

    agent = DeveloperAgent(workspace_root=tmp_path)
    agent.enable_swarm(True)
    assert "background_result" in agent.tools
    assert "delegate_subagent" in agent.tools
    agent.enable_swarm(False)
    assert "background_result" not in agent.tools


# --- a model tier is resolved, not sent literally ---------------------------


def test_a_tier_name_resolves_to_that_tier_model():
    """The scaffold ships `model: fast`. Sending that string to a provider as a
    model name fails on every request, and passes against the mock."""
    from adaptive_harness.agent.swarm import DeveloperAgentWorker
    from adaptive_harness.llm.client import LLMClient

    client = DeveloperAgentWorker(
        llm_client_factory=lambda: LLMClient(api_key="x"),
        explicit_model="fast")._client_for_assignment()
    assert client.default_model != "fast", "a tier was sent as a literal model name"
    assert "/" in client.default_model

    literal = DeveloperAgentWorker(
        llm_client_factory=lambda: LLMClient(api_key="x"),
        explicit_model="vendor/explicit-model")._client_for_assignment()
    assert literal.default_model == "vendor/explicit-model"
