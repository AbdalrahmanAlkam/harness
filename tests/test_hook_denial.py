"""Phase 0.2 and 2.3 — anything that can deny a tool call.

Two mechanisms can stop a call, and the property they share is that neither can
be routed around:

- a `PRE_TOOL` plugin hook, which runs before the risk classifier and before
  the safety profile;
- a rule in `.harness/rules.json`, evaluated inside the safety gate.

A guard a permissive profile can bypass is not a guard, so the tests here pin
that both hold under `--safety-profile turbo`, and that a refusal still leaves
a well-formed transcript — a denied call declared and then silently skipped
produces a history the next request will be rejected for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.agent.rules import (
    Decision,
    Rule,
    RuleSet,
    default_rules,
    normalize,
)
from adaptive_harness.llm.client import LLMResponse, ToolCall


def _write_rules(workspace: Path, rules) -> None:
    (workspace / ".harness").mkdir(parents=True, exist_ok=True)
    (workspace / ".harness" / "rules.json").write_text(json.dumps({"rules": rules}),
                                                        encoding="utf-8")


# --- rules: parsing ---------------------------------------------------------


def test_rules_load_in_order(tmp_path: Path):
    _write_rules(tmp_path, [
        {"pattern": "pytest", "decision": "allow"},
        {"pattern": "rm -rf", "decision": "deny", "reason": "no."},
    ])
    rules = RuleSet.load(tmp_path / ".harness" / "rules.json")
    assert len(rules) == 2
    assert rules.rules[0].index == 0 and rules.rules[1].index == 1


def test_a_missing_rules_file_is_simply_no_rules(tmp_path: Path):
    assert len(RuleSet.load(tmp_path / "absent.json")) == 0


def test_a_broken_rules_file_is_reported_not_silently_ignored(tmp_path: Path):
    """Silently becoming 'no rules' would drop a user's policy with no symptom."""
    (tmp_path / "rules.json").write_text("{not json", encoding="utf-8")
    rules = RuleSet.load(tmp_path / "rules.json")
    assert len(rules) == 0
    assert rules.error, "a broken rules file must say so"


def test_a_deny_rule_must_explain_itself():
    """A silent refusal is indistinguishable from a bug to the user."""
    with pytest.raises(ValueError):
        Rule(pattern="rm", decision=Decision.DENY)
    assert Rule(pattern="rm", decision=Decision.DENY, reason="policy").reason == "policy"


def test_a_malformed_pattern_is_refused_at_load_time(tmp_path: Path):
    _write_rules(tmp_path, [{"pattern": "[unclosed", "decision": "allow"}])
    assert RuleSet.load(tmp_path / ".harness" / "rules.json").error


# --- rules: matching --------------------------------------------------------


def test_first_match_wins(tmp_path: Path):
    rules = RuleSet(rules=[
        Rule(pattern="pytest", decision=Decision.ALLOW, index=0),
        Rule(pattern="pytest", decision=Decision.DENY, reason="no", index=1),
    ])
    assert rules.decide("run_bash", {"command": "pytest -q"}) is Decision.ALLOW


def test_deny_wins_a_tie_on_the_same_position():
    """Two rules at the same index, one denying: the stricter must win, because
    a user who wrote both meant the stricter one."""
    rules = RuleSet(rules=[
        Rule(pattern="rm", decision=Decision.ALLOW, index=0),
        Rule(pattern="rm", decision=Decision.DENY, reason="policy", index=0),
    ])
    assert rules.decide("run_bash", {"command": "rm x"}) is Decision.DENY


def test_no_matching_rule_is_no_opinion_not_a_question():
    """The distinction matters: with no rules file, an absent rule must not
    interrupt every tool call in the run."""
    assert RuleSet().decide("run_bash", {"command": "ls"}) is Decision.NONE


def test_a_rule_can_be_scoped_to_named_tools():
    rules = RuleSet(rules=[Rule(pattern="secret", decision=Decision.DENY,
                                reason="no", tools=("read_file",), index=0)])
    assert rules.decide("read_file", {"path": "secrets.txt"}) is Decision.DENY
    assert rules.decide("run_bash", {"command": "cat secrets.txt"}) is Decision.NONE


def test_a_rule_cannot_be_defeated_by_extra_whitespace():
    """If a model can slip past a rule with a second space, it is not a rule."""
    assert normalize("run_bash", {"command": "rm  -rf  /tmp"}) == "run_bash command=rm -rf /tmp"
    rules = RuleSet(rules=[Rule(pattern=r"rm\s+-[a-z]*r", decision=Decision.DENY,
                                reason="no", index=0)])
    assert rules.decide("run_bash", {"command": "rm  -rf  /tmp"}) is Decision.DENY


def test_a_rule_can_match_a_file_path(tmp_path: Path):
    _write_rules(tmp_path, [{"pattern": r"\.env", "decision": "deny",
                             "reason": "may hold secrets."}])
    rules = RuleSet.load(tmp_path / ".harness" / "rules.json")
    assert rules.decide("read_file", {"path": ".env"}) is Decision.DENY
    assert rules.decide("read_file", {"path": "README.md"}) is Decision.NONE


# --- rules: the agent applies them ------------------------------------------


def test_a_rule_denies_a_call(tmp_path: Path):
    _write_rules(tmp_path, [{"pattern": r"rm\s+-[a-z]*r", "decision": "deny",
                             "reason": "Recursive deletion is refused."}])
    agent = DeveloperAgent(workspace_root=tmp_path)
    decision, reason = agent.rule_gate("run_bash", {"command": "rm -rf /"})
    assert decision is Decision.DENY
    assert "Recursive deletion" in reason


def test_a_rule_asks_regardless_of_safety_profile(tmp_path: Path):
    """The point of rules over profiles: an ask the profile cannot skip."""
    _write_rules(tmp_path, [{"pattern": "git push", "decision": "ask",
                             "reason": "Pushing is not reversible."}])
    agent = DeveloperAgent(workspace_root=tmp_path, safety_profile="turbo")
    decision, reason = agent.rule_gate("run_bash", {"command": "git push"})
    assert decision is Decision.ASK
    assert "not reversible" in reason


def test_no_rules_file_means_no_interference(tmp_path: Path):
    agent = DeveloperAgent(workspace_root=tmp_path)
    assert agent.rule_gate("run_bash", {"command": "ls"})[0] is Decision.NONE


def test_a_broken_rules_file_surfaces_rather_than_disappearing(tmp_path: Path):
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness" / "rules.json").write_text("{broken", encoding="utf-8")
    agent = DeveloperAgent(workspace_root=tmp_path)
    assert agent.rule_gate("run_bash", {"command": "ls"})[0] is Decision.NONE
    assert agent.rules_error, "the failure must be recorded, not swallowed"


def test_the_default_rules_are_valid_and_sensible():
    for rule in default_rules():
        rule.matches("run_bash", "run_bash rm -rf /tmp")  # must not raise
    rules = RuleSet(rules=default_rules())
    assert rules.decide("run_bash", {"command": "rm -rf build"}) is Decision.DENY
    assert rules.decide("run_bash", {"command": "pytest -q"}) is Decision.ALLOW


# --- hooks that can deny ----------------------------------------------------


def _agent(tmp_path: Path, plugin_source: str, permissions=("tools", "hooks")) -> DeveloperAgent:
    directory = tmp_path / ".harness" / "plugins" / "guard"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.plugin.json").write_text(json.dumps({
        "name": "guard", "version": "1.0", "permissions": list(permissions),
        "tools": [{"name": "noop", "description": "d", "handler": "noop",
                   "risk": "read", "parameters": {"type": "object", "properties": {}}}],
        "hooks": [{"when": "pre_tool", "handler": "guard"}],
    }), encoding="utf-8")
    (directory / "plugin.py").write_text(plugin_source, encoding="utf-8")
    agent = DeveloperAgent(workspace_root=tmp_path,
                           allow_project_plugins=True)
    agent.plugins.discover()
    return agent


_DENY_ALL = """
from adaptive_harness.plugins.types import ALLOW, HookVerdict


def noop():
    return {"success": True, "output": "ok"}


def guard(tool_name, arguments):
    return HookVerdict("deny", reason=f"{tool_name} is blocked by policy")
"""


def test_a_hook_denies_a_tool_call_even_under_the_permissive_profile(tmp_path: Path):
    """A guard the most permissive profile can route around is not a guard."""
    agent = _agent(tmp_path, _DENY_ALL)
    arguments, reason, plugin = agent.plugins.consult_pre_tool("write_file", {})
    assert reason and "blocked by policy" in reason
    assert plugin == "guard", "a denial names who made it"


def test_a_denied_call_still_gets_a_result_in_the_transcript(tmp_path: Path):
    """Otherwise the declared tool_call is left unanswered and the provider
    rejects every later request in the session."""
    from adaptive_harness.llm.client import LLMResponse, ToolCall

    agent = _agent(tmp_path, _DENY_ALL)

    class Client:
        default_model = "mock/model"
        provider = "openrouter"

        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(model="mock/model", content="",
                                   tool_calls=[ToolCall(id="c1", name="write_file",
                                                        arguments={"path": "a.py",
                                                                   "content": "x"})])
            return LLMResponse(model="mock/model", content="I could not write the file.")

    agent.llm_client = Client()
    events = list(agent.run_stream("write a file"))
    blocked = [event for event in events if event.event_type == "hook_blocked"]
    assert blocked, "the denial was not surfaced as an event"

    for index, message in enumerate(agent.messages):
        for call in message.get("tool_calls") or []:
            following = agent.messages[index + 1:index + 2]
            assert following and following[0].get("tool_call_id") == call["id"], (
                "a denied call left a tool_call unanswered")


def test_a_hook_that_raises_blocks_rather_than_failing_open(tmp_path: Path):
    exploding = _DENY_ALL.replace(
        "def guard(tool_name, arguments):",
        "def guard(tool_name, arguments):\n    raise RuntimeError('hook exploded')")
    agent = _agent(tmp_path, exploding)
    _, reason, _ = agent.plugins.consult_pre_tool("write_file", {})
    assert "hook exploded" in reason
    assert "blocked rather than run unchecked" in reason
