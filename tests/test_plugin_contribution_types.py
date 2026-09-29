"""Phase 0.1 — the seven plugin contribution types.

The acceptance criterion is that a test plugin uses every one of the types and
the agent loop diff contains no new branches. The first half is this file; the
second is a source check at the bottom.

Every test builds its own fixture and points discovery at it, so nothing here
imports a plugin from the user's config or from a real project.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from adaptive_harness.plugins.host import PluginHost
from adaptive_harness.plugins.types import (
    ALLOW,
    ContextFragment,
    HookVerdict,
    PRIORITY_PROJECT,
    PRIORITY_REFERENCE,
    Trigger,
)

MODULE = '''
from adaptive_harness.plugins.types import (
    ALLOW, ContextFragment, HookVerdict, PRIORITY_PROJECT, Trigger)
from adaptive_harness.skills.registry import BaseSkill


def a_tool(path):
    from pathlib import Path
    target = Path(path)
    return {"success": target.is_file(), "output": "ok"}


def build_skill(source="plugin"):
    return BaseSkill(name="t_skill", title="T", category="T", trigger="t",
                     instructions="t", tools=("a_tool",),
                     invariants=("evidence_checked",), source=source)


def make_fragment(source="", provenance=""):
    # The host hands the declared source in; a factory that ignores it would
    # make two declared fragments collide on one name.
    return ContextFragment(source=source or "f:?", content="hello", tokens=2,
                           priority=PRIORITY_PROJECT,
                           trigger=Trigger.parse("always"),
                           provenance=provenance)


def deny_hook(tool_name, arguments):
    return HookVerdict("deny", reason="blocked for the test") if tool_name == "bad" else ALLOW


def post_hook(tool_name, arguments, output, success):
    return {"metadata": {"seen": tool_name}}


def final_hook(summary, requirement_set, evidence):
    return {"ok": True}
'''


def _host(tmp_path: Path, manifest: dict) -> PluginHost:
    (tmp_path / ".harness" / "plugins").mkdir(parents=True, exist_ok=True)
    directory = tmp_path / ".harness" / "plugins" / manifest["name"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "plugin.py").write_text(MODULE, encoding="utf-8")
    host = PluginHost(project_root=tmp_path, allow_project_plugins=True)
    host.discovery_roots = lambda: [tmp_path / ".harness" / "plugins"]
    return host


def _all_types_manifest(**overrides) -> dict:
    manifest = {
        "name": "everything", "version": "1.0",
        "permissions": ["tools", "skills", "settings", "prompts", "commands",
                        "context", "subagents", "hooks", "subprocess", "net"],
        "tools": [{"name": "a_tool", "description": "d", "handler": "a_tool",
                   "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
        "skills": [{"name": "t_skill", "factory": "build_skill"}],
        "settings": [{"key": "k", "label": "K", "kind": "cycle",
                      "choices": [1, 2], "default": "1"}],
        "prompts": {"demo.greeting": "hi"},
        "commands": {"/demo": {"description": "d", "model": "m", "tier": "fast"}},
        "mcp": [{"name": "srv", "command": ["true"], "startup_timeout_s": 1}],
        "subagents": [{"name": "spec", "system_prompt": "p", "tools": ["a_tool"]}],
        "context": [{"source": "f:1", "content": "hello", "priority": 50,
                     "trigger": "always"},
                    {"source": "f:2", "factory": "make_fragment"}],
        "hooks": [{"when": "pre_tool", "handler": "deny_hook"},
                  {"when": "post_tool", "handler": "post_hook"},
                  {"when": "on_final", "handler": "final_hook"}],
    }
    manifest.update(overrides)
    return manifest


def _loaded(tmp_path: Path, manifest: dict):
    host = _host(tmp_path, manifest)
    plugins = host.discover()
    assert len(plugins) == 1, [plugin.error for plugin in plugins]
    assert plugins[0].ok, plugins[0].error
    return host, plugins[0]


# --- 0.1 the seven types, all at once ---------------------------------------


def test_a_plugin_can_contribute_all_seven_types(tmp_path: Path):
    host, plugin = _loaded(tmp_path, _all_types_manifest())
    assert [tool.name for tool in plugin.tools] == ["a_tool"]
    assert [skill.name for skill in plugin.skills] == ["t_skill"]
    assert [setting.key for setting in plugin.settings] == ["k"]
    assert plugin.prompts == {"demo.greeting": "hi"}
    assert "/demo" in plugin.commands
    assert [server.name for server in plugin.mcp_servers] == ["srv"]
    assert [agent.name for agent in plugin.subagents] == ["spec"]
    assert sorted(fragment.source for fragment in plugin.context) == ["f:1", "f:2"]
    assert plugin.hooks.pre_tool and plugin.hooks.post_tool and plugin.hooks.on_final


def test_every_contributed_thing_reaches_the_core_through_one_accessor(tmp_path: Path):
    host, _ = _loaded(tmp_path, _all_types_manifest())
    assert [t.name for t in host.build_tools(tmp_path)] == ["a_tool"]
    assert "t_skill" in host.skills()
    assert [s.key for s in host.settings()] == ["k"]
    assert host.prompt_overrides() == {"demo.greeting": "hi"}
    assert [s.name for s in host.mcp_servers()] == ["srv"]
    assert list(host.subagent_modes()) == ["spec"]
    # Ordered by priority, not declaration order: f:2 is a project-priority
    # fragment and outranks the reference-priority f:1.
    assert [f.source for f in host.context_fragments()] == ["f:2", "f:1"]
    assert host.command_specs()["/demo"].tier == "fast"


# --- commands carry a model/tier override -----------------------------------


def test_a_command_can_pin_a_model_and_tier(tmp_path: Path):
    """The point of the extension: pin a cheap model for a mechanical command
    without touching config."""
    host, _ = _loaded(tmp_path, _all_types_manifest())
    spec = host.command_specs()["/demo"]
    assert spec.model == "m" and spec.tier == "fast"
    assert spec.argument_schema == {}


def test_a_command_may_carry_an_argument_schema(tmp_path: Path):
    manifest = _all_types_manifest()
    manifest["commands"] = {"/demo": {
        "description": "d",
        "argument_schema": {"type": "object",
                            "properties": {"path": {"type": "string"}}}}}
    host, _ = _loaded(tmp_path, manifest)
    assert host.command_specs()["/demo"].argument_schema["properties"] == {"path": {"type": "string"}}


def test_a_plain_string_command_still_works(tmp_path: Path):
    manifest = _all_types_manifest(commands={"/plain": "just a description"})
    host, _ = _loaded(tmp_path, manifest)
    assert host.commands()["/plain"] == "just a description"
    assert host.command_specs()["/plain"].tier == ""


# --- MCP: remote tools are net-risk and cannot be downgraded ----------------


def test_an_mcp_server_pins_its_tools_to_net_risk(tmp_path: Path):
    """The data crosses a process boundary to code the user did not write. The
    manifest cannot talk it down to `read`."""
    manifest = _all_types_manifest()
    manifest["mcp"] = [{"name": "srv", "command": ["true"], "risk": "read"}]
    host, plugin = _loaded(tmp_path, manifest)
    assert plugin.mcp_servers[0].risk == "net", "a remote tool was downgraded below net"


def test_an_mcp_server_without_subprocess_permission_is_refused(tmp_path: Path):
    manifest = _all_types_manifest()
    manifest["permissions"] = [p for p in manifest["permissions"] if p != "subprocess"]
    manifest["mcp"] = [{"name": "srv", "command": ["true"]}]
    host = _host(tmp_path, manifest)
    plugins = host.discover()
    assert not plugins[0].ok
    assert "subprocess" in plugins[0].error


def test_an_mcp_server_needs_a_command_list(tmp_path: Path):
    manifest = _all_types_manifest(mcp=[{"name": "srv", "command": "not-a-list"}])
    plugins = _host(tmp_path, manifest).discover()
    assert not plugins[0].ok and "command" in plugins[0].error


# --- subagents surface as modes, not new tools ------------------------------


def test_a_subagent_is_not_a_new_tool(tmp_path: Path):
    """It must be a mode of the existing delegation tool, so the agent loop
    needs no branch."""
    host, _ = _loaded(tmp_path, _all_types_manifest())
    assert "demo_specialist" not in [tool.name for tool in host.build_tools(tmp_path)]
    assert "spec" in host.subagent_modes()


def test_a_subagent_needs_a_prompt_and_a_tool_allowlist(tmp_path: Path):
    for bad in ({"name": "s", "tools": ["a_tool"]},
                {"name": "s", "system_prompt": "p"},
                {"name": "s", "system_prompt": "p", "tools": []}):
        manifest = _all_types_manifest(subagents=[bad])
        plugins = _host(tmp_path, manifest).discover()
        assert not plugins[0].ok, f"{bad} should have been refused"


def test_a_subagent_without_permission_is_refused(tmp_path: Path):
    manifest = _all_types_manifest()
    manifest["permissions"] = [p for p in manifest["permissions"] if p != "subagents"]
    plugins = _host(tmp_path, manifest).discover()
    assert not plugins[0].ok and "subagent" in plugins[0].error


# --- context fragments: proposals, validated at load ------------------------


def test_a_context_fragment_needs_a_known_trigger(tmp_path: Path):
    manifest = _all_types_manifest()
    manifest["context"] = [{"source": "f", "content": "x", "trigger": "whenever"}]
    plugins = _host(tmp_path, manifest).discover()
    assert not plugins[0].ok and "trigger" in plugins[0].error.lower()


def test_a_context_fragment_needs_a_known_priority(tmp_path: Path):
    manifest = _all_types_manifest()
    manifest["context"] = [{"source": "f", "content": "x", "priority": 999,
                            "trigger": "always"}]
    plugins = _host(tmp_path, manifest).discover()
    assert not plugins[0].ok and "priority" in plugins[0].error


def test_a_context_fragment_records_its_provenance(tmp_path: Path):
    """A user must be able to see which plugin put text in their context."""
    host, _ = _loaded(tmp_path, _all_types_manifest())
    assert all(f.provenance == "everything" for f in host.context_fragments())


def test_context_fragments_are_ordered_by_priority(tmp_path: Path):
    host, _ = _loaded(tmp_path, _all_types_manifest())
    priorities = [f.effective_priority for f in host.context_fragments()]
    assert priorities == sorted(priorities)


def test_context_without_permission_is_refused(tmp_path: Path):
    manifest = _all_types_manifest()
    manifest["permissions"] = [p for p in manifest["permissions"] if p != "context"]
    plugins = _host(tmp_path, manifest).discover()
    assert not plugins[0].ok and "context" in plugins[0].error


# --- 0.2 two-phase hooks that can deny --------------------------------------


def test_a_pre_tool_hook_denies_and_the_reason_survives(tmp_path: Path):
    host, _ = _loaded(tmp_path, _all_types_manifest())
    arguments, reason, plugin = host.consult_pre_tool("bad", {})
    assert reason and "blocked" in reason
    assert plugin == "everything", "a denial names who made it"
    allowed, reason, _ = host.consult_pre_tool("a_tool", {})
    assert reason == ""


def test_a_pre_tool_hook_can_rewrite_arguments(tmp_path: Path):
    module = MODULE + '''

def rewrite_hook(tool_name, arguments):
    if tool_name == "a_tool":
        return HookVerdict("rewrite", arguments={**arguments, "path": "rewritten"})
    return ALLOW
'''
    host = _host(tmp_path, _all_types_manifest())
    plugin_file = tmp_path / ".harness" / "plugins" / "everything" / "plugin.py"
    plugin_file.write_text(module, encoding="utf-8")
    manifest = _all_types_manifest()
    manifest["hooks"] = [{"when": "pre_tool", "handler": "rewrite_hook"}]
    (tmp_path / ".harness" / "plugins" / "everything" / "plugin.plugin.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    host.discover()
    arguments, reason, _ = host.consult_pre_tool("a_tool", {"path": "original"})
    assert reason == ""
    assert arguments["path"] == "rewritten"


def test_a_hook_that_raises_blocks_rather_than_failing_open(tmp_path: Path):
    """A guard that crashes open is not a guard."""
    module = MODULE.replace(
        "def deny_hook(tool_name, arguments):",
        "def deny_hook(tool_name, arguments):\n    raise RuntimeError('hook exploded')")
    host = _host(tmp_path, _all_types_manifest())
    (tmp_path / ".harness" / "plugins" / "everything" / "plugin.py").write_text(module, encoding="utf-8")
    host.discover()
    _, reason, _ = host.consult_pre_tool("a_tool", {})
    assert "hook exploded" in reason
    assert "blocked rather than run unchecked" in reason


def test_a_denial_without_a_reason_is_a_load_error(tmp_path: Path):
    """A silent denial is indistinguishable from a hang to the user."""
    with pytest.raises(ValueError):
        HookVerdict("deny")
    assert HookVerdict("deny", reason="because").action == "deny"


def test_a_post_tool_hook_can_redact_success_but_never_failure(tmp_path: Path):
    """Errors are never filtered or summarized -- sacred invariant 5."""
    module = MODULE + '''

def redactor(tool_name, arguments, output, success):
    return "REDACTED" if success else output
'''
    host = _host(tmp_path, _all_types_manifest())
    (tmp_path / ".harness" / "plugins" / "everything" / "plugin.py").write_text(module, encoding="utf-8")
    manifest = _all_types_manifest()
    manifest["hooks"] = [{"when": "post_tool", "handler": "redactor"}]
    (tmp_path / ".harness" / "plugins" / "everything" / "plugin.plugin.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    host.discover()

    redacted, _ = host.run_post_tool("a_tool", {}, "the real output", True)
    assert redacted == "REDACTED"
    untouched, _ = host.run_post_tool("a_tool", {}, "ERROR: it broke", False)
    assert untouched == "ERROR: it broke", "a failure's detail must survive"


def test_a_post_tool_hook_can_attach_metadata(tmp_path: Path):
    host, _ = _loaded(tmp_path, _all_types_manifest())
    _, metadata = host.run_post_tool("a_tool", {}, "out", True)
    assert metadata == {"seen": "a_tool"}


def test_a_hook_naming_a_missing_function_is_a_load_error(tmp_path: Path):
    """A policy you think is enforced and is not is the worst outcome here."""
    manifest = _all_types_manifest(hooks=[{"when": "pre_tool", "handler": "not_defined"}])
    plugins = _host(tmp_path, manifest).discover()
    assert not plugins[0].ok and "not_defined" in plugins[0].error


def test_an_unknown_hook_phase_is_a_load_error(tmp_path: Path):
    manifest = _all_types_manifest(hooks=[{"when": "during_lunch", "handler": "x"}])
    plugins = _host(tmp_path, manifest).discover()
    assert not plugins[0].ok and "during_lunch" in plugins[0].error


def test_a_denial_requires_the_hooks_permission(tmp_path: Path):
    manifest = _all_types_manifest()
    manifest["permissions"] = [p for p in manifest["permissions"] if p != "hooks"]
    plugins = _host(tmp_path, manifest).discover()
    assert not plugins[0].ok and "hooks" in plugins[0].error


# --- triggers are cheap and never spend a model token -----------------------


@pytest.mark.parametrize("raw,text,tools,commands,expected", [
    ("always", "anything", (), (), True),
    ("keyword:fallback", "prefer the fallback path", (), (), True),
    ("keyword:fallback", "no match here", (), (), False),
    ("regex:\\bgit\\b", "run git status", (), (), True),
    ("regex:\\bgit\\b", "run the tests", (), (), False),
    ("command:/build", "", (), ("/build now",), True),
    ("command:/build", "", (), ("/test now",), False),
    ("on_tool:run_bash", "", ("run_bash",), (), True),
    ("on_tool:run_bash", "", ("read_file",), (), False),
])
def test_triggers_match_without_a_model(raw, text, tools, commands, expected):
    """A trigger is a cheap local predicate. It must never spend a model token
    to decide whether content is worth considering."""
    assert Trigger.parse(raw).matches(text=text, tools_used=tools,
                                      commands=commands) is expected


def test_a_malformed_regex_does_not_admit_content():
    """Failing to match is the safe direction for a broken pattern."""
    assert Trigger("regex", "[unclosed").matches(text="anything") is False


def test_an_unknown_trigger_is_rejected():
    with pytest.raises(ValueError):
        Trigger.parse("sometime")


def test_a_pinned_fragment_must_use_the_pinned_priority():
    """`pinned` and the eviction order must not be able to disagree."""
    with pytest.raises(ValueError):
        ContextFragment(source="s", content="c", tokens=1,
                        priority=PRIORITY_REFERENCE, trigger=Trigger.parse("always"),
                        pinned=True)


# --- the agent loop gained no branches --------------------------------------


def test_the_agent_loop_contains_no_plugin_provenance_branch():
    """Sacred invariant 3: the loop never branches on provenance.

    The loop consults the host for behaviour -- is this tool read-only, may this
    call proceed -- but it never asks where a tool came from.
    """
    source = (Path(__file__).resolve().parent.parent / "src" / "adaptive_harness"
              / "agent" / "agent.py").read_text(encoding="utf-8")
    # Strip comments and docstrings before looking, so prose about plugins is
    # not mistaken for a branch.
    code = re.sub(r'"""(?:.|\n)*?"""', "", source)
    code = re.sub(r"#.*", "", code)
    for banned in ("if is_plugin", "if plugin", "is_plugin =", "from_plugin",
                   "if spec.plugin", "if tool.plugin"):
        assert banned not in code, f"the agent loop branches on provenance: {banned}"
