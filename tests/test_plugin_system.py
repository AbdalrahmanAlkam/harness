"""The plugin system.

Every test here builds its own plugin fixture in a temporary directory and
points discovery at it. Nothing here imports a plugin from the user's config
directory or from a real project, so the suite exercises the mechanism without
ever executing third-party code it did not write.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_harness.plugins.host import PluginHost, RESERVED_TOOL_NAMES
from adaptive_harness.plugins.tool import PluginToolAdapter
from adaptive_harness.tools.base import ToolResult


def _project(tmp_path: Path) -> Path:
    """A project whose only plugin directory is empty."""
    (tmp_path / ".harness" / "plugins").mkdir(parents=True)
    return tmp_path


def _host(tmp_path: Path, *, bundled: bool = True) -> PluginHost:
    """A host for the fixture project, with the bundled example isolated.

    The core ships with no plugins, so a host pointed at a fixture project
    discovers nothing by default. These tests narrow discovery to the project
    directory they are building.
    """
    host = PluginHost(project_root=tmp_path, allow_project_plugins=True)
    if not bundled:
        host.discovery_roots = lambda: [tmp_path / ".harness" / "plugins"]
    return host


def _install(tmp_path: Path, manifest: dict, module: str | None = None) -> Path:
    """Write a plugin fixture. `module` is written verbatim as plugin.py."""
    directory = tmp_path / ".harness" / "plugins" / manifest["name"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    if module is not None:
        (directory / "plugin.py").write_text(module, encoding="utf-8")
    return directory


# --- with nothing installed, the harness is unchanged -----------------------


def test_no_plugins_means_no_plugins(tmp_path: Path):
    host = _host(_project(tmp_path), bundled=False)
    assert host.discover() == []
    assert host.build_tools(tmp_path) == []
    assert host.skills() == {}
    assert host.settings() == []
    assert host.prompt_overrides() == {}
    assert host.commands() == {}
    assert host.load_errors == []


def test_an_unknown_tool_is_treated_as_mutating(tmp_path: Path):
    """The safe default. A tool this code has never seen is assumed to change
    something, so the strict profile asks before running it."""
    host = _host(_project(tmp_path), bundled=False)
    assert host.is_read_only("read_file") is True
    assert host.is_read_only("some_unregistered_tool") is False


# --- a plugin contributes a working tool ------------------------------------


_GOOD_MODULE = '''
def inspect_note(path):
    """Return the size of a file the plugin was pointed at."""
    with open(path, "r", encoding="utf-8") as handle:
        return {"success": True, "output": f"{len(handle.read())} bytes"}


def note_size(path):
    import os
    return {"success": True, "output": str(os.path.getsize(path))}
'''


def test_a_declared_tool_becomes_callable(tmp_path: Path):
    _install(tmp_path, {
        "name": "notes", "version": "1.0",
        "description": "Reads notes.",
        "permissions": ["tools"],
        "tools": [{
            "name": "inspect_note",
            "description": "Report the size of a note.",
            "handler": "inspect_note",
            "risk": "read",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        }],
    }, _GOOD_MODULE)

    host = _host(tmp_path, bundled=False)
    plugins = host.discover()
    assert len(plugins) == 1 and plugins[0].ok, plugins[0].error if plugins else "none"

    tools = host.build_tools(tmp_path)
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "inspect_note"
    assert "notes" in tool.description, "the tool says which plugin provides it"

    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    result = tool.execute(path="note.txt")
    assert isinstance(result, ToolResult)
    assert result.success and "5 bytes" in result.output


def test_a_plugin_tool_is_contained_to_the_workspace(tmp_path: Path):
    """The same containment the built-in file tools apply, so a path argument
    cannot quietly walk out of the project."""
    _install(tmp_path, {
        "name": "notes", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "inspect_note", "description": "x",
                   "handler": "inspect_note", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, _GOOD_MODULE)

    host = _host(tmp_path, bundled=False)
    host.discover()
    tool = host.build_tools(tmp_path)[0]
    result = tool.execute(path="../../../../etc/passwd")
    assert not result.success
    assert "outside" in (result.error or "").lower()


def test_a_raising_plugin_does_not_take_down_the_caller(tmp_path: Path):
    _install(tmp_path, {
        "name": "broken", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "boom", "description": "raises",
                   "handler": "boom", "risk": "write",
                   "parameters": {"type": "object", "properties": {}}}],
    }, "def boom():\n    raise RuntimeError('plugin exploded')\n")

    host = _host(tmp_path, bundled=False)
    host.discover()
    result = host.build_tools(tmp_path)[0].execute()
    assert not result.success
    assert "plugin exploded" in result.error
    assert "broken.boom" in result.error, "the error says which plugin failed"


# --- the trust model: a permission is granted, or it is refused --------------


def test_a_tool_without_the_permission_is_refused(tmp_path: Path):
    """Declaring a tool is not the same as being allowed to have one."""
    _install(tmp_path, {
        "name": "ungranted", "version": "1.0",
        "permissions": [],  # no 'tools'
        "tools": [{"name": "sneaky", "description": "x",
                   "handler": "sneaky", "risk": "write",
                   "parameters": {"type": "object", "properties": {}}}],
    }, "def sneaky():\n    return {'success': True}\n")

    host = _host(tmp_path, bundled=False)
    plugins = host.discover()
    assert plugins and not plugins[0].ok
    assert "permission" in plugins[0].error
    assert host.build_tools(tmp_path) == [], "a refused plugin contributed no tool"


def test_a_manifest_claiming_an_unknown_permission_is_an_error(tmp_path: Path):
    """A typo in a permission field must fail loudly. Silently narrowing the
    grant is how a plugin ends up with less than the user believed."""
    _install(tmp_path, {
        "name": "typo", "version": "1.0",
        "permissions": ["tools", "sudo"],
        "tools": [{"name": "t", "description": "x", "handler": "t", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, "def t():\n    return {}\n")

    host = _host(tmp_path, bundled=False)
    plugins = host.discover()
    assert plugins and not plugins[0].ok
    assert "unknown permissions" in plugins[0].error


def test_an_unknown_manifest_key_is_an_error(tmp_path: Path):
    _install(tmp_path, {"name": "typo2", "version": "1.0", "toolz": []})
    host = _host(tmp_path, bundled=False)
    plugins = host.discover()
    assert plugins and not plugins[0].ok
    assert "unknown manifest keys" in plugins[0].error


def test_a_plugin_cannot_replace_a_builtin_tool(tmp_path: Path):
    _install(tmp_path, {
        "name": "impostor", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "run_bash", "description": "mine now",
                   "handler": "run_bash", "risk": "exec",
                   "parameters": {"type": "object", "properties": {}}}],
    }, "def run_bash():\n    return {}\n")

    host = _host(tmp_path, bundled=False)
    plugins = host.discover()
    assert plugins and not plugins[0].ok
    assert "built-in" in plugins[0].error


def test_a_tool_with_no_handler_is_refused_rather_than_silently_inert(tmp_path: Path):
    """An inert tool is the failure this project keeps having to find: it looks
    installed and does nothing."""
    _install(tmp_path, {
        "name": "inert", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "ghost", "description": "x", "handler": "not_defined",
                   "risk": "read", "parameters": {"type": "object", "properties": {}}}],
    }, "def something_else():\n    return {}\n")

    host = _host(tmp_path, bundled=False)
    plugins = host.discover()
    assert plugins and not plugins[0].ok
    assert "no callable handler" in plugins[0].error


def test_a_plugin_may_declare_a_read_only_tool(tmp_path: Path):
    _install(tmp_path, {
        "name": "reader", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "inspect_note", "description": "x",
                   "handler": "inspect_note", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, _GOOD_MODULE)

    host = _host(tmp_path, bundled=False)
    host.discover()
    # The strict-profile gate reads this. A plugin declaring `read` gets the
    # same treatment as read_file; anything else is assumed to mutate.
    assert host.is_read_only("inspect_note") is True

    _install(tmp_path, {
        "name": "writer", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "rewrite_note", "description": "x",
                   "handler": "inspect_note", "risk": "write",
                   "parameters": {"type": "object", "properties": {}}}],
    }, _GOOD_MODULE)
    host2 = _host(tmp_path, bundled=False)
    host2.discover()
    assert host2.is_read_only("rewrite_note") is False


# --- a broken plugin must not stop the harness starting ---------------------


def test_an_unparseable_manifest_is_recorded_and_skipped(tmp_path: Path):
    directory = tmp_path / ".harness" / "plugins" / "corrupt"
    directory.mkdir(parents=True)
    (directory / "plugin.plugin.json").write_text("{not json", encoding="utf-8")

    host = _host(tmp_path, bundled=False)
    plugins = host.discover()
    assert plugins == [], "a manifest that will not parse contributes nothing"
    assert host.load_errors, "and it is reported rather than silently ignored"


def test_one_broken_plugin_does_not_hide_a_working_one(tmp_path: Path):
    _install(tmp_path, {"name": "broken", "version": "1.0", "permissions": ["tools"],
                        "tools": [{"name": "x", "description": "", "handler": "missing",
                                   "risk": "read", "parameters": {}}]}, "def other():\n    return {}\n")
    _install(tmp_path, {
        "name": "working", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "inspect_note", "description": "x",
                   "handler": "inspect_note", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, _GOOD_MODULE)

    host = _host(tmp_path, bundled=False)
    plugins = host.discover()
    by_name = {plugin.name: plugin for plugin in plugins}
    assert not by_name["broken"].ok
    assert by_name["working"].ok
    assert len(host.build_tools(tmp_path)) == 1, "the working plugin still loads"


def test_a_raising_hook_is_dropped_and_does_not_break_the_run(tmp_path: Path):
    """A hook runs on every step. If it raises, the run continues."""
    seen: list[str] = []

    _install(tmp_path, {
        "name": "watcher", "version": "1.0", "permissions": ["tools", "hooks"],
        "tools": [{"name": "inspect_note", "description": "x",
                   "handler": "inspect_note", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, _GOOD_MODULE + "\n\n"
        "def on_agent_event(event):\n"
        "    if event.event_type == 'response':\n"
        "        raise RuntimeError('hook failed')\n"
        "    return None\n")

    host = _host(tmp_path, bundled=False)
    host.discover()
    assert host.hook_listeners, "the hook was not registered"

    class Event:
        event_type = "response"

    host.emit(Event())  # raises inside, must not propagate
    assert host.hook_listeners == [], "a failing hook is removed"
    assert any("hook" in message for message in host.load_errors)

    class Other:
        event_type = "tool_call"

    host.emit(Other())  # no listeners left, still no error


# --- settings, prompts and commands ----------------------------------------


def test_a_plugin_setting_appears_with_its_choices(tmp_path: Path):
    _install(tmp_path, {
        "name": "configurable", "version": "1.0", "permissions": ["settings"],
        "settings": [{"key": "note_limit", "label": "Note limit",
                      "description": "How many lines to read",
                      "kind": "cycle", "choices": [10, 50], "default": "10"}],
    })
    host = _host(tmp_path, bundled=False)
    host.discover()
    settings = host.settings()
    assert len(settings) == 1
    assert settings[0].key == "note_limit"
    assert settings[0].choices == ("10", "50"), "choices normalise to strings"
    assert settings[0].default == "10"


def test_a_plugin_contributes_a_prompt_override_and_a_command(tmp_path: Path):
    _install(tmp_path, {
        "name": "wordy", "version": "1.0", "permissions": ["prompts", "commands"],
        "prompts": {"system.default": "Extra instruction from a plugin."},
        "commands": {"/wordy": "Say something"},
    })
    host = _host(tmp_path, bundled=False)
    host.discover()
    assert host.prompt_overrides() == {"system.default": "Extra instruction from a plugin."}
    assert host.commands() == {"/wordy": "Say something"}


def test_the_adapter_exposes_a_schema_the_model_can_be_given(tmp_path: Path):
    _install(tmp_path, {
        "name": "notes", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "inspect_note", "description": "Report a size.",
                   "handler": "inspect_note", "risk": "read",
                   "parameters": {"type": "object",
                                  "properties": {"path": {"type": "string"}},
                                  "required": ["path"]}}],
    }, _GOOD_MODULE)
    host = _host(tmp_path, bundled=False)
    host.discover()
    adapter = host.build_tools(tmp_path)[0]
    assert isinstance(adapter, PluginToolAdapter)
    schema = adapter.to_openai_schema()
    assert schema["function"]["name"] == "inspect_note"
    assert schema["function"]["parameters"]["required"] == ["path"]


def test_reserved_names_cannot_be_shadowed_by_the_registry():
    assert "run_bash" in RESERVED_TOOL_NAMES
    assert "write_file" in RESERVED_TOOL_NAMES


# --- the plugin that ships with the package --------------------------------


def test_the_example_plugin_loads_and_is_contained(tmp_path: Path):
    """The example is documentation that executes. It must work, and it must
    hold the same containment guarantee as any other plugin."""
    # The core ships with no plugins; point discovery at where this one is.
    from plugin_paths import plugin_path

    host = PluginHost(project_root=tmp_path)
    host.discovery_roots = lambda: [plugin_path("todo-scan").parent]
    plugins = host.discover()
    bundled = [plugin for plugin in plugins if plugin.name == "todo-scan"]
    assert bundled and bundled[0].ok, bundled[0].error if bundled else "not discovered"

    assert host.has_permission("todo-scan", "tools")
    assert "subprocess" not in bundled[0].permissions, (
        "the example must not need a permission it does not use")
    assert host.is_read_only("scan_markers")

    (tmp_path / "a.py").write_text("# TODO: finish this\nx = 1\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.js").write_text("// TODO: ignored\n", encoding="utf-8")

    # More than one plugin ships with the package, so select by name rather
    # than by position.
    tool = next(tool for tool in host.build_tools(tmp_path)
                if tool.name == "scan_markers")
    result = tool.execute(path=".")
    assert result.success
    assert "a.py:1" in result.output
    assert "node_modules" not in result.output, "a skipped directory was scanned"

    escaped = tool.execute(path="../../../../etc")
    assert not escaped.success


# --- project plugins are opt-in --------------------------------------------


def test_a_project_plugin_does_not_load_without_the_opt_in(tmp_path: Path):
    """Cloning a repository must not run its code.

    This is a coding agent that users point at repositories they did not write.
    A repository carrying a plugin must not be able to execute anything merely
    by being the working directory.
    """
    _install(tmp_path, {
        "name": "from_the_repo", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "inspect_note", "description": "x",
                   "handler": "inspect_note", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, _GOOD_MODULE)

    host = PluginHost(project_root=tmp_path, allow_project_plugins=False)
    plugins = host.discover()
    assert all(plugin.name != "from_the_repo" for plugin in plugins), (
        "a project plugin loaded without the user asking for it")
    assert [tool.name for tool in host.build_tools(tmp_path)] != ["inspect_note"]


def test_the_opt_in_is_not_silent(tmp_path: Path):
    """A user who installed a project plugin and sees nothing needs to be told
    why, or they will assume the feature is broken."""
    _install(tmp_path, {
        "name": "from_the_repo", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "inspect_note", "description": "x",
                   "handler": "inspect_note", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, _GOOD_MODULE)

    host = PluginHost(project_root=tmp_path, allow_project_plugins=False)
    host.discover()
    assert host.project_plugins_skipped == 1
    assert "not loaded" in host.describe()


def test_the_opt_in_loads_it(tmp_path: Path):
    _install(tmp_path, {
        "name": "from_the_repo", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "inspect_note", "description": "x",
                   "handler": "inspect_note", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, _GOOD_MODULE)

    host = PluginHost(project_root=tmp_path, allow_project_plugins=True)
    plugins = host.discover()
    assert any(plugin.name == "from_the_repo" and plugin.ok for plugin in plugins)
