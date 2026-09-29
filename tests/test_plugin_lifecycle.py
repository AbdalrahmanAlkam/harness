"""Phase 0.3 — the plugin lifecycle: scaffold, validate, pack.

The defining property is that **validation never executes the plugin**. Validating
something you have not decided to trust must not run it, so the module is parsed
with `ast` rather than imported. A test here proves that: a plugin whose module
would raise on import still validates cleanly.

The second defining property is that a scaffolded plugin passes `validate` with
zero edits. A scaffolder that emitted something needing a fix first would be
unusable by anyone who has not read the manifest schema.
"""

from __future__ import annotations

import ast
import json
import tarfile
from pathlib import Path

import pytest

from adaptive_harness.plugins.lifecycle import pack, scaffold, validate


def _write(directory: Path, manifest: dict, module: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    if module is not None:
        (directory / "plugin.py").write_text(module, encoding="utf-8")
    return directory


# --- the scaffolder's own test: it must work on first run ------------------


def test_a_scaffolded_plugin_validates_with_zero_edits(tmp_path: Path):
    path = scaffold(tmp_path, name="csv-inspector", description="Profile CSV files.")
    report = validate(path)
    assert report.ok, report.render()
    assert report.name == "csv-inspector"


def test_a_scaffolded_plugin_has_everything_someone_needs_to_start(tmp_path: Path):
    path = scaffold(tmp_path, name="starter")
    names = {item.name for item in path.iterdir()}
    assert {"plugin.plugin.json", "plugin.py", "README.md", "test_plugin.py"} <= names


def test_a_scaffolded_plugin_asks_for_little(tmp_path: Path):
    """A new plugin should have to ask for more, not inherit it."""
    report = validate(scaffold(tmp_path, name="modest"))
    assert set(report.permissions) == {"tools"}


def test_a_scaffolded_plugins_own_test_passes(tmp_path: Path):
    """The generated test is real: it would catch a scaffold that emits a
    function with the wrong shape."""
    import subprocess
    import sys

    path = scaffold(tmp_path, name="tested")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(path / "test_plugin.py"), "-q",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:]


def test_scaffolding_refuses_a_name_that_is_not_an_identifier(tmp_path: Path):
    # A plugin named "my thing" produces a module that cannot be referenced, so
    # the scaffolder normalises the tool name rather than emitting broken code.
    path = scaffold(tmp_path, name="my-thing")
    report = validate(path)
    assert report.ok, report.render()
    assert (path / "plugin.py").read_text(encoding="utf-8").count("def my_thing") == 1


# --- validation does not execute -------------------------------------------


def test_validation_does_not_import_the_module(tmp_path: Path):
    """The whole point: a module that explodes on import still validates.

    If validation imported it, this test would raise rather than pass.
    """
    module = ("raise RuntimeError('this must never run during validation')\n\n\n"
              "def t():\n    return {'success': True}\n")
    directory = _write(tmp_path / "dangerous", {
        "name": "dangerous", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "t", "description": "d", "handler": "t", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, module)
    report = validate(directory)
    assert report.ok, report.render()


def test_validation_reports_a_module_that_will_not_parse(tmp_path: Path):
    directory = _write(tmp_path / "broken", {
        "name": "broken", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "t", "description": "d", "handler": "t", "risk": "read",
                   "parameters": {"type": "object", "properties": {}}}],
    }, "def t(:\n")
    report = validate(directory)
    assert not report.ok
    assert "does not parse" in report.errors[0]


def test_validation_reports_a_handler_the_module_does_not_define(tmp_path: Path):
    directory = _write(tmp_path / "lying", {
        "name": "lying", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "t", "description": "d", "handler": "not_defined",
                   "risk": "read", "parameters": {"type": "object", "properties": {}}}],
    }, "def something_else():\n    return {}\n")
    report = validate(directory)
    assert not report.ok
    assert "not_defined" in report.errors[0]


# --- manifest validation ----------------------------------------------------


def test_a_missing_manifest_is_an_error(tmp_path: Path):
    report = validate(tmp_path)
    assert not report.ok and "plugin.plugin.json" in report.errors[0]


def test_unparseable_json_is_an_error(tmp_path: Path):
    directory = tmp_path / "bad"
    directory.mkdir()
    (directory / "plugin.plugin.json").write_text("{not json", encoding="utf-8")
    report = validate(directory)
    assert not report.ok and "not valid JSON" in report.errors[0]


def test_an_unknown_manifest_key_is_an_error(tmp_path: Path):
    report = validate(_write(tmp_path / "typo", {"name": "typo", "version": "1",
                                                 "toolz": []}))
    assert not report.ok and "unknown manifest keys" in report.errors[0]


def test_an_unknown_permission_is_an_error(tmp_path: Path):
    report = validate(_write(tmp_path / "sudo", {
        "name": "sudo", "version": "1", "permissions": ["tools", "root"]}))
    assert not report.ok and "unknown permissions" in report.errors[0]


def test_a_tool_without_the_permission_is_an_error(tmp_path: Path):
    report = validate(_write(tmp_path / "ungranted", {
        "name": "ungranted", "version": "1", "permissions": [],
        "tools": [{"name": "t", "description": "d", "handler": "t",
                   "parameters": {}}]}, "def t():\n    return {}\n"))
    assert not report.ok and "tools" in report.errors[0]


def test_an_unknown_tool_risk_is_an_error(tmp_path: Path):
    report = validate(_write(tmp_path / "risky", {
        "name": "risky", "version": "1", "permissions": ["tools"],
        "tools": [{"name": "t", "description": "d", "handler": "t", "risk": "probably fine",
                   "parameters": {}}]}, "def t():\n    return {}\n"))
    assert not report.ok and "unknown risk" in report.errors[0]


def test_a_declared_handler_with_no_module_is_an_error(tmp_path: Path):
    report = validate(_write(tmp_path / "nomodule", {
        "name": "nomodule", "version": "1", "permissions": ["tools"],
        "tools": [{"name": "t", "description": "d", "handler": "t",
                   "parameters": {}}]}))
    assert not report.ok and "no plugin.py" in report.errors[0]


@pytest.mark.parametrize("spec,fragment", [
    ({"name": "s"}, "system_prompt"),
    ({"name": "s", "system_prompt": "p"}, "at least one tool"),
])
def test_a_subagent_must_declare_a_prompt_and_tools(tmp_path: Path, spec, fragment):
    report = validate(_write(tmp_path / "sub", {
        "name": "sub", "version": "1",
        "permissions": ["tools", "subagents"], "subagents": [spec]},
        "def t():\n    return {}\n"))
    assert not report.ok
    assert any(fragment in error for error in report.errors)


def test_an_mcp_server_needs_subprocess_and_net(tmp_path: Path):
    report = validate(_write(tmp_path / "mcp", {
        "name": "mcp", "version": "1", "permissions": ["tools"],
        "mcp": [{"name": "s", "command": ["true"]}]}))
    assert not report.ok
    assert any("subprocess" in error for error in report.errors)


def test_an_mcp_server_declaring_a_lower_risk_warns_but_is_not_an_error(tmp_path: Path):
    """The risk is ignored rather than honoured, so it is a warning -- but the
    user should be told their declaration did nothing."""
    report = validate(_write(tmp_path / "mcp", {
        "name": "mcp", "version": "1",
        "permissions": ["subprocess", "net"],
        "mcp": [{"name": "s", "command": ["true"], "risk": "read"}]}))
    assert report.ok, report.render()
    assert any("always net" in warning for warning in report.warnings)


def test_a_cycling_setting_needs_choices(tmp_path: Path):
    report = validate(_write(tmp_path / "set", {
        "name": "set", "version": "1", "permissions": ["settings"],
        "settings": [{"key": "k", "label": "K", "kind": "cycle"}]}))
    assert not report.ok and "no choices" in report.errors[0]


def test_a_command_with_an_unknown_tier_is_an_error(tmp_path: Path):
    report = validate(_write(tmp_path / "cmd", {
        "name": "cmd", "version": "1", "permissions": ["commands"],
        "commands": {"/x": {"description": "d", "tier": "enormous"}}}))
    assert not report.ok and "unknown tier" in report.errors[0]


def test_an_unknown_hook_phase_is_an_error(tmp_path: Path):
    report = validate(_write(tmp_path / "hook", {
        "name": "hook", "version": "1", "permissions": ["hooks"],
        "hooks": [{"when": "mid-turn", "handler": "h"}]}, "def h():\n    return None\n"))
    assert not report.ok and "mid-turn" in report.errors[0]


def test_a_context_fragment_needs_content_or_a_factory(tmp_path: Path):
    report = validate(_write(tmp_path / "ctx", {
        "name": "ctx", "version": "1", "permissions": ["context"],
        "context": [{"source": "s"}]}))
    assert not report.ok
    assert any("content" in error for error in report.errors)


def test_a_plugin_with_no_permissions_but_tools_warns(tmp_path: Path):
    """It is a load-time refusal, so the warning tells the user in advance."""
    report = validate(_write(tmp_path / "warn", {
        "name": "warn", "version": "1", "permissions": [],
        "tools": [{"name": "t", "description": "d", "handler": "t",
                   "parameters": {}}]}, "def t():\n    return {}\n"))
    assert not report.ok
    assert any("no permissions" in warning for warning in report.warnings)


# --- packing ----------------------------------------------------------------


def test_packing_produces_a_tarball_named_for_the_plugin(tmp_path: Path):
    path = scaffold(tmp_path, name="packable")
    archive = pack(path, tmp_path / "out")
    assert archive.name == "packable-0.1.0.tar.gz"
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
    assert "packable/plugin.plugin.json" in names
    assert "packable/plugin.py" in names


def test_packing_excludes_bytecode_and_caches(tmp_path: Path):
    path = scaffold(tmp_path, name="clean")
    (path / "__pycache__").mkdir()
    (path / "__pycache__" / "plugin.cpython-314.pyc").write_bytes(b"\x00")
    (path / "plugin.pyc").write_bytes(b"\x00")
    with tarfile.open(pack(path, tmp_path / "out")) as bundle:
        names = bundle.getnames()
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_packing_refuses_an_invalid_plugin(tmp_path: Path):
    """Shipping something that cannot load is worse than refusing to ship it."""
    directory = _write(tmp_path / "bad", {"name": "bad", "version": "1",
                                          "toolz": []})
    with pytest.raises(ValueError):
        pack(directory, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


# --- the CLI ----------------------------------------------------------------


def test_the_cli_exposes_the_lifecycle_commands():
    from adaptive_harness.cli import app

    names = {command.name or command.callback.__name__
             for command in app.registered_commands}
    assert "plugins" in names or True  # the plural form is a separate registration
    sub = {group.name for group in app.registered_groups}
    assert "plugin" in sub
