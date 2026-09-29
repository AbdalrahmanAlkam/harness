"""Plugin lifecycle: scaffold, validate, pack.

Three operations, and one rule that shapes all of them: **validation must not
execute the plugin.** A manifest is checked, permissions are checked, and the
module is *parsed* rather than run. Importing a plugin to validate it would mean
`harness plugin validate` executes the very code you are trying to decide
whether to trust.

The scaffold is the other half of that rule. A scaffolded plugin must pass
`validate` on first run with zero edits, which is what makes the tool usable by
someone who has not read this file.
"""

from __future__ import annotations

import ast
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from adaptive_harness.plugins.host import MANIFEST_KEYS, PERMISSIONS
from adaptive_harness.plugins.types import Trigger, VALID_PRIORITIES

MANIFEST_NAME = "plugin.plugin.json"
MODULE_NAME = "plugin.py"

#: The permission set a scaffolded plugin gets. Deliberately minimal: a new
#: plugin should have to ask for more rather than inherit it.
SCAFFOLD_PERMISSIONS = ("tools",)


@dataclass
class ValidationReport:
    """The result of checking a plugin without running it."""

    path: Path
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    name: str = ""
    version: str = ""
    permissions: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"{self.path} — {self.name or '?'} {self.version}".rstrip()]
        if self.permissions:
            lines.append(f"  permissions: {', '.join(self.permissions) or 'none'}")
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        for error in self.errors:
            lines.append(f"  ERROR: {error}")
        lines.append("  OK" if self.ok else f"  {len(self.errors)} error(s)")
        return "\n".join(lines)


def validate(path: Path | str, *, import_module: bool = False) -> ValidationReport:
    """Check a plugin without executing it.

    The module is parsed with :mod:`ast`, not imported, so every declared
    handler, skill factory, context factory and hook is verified to *exist* and
    to be callable-shaped without a single line of it running. ``import_module``
    exists for the one case that genuinely needs it, and is opt-in.
    """
    directory = Path(path)
    if not directory.is_dir():
        directory = directory.parent if directory.suffix == ".json" else directory
    report = ValidationReport(path=directory)

    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        report.errors.append(f"no {MANIFEST_NAME} in {directory}")
        return report

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report.errors.append(f"{MANIFEST_NAME} is not valid JSON: {exc}")
        return report
    if not isinstance(manifest, dict):
        report.errors.append(f"{MANIFEST_NAME} must contain a JSON object")
        return report

    unknown = set(manifest) - MANIFEST_KEYS
    if unknown:
        report.errors.append(
            f"unknown manifest keys: {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(MANIFEST_KEYS))}")

    report.name = str(manifest.get("name", ""))
    report.version = str(manifest.get("version", ""))
    if not report.name:
        report.errors.append("the manifest has no name")
    elif not report.name.replace("-", "_").isidentifier():
        report.errors.append(f"plugin name {report.name!r} is not a valid identifier")
    if not report.version:
        report.errors.append("the manifest has no version")

    permissions = manifest.get("permissions", [])
    report.permissions = tuple(str(item) for item in permissions)
    if not isinstance(permissions, (list, tuple)):
        report.errors.append("permissions must be a list")
    else:
        unknown_permissions = set(permissions) - PERMISSIONS
        if unknown_permissions:
            report.errors.append(
                f"unknown permissions: {', '.join(sorted(unknown_permissions))}. "
                f"Known permissions: {', '.join(sorted(PERMISSIONS))}")

    module_path = directory / MODULE_NAME
    defined: set[str] = set()
    if module_path.is_file():
        try:
            tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        except (OSError, SyntaxError) as exc:
            report.errors.append(f"{MODULE_NAME} does not parse: {exc}")
            tree = None
        if tree is not None:
            defined = _top_level_names(tree)
    elif manifest.get("tools") or manifest.get("skills") or manifest.get("hooks") \
            or manifest.get("context"):
        # An MCP server is exempt: its code is the `command` it names, run as a
        # subprocess, so a plugin that only declares servers needs no module.
        report.errors.append(
            f"the manifest declares code but there is no {MODULE_NAME}")

    _check_tools(manifest, defined, report)
    _check_skills(manifest, defined, report)
    _check_settings(manifest, report)
    _check_commands(manifest, report)
    _check_mcp(manifest, report)
    _check_subagents(manifest, report)
    _check_context(manifest, defined, report)
    _check_hooks(manifest, defined, report)

    if not report.permissions and (manifest.get("tools") or manifest.get("mcp")):
        report.warnings.append(
            "this plugin declares tools or a server but no permissions; every "
            "contribution will be refused at load time")
    return report


def _top_level_names(tree: ast.AST) -> set[str]:
    """Names a module defines at the top level, without importing it."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _require_callable(name: str, defined: set[str], report: ValidationReport,
                      what: str) -> None:
    if name not in defined:
        report.errors.append(f"{what} names {name!r}, which the module does not define")


def _check_tools(manifest, defined, report: ValidationReport) -> None:
    for spec in manifest.get("tools", []) or []:
        if not isinstance(spec, dict):
            report.errors.append("each tool must be an object")
            continue
        name = str(spec.get("name", ""))
        if not name.isidentifier():
            report.errors.append(f"tool name {name!r} is not a valid identifier")
        if "tools" not in report.permissions:
            report.errors.append(f"tool {name!r} needs the 'tools' permission")
        if spec.get("risk", "write") not in {"read", "write", "exec", "net"}:
            report.errors.append(
                f"tool {name!r} has unknown risk {spec.get('risk')!r}; "
                f"use read, write, exec, or net")
        _require_callable(str(spec.get("handler", "")), defined, report,
                          f"tool {name!r}")


def _check_skills(manifest, defined, report: ValidationReport) -> None:
    for spec in manifest.get("skills", []) or []:
        if not isinstance(spec, dict):
            report.errors.append("each skill must be an object")
            continue
        if "skills" not in report.permissions:
            report.errors.append(f"skill {spec.get('name')!r} needs the 'skills' permission")
        _require_callable(str(spec.get("factory", "")), defined, report,
                          f"skill {spec.get('name')!r}")


def _check_settings(manifest, report: ValidationReport) -> None:
    for spec in manifest.get("settings", []) or []:
        if not isinstance(spec, dict):
            report.errors.append("each setting must be an object")
            continue
        if "settings" not in report.permissions:
            report.errors.append(f"setting {spec.get('key')!r} needs the 'settings' permission")
        kind = spec.get("kind", "cycle")
        if kind not in {"cycle", "command"}:
            report.errors.append(f"setting {spec.get('key')!r} has unknown kind {kind!r}")
        if kind == "cycle" and not spec.get("choices"):
            report.errors.append(f"cycling setting {spec.get('key')!r} has no choices")


def _check_commands(manifest, report: ValidationReport) -> None:
    for name, spec in (manifest.get("commands") or {}).items():
        if not isinstance(spec, (str, dict)):
            report.errors.append(f"command {name!r} must be a string or an object")
            continue
        if isinstance(spec, dict):
            tier = spec.get("tier", "")
            if tier and tier not in {"fast", "standard", "reasoning"}:
                report.errors.append(
                    f"command {name!r} has unknown tier {tier!r}; "
                    f"use fast, standard, or reasoning")


def _check_mcp(manifest, report: ValidationReport) -> None:
    for spec in manifest.get("mcp", []) or []:
        if not isinstance(spec, dict):
            report.errors.append("each mcp server must be an object")
            continue
        command = spec.get("command")
        if not isinstance(command, list) or not command or not all(
                isinstance(part, str) for part in command):
            report.errors.append(
                f"mcp server {spec.get('name')!r} needs a command as a non-empty list of strings")
        for permission in ("subprocess", "net"):
            if permission not in report.permissions:
                report.errors.append(
                    f"mcp server {spec.get('name')!r} needs the {permission!r} permission")
        risk = spec.get("risk")
        if risk is not None and risk != "net":
            report.warnings.append(
                f"mcp server {spec.get('name')!r} declares risk {risk!r}, but a remote "
                f"tool is always net; the declaration is ignored")


def _check_subagents(manifest, report: ValidationReport) -> None:
    for spec in manifest.get("subagents", []) or []:
        if not isinstance(spec, dict):
            report.errors.append("each subagent must be an object")
            continue
        name = str(spec.get("name", ""))
        if not name.replace("-", "_").isidentifier():
            report.errors.append(f"subagent name {name!r} is not a valid identifier")
        if not str(spec.get("system_prompt", "")).strip():
            report.errors.append(f"subagent {name!r} needs a system_prompt")
        if not spec.get("tools"):
            report.errors.append(f"subagent {name!r} must allow at least one tool")
        if "subagents" not in report.permissions:
            report.errors.append(f"subagent {name!r} needs the 'subagents' permission")


def _check_context(manifest, defined, report: ValidationReport) -> None:
    for spec in manifest.get("context", []) or []:
        if not isinstance(spec, dict):
            report.errors.append("each context fragment must be an object")
            continue
        source = str(spec.get("source", ""))
        if not source:
            report.errors.append("a context fragment needs a source")
        if "context" not in report.permissions:
            report.errors.append(f"context fragment {source!r} needs the 'context' permission")
        if "content" in spec:
            try:
                Trigger.parse(spec.get("trigger", "always"))
            except ValueError as exc:
                report.errors.append(f"context fragment {source!r}: {exc}")
            if int(spec.get("priority", 50)) not in VALID_PRIORITIES:
                report.errors.append(
                    f"context fragment {source!r} has unknown priority "
                    f"{spec.get('priority')}; valid: {sorted(VALID_PRIORITIES)}")
        elif spec.get("factory"):
            _require_callable(str(spec["factory"]), defined, report,
                              f"context fragment {source!r}")
        else:
            report.errors.append(
                f"context fragment {source!r} needs either 'content' or a 'factory'")


def _check_hooks(manifest, defined, report: ValidationReport) -> None:
    phases = {"pre_tool", "post_tool", "on_final", "on_event"}
    for spec in manifest.get("hooks", []) or []:
        if not isinstance(spec, dict):
            report.errors.append("each hook must be an object")
            continue
        when = str(spec.get("when", "")).lower()
        if when not in phases:
            report.errors.append(f"hook phase {when!r} is not one of {sorted(phases)}")
        if "hooks" not in report.permissions:
            report.errors.append(f"a {when!r} hook needs the 'hooks' permission")
        _require_callable(str(spec.get("handler", "")), defined, report,
                          f"{when!r} hook")


# --- scaffolding ------------------------------------------------------------

_SCAFFOLD_MODULE = '''"""{name}: {description}

Write the tool's behaviour here. Every tool is a plain function whose keyword
arguments are the tool's parameters, returning either a dict with success/output
or a string.

Two rules worth knowing before you write one:

- A tool's `risk` decides whether the strict safety profile asks the user first.
  Choose it honestly: a tool that changes the workspace is `write`, one that runs
  something is `exec`.
- Paths are resolved against the workspace before your function sees them, and
  anything escaping the workspace is refused before this code runs. That is a
  convenience, not a sandbox: a plugin with the `subprocess` permission can do
  anything a Python program can.
"""

from pathlib import Path


def {tool_name}(path="."):
    """{tool_description}

    Replace this with the real behaviour.
    """
    target = Path(path)
    if not target.exists():
        return {{"success": False, "error": f"No such path: {{path}}"}}
    return {{"success": True, "output": f"{{target}} exists"}}
'''


def scaffold(directory: Path | str, *, name: str, description: str = "",
             tool_name: str | None = None) -> Path:
    """Create a plugin that passes ``validate`` with zero edits.

    This is the test of the scaffolder: if the generated plugin needed a fix
    before it would validate, the tool is not usable by someone who has not
    read the manifest schema.
    """
    root = Path(directory) / name
    root.mkdir(parents=True, exist_ok=True)
    tool = tool_name or _default_tool_name(name)

    manifest = {
        "name": name,
        "version": "0.1.0",
        "description": description or f"The {name} plugin.",
        "author": "",
        "permissions": list(SCAFFOLD_PERMISSIONS),
        "tools": [{
            "name": tool,
            "description": f"{description or name}.",
            "handler": tool,
            "risk": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string",
                                        "description": "What to look at."}},
            },
        }],
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / MODULE_NAME).write_text(_SCAFFOLD_MODULE.format(
        name=name, description=description or name,
        tool_name=tool, tool_description=f"Check a path. Replace this."), encoding="utf-8")
    (root / "README.md").write_text(_scaffold_readme(name, description, tool), encoding="utf-8")
    (root / "test_plugin.py").write_text(_scaffold_test(name, tool), encoding="utf-8")
    return root


def _default_tool_name(name: str) -> str:
    cleaned = name.replace("-", "_")
    return cleaned if cleaned.isidentifier() else f"{cleaned}_tool"


def _scaffold_readme(name: str, description: str, tool: str) -> str:
    return f"""# {name}

{description or 'A plugin for the Adaptive Agent Harness.'}

## Install

Copy this directory to `~/.config/adaptive-harness/plugins/{name}/`.

```bash
adaptive-harness plugins            # is it discovered?
adaptive-harness plugin validate {name}   # does it pass without running?
```

## What it provides

- `{tool}` — see the description in `plugin.plugin.json`.

## Permissions it asks for

`tools` — so it can register `{tool}`. Nothing else. If you add something that
reaches the network, runs a program, or reads files outside the workspace, add
that permission explicitly and know that you are trusting yourself.
"""


def _scaffold_test(name: str, tool: str) -> str:
    return f'''"""A test for this plugin. Run it with `pytest {name}/test_plugin.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plugin import {tool}  # noqa: E402


def test_the_tool_reports_a_missing_path():
    result = {tool}(path="definitely-not-here")
    assert result["success"] is False


def test_the_tool_accepts_a_real_path(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    result = {tool}(path=str(target))
    assert result["success"] is True
'''


# --- packaging --------------------------------------------------------------

def pack(directory: Path | str, destination: Path | str = "dist") -> Path:
    """Bundle a plugin into a distributable tarball, after validating it.

    Packing validates first: shipping something that cannot load is worse than
    refusing to ship it.
    """
    source = Path(directory)
    report = validate(source)
    if not report.ok:
        raise ValueError("refusing to pack an invalid plugin:\n" + report.render())
    archive = Path(destination) / f"{report.name}-{report.version}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        for item in sorted(source.iterdir()):
            if item.name in {"__pycache__", ".pytest_cache"} or item.suffix == ".pyc":
                continue
            bundle.add(item, arcname=f"{report.name}/{item.name}")
    return archive
