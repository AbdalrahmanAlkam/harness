"""Plugin discovery and registration.

A plugin adds a capability to the harness without editing the harness. It can
contribute a tool the model may call, a skill, a setting that persists and
appears on the settings screen, a prompt override, and a command. The core
consults :class:`PluginHost` for each of those, so adding a capability never
requires touching a file in ``src/adaptive_harness``.

The design constraint throughout is that a plugin is **untrusted code**. It is
declared in a manifest that states what it wants to be allowed to do, the user
grants exactly that, and anything not granted is refused. A plugin that
misbehaves is disabled rather than allowed to take the harness down with it.

Discovery roots, in increasing precedence:

1. the package's own ``plugins/`` directory (bundled capabilities);
2. ``~/.config/adaptive-harness/plugins/`` (the user's own);
3. ``<project>/.harness/plugins/`` (a repository's).

Root 3 is **opt-in and off by default**. A repository you have merely cloned
should not be able to run code, and this one is a coding agent that a user points
at repositories they did not write. Project plugins are therefore loaded only
when the user has said so -- ``--with-project-plugins`` on the CLI, or
``allow_project_plugins`` in the config -- and the refusal is reported rather
than being silent, so a user who installed one knows why it is not loading.

Roots 1 and 2 need no such gate: the first ships in the package the user
installed, and the second is code the user put there themselves.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from adaptive_harness.data.config import DEFAULT_CONFIG_DIR

#: Manifest keys we understand. An unknown key is an error rather than a
#: silent no-op: a typo in a permission field must fail loudly, because the
#: permission list is what the user reviews before trusting a plugin.
MANIFEST_KEYS = {
    "name", "version", "description", "author", "homepage",
    "min_harness", "entry_point", "permissions", "tools", "skills", "settings",
    "prompts", "commands",
}

#: What a plugin may ask for. Each is opt-in; an empty list means the plugin
#: can only add declarative metadata and no Python runs at all.
PERMISSIONS = {
    "tools",       # register a callable tool
    "skills",      # register a skill
    "settings",    # contribute a persistent setting
    "prompts",     # override a prompt
    "commands",    # add a slash command
    "hooks",       # observe the agent event stream
    "net",         # make network requests
    "subprocess",  # execute external programs
    "fs_home",     # read or write outside the workspace
    "env",         # read the process environment
}

#: Names a plugin may not take. Shadowing a built-in would let it replace the
#: file tools or the compiler and then answer for them.
RESERVED_TOOL_NAMES = {
    "run_bash", "read_file", "write_file", "edit_file", "list_directory",
    "search_files", "run_pytest", "ask_user", "run_python_repl",
    "verify_equation", "compile_typst", "read_full_output", "calculate",
    "check_convergence", "web_search", "run_lean_proof", "delegate_subagent",
}

#: Tools that only observe. The strict safety profile asks before anything else,
#: and a plugin tool declaring one of these is the only kind that skips that.
READ_ONLY_TOOLS = {
    "read_file", "list_directory", "search_files", "read_full_output",
    "calculate", "verify_equation", "check_convergence",
}


@dataclass
class PluginTool:
    """A tool a plugin contributes."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    #: read | write | exec | net. Drives the strict-profile approval gate, so
    #: a plugin cannot opt itself out of being reviewed by calling itself
    #: something other than run_bash.
    risk: str = "write"
    domains: tuple[str, ...] = ()


@dataclass
class PluginSetting:
    """A persistent setting a plugin contributes to the settings screen."""

    key: str
    label: str
    description: str
    kind: str = "cycle"
    choices: tuple[str, ...] = ()
    default: str = ""


@dataclass
class LoadedPlugin:
    """One discovered plugin, whether or not it loaded successfully."""

    name: str
    version: str
    directory: Path
    description: str = ""
    author: str = ""
    permissions: frozenset[str] = frozenset()
    tools: List[PluginTool] = field(default_factory=list)
    skills: List[Any] = field(default_factory=list)
    settings: List[PluginSetting] = field(default_factory=list)
    prompts: Dict[str, str] = field(default_factory=dict)
    commands: Dict[str, str] = field(default_factory=dict)
    #: Set when the plugin failed to load. A broken plugin is reported and
    #: skipped, never allowed to abort startup.
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class PluginHost:
    """Discovers plugins and answers what the core asks it.

    The core consults this object for tools, skills, settings, prompts and
    commands. It is deliberately boring: discovery is filesystem scanning, a
    manifest is a dict, and a failure in one plugin is recorded and skipped.
    """

    def __init__(self, *, project_root: Path | str | None = None,
                 allow_project_plugins: bool = False) -> None:
        self.project_root = Path(project_root or Path.cwd())
        #: Project-supplied plugins execute code the user did not write, so they
        #: are off unless the user opted in. A cloned repository must not be able
        #: to run anything just by being pointed at.
        self.allow_project_plugins = allow_project_plugins
        self.plugins: List[LoadedPlugin] = []
        #: Called with each event the agent emits, for plugins with `hooks`.
        self.hook_listeners: List[Callable[[Any], None]] = []
        self.load_errors: List[str] = []
        #: Set when project plugins were found but not loaded, so the reason can
        #: be shown rather than leaving the user to wonder.
        self.project_plugins_skipped = 0

    # -- discovery ---------------------------------------------------------

    def discovery_roots(self) -> List[Path]:
        roots = [Path(__file__).resolve().parent.parent / "plugins" / "bundled",
                 Path(DEFAULT_CONFIG_DIR) / "plugins"]
        project_root = self.project_root / ".harness" / "plugins"
        if self.allow_project_plugins:
            roots.append(project_root)
        elif project_root.is_dir():
            self.project_plugins_skipped = sum(
                1 for path in project_root.iterdir()
                if path.is_dir() and list(path.glob("*.plugin.json")))
        return [root for root in roots if root.is_dir()]

    def discover(self) -> List[LoadedPlugin]:
        """Load every plugin found, never raising.

        A plugin is a directory containing a manifest and optionally a Python
        file, so discovery walks one level into each root. A plugin that fails
        to parse or import is recorded with its error and skipped. One bad
        plugin must not stop the harness from starting, which is the whole
        reason plugins are not part of the core.
        """
        found: List[LoadedPlugin] = []
        for root in self.discovery_roots():
            for manifest_path in self._manifests(root):
                plugin = self._load(manifest_path, root)
                if plugin is not None:
                    found.append(plugin)
        # Later roots win on a name collision, so a project can override a
        # plugin the user installed globally.
        by_name: Dict[str, LoadedPlugin] = {}
        for plugin in found:
            by_name[plugin.name] = plugin
        self.plugins = list(by_name.values())
        return self.plugins

    @staticmethod
    def _manifests(root: Path) -> List[Path]:
        """Manifests under one root: top-level files and one directory deep."""
        manifests = sorted(root.glob("*.plugin.json"))
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if child.is_dir():
                manifests.extend(sorted(child.glob("*.plugin.json")))
        return manifests

    def _load(self, manifest_path: Path, root: Path) -> Optional[LoadedPlugin]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.load_errors.append(f"{manifest_path}: {exc}")
            return None
        directory = manifest_path.parent
        try:
            unknown = set(manifest) - MANIFEST_KEYS
            if unknown:
                raise ValueError(
                    f"unknown manifest keys: {', '.join(sorted(unknown))}. "
                    f"Known keys: {', '.join(sorted(MANIFEST_KEYS))}")
            name = manifest["name"]
            plugin = LoadedPlugin(
                name=name,
                version=str(manifest.get("version", "0")),
                directory=directory,
                description=manifest.get("description", ""),
                author=manifest.get("author", ""),
                permissions=frozenset(manifest.get("permissions", ())),
            )
            requested = set(plugin.permissions)
            unknown_permissions = requested - PERMISSIONS
            if unknown_permissions:
                raise ValueError(
                    f"unknown permissions: {', '.join(sorted(unknown_permissions))}. "
                    f"A plugin may not claim a permission that does not exist.")
            entry_point = manifest.get("entry_point")
            module = None
            # Any tool, skill or hook needs the plugin's Python, so the module is
            # imported whenever the manifest asks for one of those. A
            # metadata-only plugin -- settings, prompts and commands alone --
            # imports nothing, so listing one can never execute code.
            if entry_point or manifest.get("tools") or manifest.get("skills") or \
                    "hooks" in plugin.permissions:
                module = self._import_plugin_module(directory, manifest_path)
                if module is None:
                    raise ValueError("the plugin's Python file could not be imported")
            self._collect(plugin, manifest, module)
            return plugin
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not break startup
            detail = f"{type(exc).__name__}: {exc}"
            self.load_errors.append(f"{manifest_path.name}: {detail}")
            failed = LoadedPlugin(name=manifest.get("name", manifest_path.stem),
                                  version=str(manifest.get("version", "0")),
                                  directory=directory, error=detail)
            return failed

    @staticmethod
    def _import_plugin_module(directory: Path, manifest_path: Path):
        """Import a plugin's Python file under a unique module name."""
        module_name = f"_adaptive_plugin_{manifest_path.parent.name}_{manifest_path.stem}"
        candidates = sorted(directory.glob("*.py"))
        if not candidates:
            return None
        spec = importlib.util.spec_from_file_location(module_name, candidates[0])
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _collect(self, plugin: LoadedPlugin, manifest: Dict[str, Any], module) -> None:
        """Turn a manifest into the things the core consumes."""
        for spec in manifest.get("tools", ()):
            name = spec["name"]
            if name in RESERVED_TOOL_NAMES:
                raise ValueError(f"{name!r} is a built-in tool and cannot be replaced")
            if not name.isidentifier():
                raise ValueError(f"tool name {name!r} is not a valid identifier")
            if "tools" not in plugin.permissions:
                raise ValueError(f"{plugin.name} declares tool {name!r} without the "
                                 f"'tools' permission")
            risk = spec.get("risk", "write")
            if risk not in {"read", "write", "exec", "net"}:
                raise ValueError(f"tool {name!r} has unknown risk {risk!r}")
            handler = getattr(module, spec["handler"], None) if module else None
            if handler is None:
                # A declarative tool with no handler is a manifest error, not a
                # silently inert tool -- an inert tool is the exact failure mode
                # this project keeps having to fix elsewhere.
                raise ValueError(f"tool {name!r} has no callable handler "
                                 f"(module has no attribute {spec.get('handler')!r})")
            plugin.tools.append(PluginTool(
                name=name,
                description=spec.get("description", ""),
                parameters=spec.get("parameters", {"type": "object", "properties": {}}),
                handler=handler,
                risk=risk,
                domains=tuple(spec.get("domains", ())),
            ))

        for spec in manifest.get("skills", ()):
            if "skills" not in plugin.permissions:
                raise ValueError(f"{plugin.name} declares a skill without the 'skills' permission")
            factory = getattr(module, spec["factory"], None) if module else None
            if factory is None:
                raise ValueError(f"skill {spec['name']!r} has no factory "
                                 f"(module has no attribute {spec.get('factory')!r})")
            # `source` marks it as plugin-provided everywhere it is displayed.
            plugin.skills.append(factory(source=f"plugin:{plugin.name}"))

        for spec in manifest.get("settings", ()):
            if "settings" not in plugin.permissions:
                raise ValueError(f"{plugin.name} declares a setting without the "
                                 f"'settings' permission")
            plugin.settings.append(PluginSetting(
                key=spec["key"],
                label=spec.get("label", spec["key"]),
                description=spec.get("description", ""),
                kind=spec.get("kind", "cycle"),
                choices=tuple(str(choice) for choice in spec.get("choices", ())),
                default=str(spec.get("default", "")),
            ))

        if "prompts" in plugin.permissions:
            plugin.prompts = {str(key): str(value)
                              for key, value in (manifest.get("prompts") or {}).items()}

        if "commands" in plugin.permissions:
            plugin.commands = {str(key): str(value)
                               for key, value in (manifest.get("commands") or {}).items()}

        if "hooks" in plugin.permissions and module is not None:
            listener = getattr(module, "on_agent_event", None)
            if callable(listener):
                self.hook_listeners.append(listener)

    # -- what the core asks ------------------------------------------------

    def build_tools(self, workspace_root: Path | str) -> List[Any]:
        """Instantiate every plugin tool, wrapped so the host sees the call.

        Each tool is confined the way the built-in file tools are: the wrapper
        resolves its path arguments against the workspace and refuses anything
        that escapes, so a plugin cannot reach outside the project by accident.
        """
        from adaptive_harness.plugins.tool import PluginToolAdapter

        tools: List[Any] = []
        for plugin in self.plugins:
            if not plugin.ok:
                continue
            for spec in plugin.tools:
                tools.append(PluginToolAdapter(spec, plugin, workspace_root))
        return tools

    def skills(self) -> Dict[str, Any]:
        """Plugin skills by name, for the skill router to consider."""
        collected: Dict[str, Any] = {}
        for plugin in self.plugins:
            if not plugin.ok:
                continue
            for skill in plugin.skills:
                collected[skill.name] = skill
        return collected

    def settings(self) -> List[PluginSetting]:
        collected: List[PluginSetting] = []
        for plugin in self.plugins:
            if not plugin.ok:
                continue
            collected.extend(plugin.settings)
        return collected

    def prompt_overrides(self) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for plugin in self.plugins:
            if not plugin.ok:
                continue
            merged.update(plugin.prompts)
        return merged

    def commands(self) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for plugin in self.plugins:
            if not plugin.ok:
                continue
            merged.update(plugin.commands)
        return merged

    def is_read_only(self, tool_name: str) -> bool:
        """Whether a tool only observes, for the strict-profile gate.

        Built-ins answer from the known set. A plugin tool answers from the risk
        it *declared*, and defaults to not-read-only, so a plugin cannot reach
        execution by calling itself something other than run_bash.
        """
        if tool_name in RESERVED_TOOL_NAMES:
            return tool_name in READ_ONLY_TOOLS
        for plugin in self.plugins:
            for spec in plugin.tools:
                if spec.name == tool_name:
                    return spec.risk == "read"
        return False

    def has_permission(self, plugin_name: str, permission: str) -> bool:
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                return permission in plugin.permissions
        return False

    def emit(self, event: Any) -> None:
        """Publish an agent event to hook listeners. Never raises.

        A plugin's hook is the one place where third-party code runs on every
        step, so a failure there is swallowed and reported once rather than
        being allowed to break the run.
        """
        for listener in self.hook_listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - a hook must not break the agent
                self.load_errors.append(
                    f"hook {getattr(listener, '__module__', '?')} raised on "
                    f"{getattr(event, 'event_type', '?')}:\n{traceback.format_exc(limit=3)}")
                self.hook_listeners.remove(listener)

    def describe(self) -> str:
        """A one-line-per-plugin summary for the CLI and the TUI."""
        lines: list[str] = []
        for plugin in sorted(self.plugins, key=lambda item: item.name):
            if plugin.ok:
                bits = []
                if plugin.tools:
                    bits.append(f"{len(plugin.tools)} tool(s)")
                if plugin.skills:
                    bits.append(f"{len(plugin.skills)} skill(s)")
                if plugin.settings:
                    bits.append(f"{len(plugin.settings)} setting(s)")
                if plugin.commands:
                    bits.append(f"{len(plugin.commands)} command(s)")
                granted = ", ".join(sorted(plugin.permissions)) or "no permissions"
                lines.append(f"{plugin.name} {plugin.version} — "
                             f"{plugin.description or 'no description'} "
                             f"[{granted}] {', '.join(bits)}")
            else:
                lines.append(f"{plugin.name} — FAILED TO LOAD: {plugin.error}")
        if self.project_plugins_skipped:
            lines.append(
                f"({self.project_plugins_skipped} plugin(s) in this project are not loaded. "
                f"Project plugins run code from the repository, so they need "
                f"--with-project-plugins or `allow_project_plugins` in the config.)")
        return "\n".join(lines) or "No plugins installed."
