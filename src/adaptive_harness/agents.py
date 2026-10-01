"""User-defined subagents.

Today the only way to spawn a subagent is ``delegate_subagent`` with a role
picked from a hardcoded enum of four. That makes the agent surface a fixed
property of the binary: you cannot add a "database reviewer" or a "perf auditor"
without editing the source, and you cannot give a subagent its own model,
permissions, or sandbox because there is nowhere to say so.

So a subagent becomes a *file*. A directory of markdown, with a small
frontmatter block, is a subagent you can write, share, and version:

    .harness/agents/db-reviewer.md

        ---
        name: db-reviewer
        description: Reviews a migration for safety and reversibility.
        tools: read_file, grep, run_bash
        model: fast
        permissionMode: plan
        maxTurns: 12
        isolation: worktree
        ---

        You review database migrations...

The frontmatter keys deliberately match the ones Claude Code documents, so an
agent definition written for one works here. That is not cosmetic: it is what
lets a team move a set of agents between the two without rewriting them.

Only a handful of keys change behaviour; the rest are carried so a definition
does not lose information when it crosses tools. An unknown key is a warning
rather than a failure, because a file that a newer version of another tool wrote
should still run here.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from adaptive_harness.data.config import DEFAULT_CONFIG_DIR

#: ``.harness/agents/*.md`` in a project; ``~/.config/adaptive-harness/agents/``
#: for the user's own.
AGENT_SUFFIX = ".md"
PROJECT_DIR = Path(".harness") / "agents"

#: Keys that change behaviour here. Everything else is recorded and passed
#: through so a definition written for another tool is not silently truncated.
KNOWN_KEYS = {
    "name", "description", "tools", "disallowedTools", "model", "permissionMode",
    "maxTurns", "effort", "isolation", "background", "color", "skills",
    "memory", "omitClaudeMd", "mcpServers", "hooks", "initialPrompt",
}

#: Keys this harness acts on. The rest are carried, not enforced.
ACTED_ON = {"name", "description", "tools", "disallowedTools", "model",
            "permissionMode", "maxTurns", "effort", "isolation", "background"}

_LIST_KEYS = {"tools", "disallowedTools", "skills", "mcpServers"}
_TRUE = {"true", "yes", "on", "1"}


class AgentDefinitionError(Exception):
    """A definition file could not be read. The message names the file."""


@dataclass(frozen=True)
class AgentDefinition:
    """One subagent, as declared by a file."""

    name: str
    description: str
    system_prompt: str = ""
    tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    #: A tier (fast/standard/reasoning) or a model id. A tier is resolved by the
    #: caller; a literal id is used as-is.
    model: str = ""
    #: inherited | plan | acceptEdits | bypass -- the permission mode the
    #: subagent runs under.
    permission_mode: str = ""
    max_turns: int = 0
    effort: str = ""
    #: "" or "worktree" -- run the agent against its own copy of the repository.
    isolation: str = ""
    background: bool = False
    color: str = ""
    memory: str = ""
    source: Path = field(default=Path("."), compare=False)
    #: Frontmatter keys this version does not act on, kept for pass-through.
    extra: Dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_write_capable(self) -> bool:
        """Whether this agent may change anything, derived from its permissions.

        A subagent that cannot ask before writing and is not restricted to
        reading should not be handed the write tools at all.
        """
        if self.permission_mode in {"plan"}:
            return False
        return bool(self.tools) and self.tools != self.disallowed_tools

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "tools": list(self.tools), "disallowedTools": list(self.disallowed_tools),
            "model": self.model, "permissionMode": self.permission_mode,
            "maxTurns": self.max_turns, "effort": self.effort,
            "isolation": self.isolation, "background": self.background,
            "color": self.color, "memory": self.memory,
            "source": str(self.source),
        }

    def line(self) -> str:
        flags = []
        if self.background:
            flags.append("background")
        if self.isolation:
            flags.append(self.isolation)
        if self.model:
            flags.append(self.model)
        if self.permission_mode:
            flags.append(self.permission_mode)
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        return f"  {self.name:<24} {self.description[:56]}{suffix}"


def parse_frontmatter(text: str, source: Path | str = ".") -> tuple[Dict[str, Any], str]:
    """Split a definition file into frontmatter and body.

    A deliberately small parser rather than a YAML dependency: the core has no
    third-party dependencies by design, and the frontmatter here is a flat block
    of scalars and comma lists. Anything more exotic belongs in the body.
    """
    source = Path(source)
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not match:
        raise AgentDefinitionError(
            f"{source} has no frontmatter. A subagent definition starts with a "
            f"'---' line, then keys, then '---' again.")
    block, body = match.group(1), match.group(2)
    fields: Dict[str, Any] = {}
    pending_key: Optional[str] = None
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # A list may be continued on following indented lines.
        if line.startswith("- ") and pending_key:
            # The placeholder is a string, not a list, so it has to be replaced
            # rather than appended to -- otherwise every item was dropped and
            # the field stayed empty.
            if not isinstance(fields.get(pending_key), list):
                fields[pending_key] = []
            fields[pending_key].append(line[2:].strip().strip("'\""))
            continue
        if ":" not in line:
            raise AgentDefinitionError(f"{source}: cannot read the line {line!r} in the frontmatter.")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            fields[key] = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            pending_key = None
        elif not value:
            # Either a multi-line list follows, or the value is empty.
            fields[key] = ""
            pending_key = key
        elif "," in value and key in _LIST_KEYS:
            # `tools: read_file, grep` is the natural way to write it, so it is
            # accepted directly rather than needing brackets. Without this the
            # value stayed a string and the next key was swallowed as if it were
            # a continuation of the list.
            fields[key] = [item.strip().strip("'\"") for item in value.split(",") if item.strip()]
            pending_key = None
        elif value.lower() in _TRUE:
            fields[key] = True
        elif value.lower() in {"false", "no", "off", "0"}:
            fields[key] = False
        elif value.isdigit():
            fields[key] = int(value)
        else:
            fields[key] = value.strip("'\"")
            pending_key = None
    return fields, body.strip()


def load_definition(path: Path) -> AgentDefinition:
    """Read one definition file, or explain why it cannot be used."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentDefinitionError(f"{path} could not be read: {exc}") from exc
    fields, body = parse_frontmatter(text, path)

    name = str(fields.get("name") or path.stem).strip()
    if not name:
        raise AgentDefinitionError(f"{path} has no name.")
    description = str(fields.get("description") or "").strip()
    if not description:
        raise AgentDefinitionError(
            f"{path} has no description. The model reads it to decide when to "
            f"spawn this agent, so an empty one is never chosen.")

    def as_tuple(key: str) -> tuple[str, ...]:
        value = fields.get(key)
        if isinstance(value, list):
            return tuple(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, str) and value.strip():
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return ()

    try:
        max_turns = int(fields.get("maxTurns") or 0)
    except (TypeError, ValueError):
        raise AgentDefinitionError(
            f"{path}: maxTurns must be a whole number.") from None

    isolation = str(fields.get("isolation") or "").strip()
    if isolation and isolation != "worktree":
        raise AgentDefinitionError(
            f"{path}: isolation {isolation!r} is not supported. The only kind is "
            f"'worktree'.")

    return AgentDefinition(
        name=name,
        description=description,
        system_prompt=body,
        tools=as_tuple("tools"),
        disallowed_tools=as_tuple("disallowedTools"),
        model=str(fields.get("model") or "").strip(),
        permission_mode=str(fields.get("permissionMode") or "").strip(),
        max_turns=max_turns,
        effort=str(fields.get("effort") or "").strip(),
        isolation=isolation,
        background=bool(fields.get("background")),
        color=str(fields.get("color") or "").strip(),
        memory=str(fields.get("memory") or "").strip(),
        source=path,
        extra={key: value for key, value in fields.items() if key not in KNOWN_KEYS},
    )


class AgentRegistry:
    """Every subagent a user has declared.

    Discovery order is user, then project, then the built-in roles -- so a
    project can shadow a user's agent deliberately, and anything a user writes
    wins over what ships.
    """

    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd())
        self._lock = threading.Lock()
        self._agents: Dict[str, AgentDefinition] = {}
        self.warnings: List[str] = []

    def roots(self) -> List[Path]:
        return [Path(DEFAULT_CONFIG_DIR) / "agents", self.workspace / PROJECT_DIR]

    def discover(self) -> Dict[str, AgentDefinition]:
        found: Dict[str, AgentDefinition] = {}
        for root in self.roots():
            if not root.is_dir():
                continue
            for path in sorted(root.glob(f"*{AGENT_SUFFIX}")):
                try:
                    definition = load_definition(path)
                except AgentDefinitionError as exc:
                    # One bad file must not hide the rest, and must not stop the
                    # harness starting.
                    self.warnings.append(str(exc))
                    continue
                found[definition.name] = definition
        with self._lock:
            self._agents = found
        return found

    def get(self, name: str) -> Optional[AgentDefinition]:
        with self._lock:
            return self._agents.get(name)

    def all(self) -> List[AgentDefinition]:
        with self._lock:
            return sorted(self._agents.values(), key=lambda item: item.name)

    def names(self) -> List[str]:
        return [definition.name for definition in self.all()]

    def describe(self) -> str:
        agents = self.all()
        if not agents:
            return ("No subagents defined.\n"
                    "Write one at .harness/agents/<name>.md, starting with a "
                    "'---' frontmatter block. `harness agent init <name>` scaffolds one.")
        lines = [f"Defined subagents ({len(agents)})"]
        for definition in agents:
            lines.append(definition.line())
            for key in sorted(definition.extra):
                lines.append(f"      note: {key} is not acted on by this version")
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._agents.clear()
        self.warnings.clear()


#: The roles that ship, so `delegate_subagent` still works with no files on
#: disk. They are definitions like any other -- built-in rather than declared.
BUILTIN_ROLES: Dict[str, Dict[str, Any]] = {
    "architect": {
        "description": "Plans an implementation before any code is written.",
        "tools": "read_file, list_directory, search_files, run_bash, run_pytest",
        "permissionMode": "plan",
    },
    "coder": {
        "description": "Implements a change and validates it.",
        "tools": "read_file, write_file, edit_file, list_directory, search_files, "
                 "run_bash, run_pytest",
    },
    "reviewer": {
        "description": "Independently verifies a change against its tests.",
        "tools": "read_file, list_directory, search_files, run_bash, run_pytest",
    },
    "security": {
        "description": "Reviews a change for injection, traversal and unsafe commands.",
        "tools": "read_file, list_directory, search_files, run_bash",
    },
}


def builtin_definitions() -> Dict[str, AgentDefinition]:
    """The shipped roles, as definitions."""
    built: Dict[str, AgentDefinition] = {}
    for name, spec in BUILTIN_ROLES.items():
        built[name] = AgentDefinition(
            name=name,
            description=spec["description"],
            system_prompt="",
            tools=tuple(item.strip() for item in spec["tools"].split(",") if item.strip()),
            permission_mode=spec.get("permissionMode", ""),
            source=Path("<built-in>"),
        )
    return built


def resolve(name: str, registry: Optional[AgentRegistry] = None) -> Optional[AgentDefinition]:
    """Find an agent by name: user-defined first, then the built-in roles."""
    if registry is not None:
        found = registry.get(name)
        if found is not None:
            return found
    return builtin_definitions().get(name)


def scaffold(directory: Path | str, name: str, description: str = "") -> Path:
    """Write a definition that validates, so the first edit is a real one."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}{AGENT_SUFFIX}"
    if path.exists():
        raise AgentDefinitionError(f"{path} already exists.")
    path.write_text(SCAFFOLD.format(
        name=name,
        description=description or f"What {name} does, and when to use it.",
    ), encoding="utf-8")
    return path


SCAFFOLD = """---
name: {name}
description: {description}
tools: read_file, list_directory, search_files
model: fast
permissionMode: plan
maxTurns: 12
---

You are a {name} agent.

Replace this paragraph with what the agent should actually do. Everything above
the line is read by the model to decide whether to spawn it, so make the
description say when it is the right choice and not otherwise.

Two keys worth knowing:

  tools           what this agent may call. A read-only agent that is handed
                   write_file will not stay read-only.
  permissionMode  `plan` means the agent may read and investigate but may not
                   change anything without asking.
"""
