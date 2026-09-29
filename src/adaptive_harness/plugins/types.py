"""Shared types for the plugin contract.

These are the vocabulary every contribution type speaks. They live here rather
than in ``host.py`` so a plugin can import them without importing the loader,
and so the core can depend on the types without depending on discovery.

The governing law of this project is that anything not everyone needs is a
plugin. These types are the surface that makes that possible: they define what
a plugin may *say*, never what the core must therefore do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# --- triggers --------------------------------------------------------------

#: A trigger decides *when* a context fragment should be considered for the
#: next request. It is deliberately cheap: a regex, a literal, or a name the
#: core hands to an existing local classifier. Nothing here may spend a
#: foundation-model token -- that is sacred invariant 1.
#
#: Serialized form:  ``"always"`` | ``"regex:<pattern>"`` | ``"keyword:<word>"``
#:                   | ``"command:/name"`` | ``"on_tool:<tool>"``
#:                   | ``"classifier:<label>"``
@dataclass(frozen=True)
class Trigger:
    kind: str  # always | regex | keyword | command | on_tool | classifier
    value: str = ""

    def matches(self, *, text: str = "", tools_used: tuple[str, ...] = (),
                commands: tuple[str, ...] = ()) -> bool:
        """Cheap first-stage matching. Never calls a model.

        Returning True means "worth scoring", not "include it". A fragment still
        has to survive the relevance classifier before it reaches the message
        list.
        """
        if self.kind == "always":
            return True
        if self.kind == "regex":
            try:
                return bool(re.search(self.value, text, re.I))
            except re.error:
                # A malformed pattern must not silently admit or exclude content;
                # failing to match is the safe direction, and the fragment's
                # own validation rejects it at load time anyway.
                return False
        if self.kind == "keyword":
            return self.value.casefold() in text.casefold()
        if self.kind == "command":
            return any(command.startswith(self.value) for command in commands)
        if self.kind == "on_tool":
            return self.value in tools_used
        if self.kind == "classifier":
            # The core supplies the label set; a fragment keyed to a label the
            # run never produced simply does not fire.
            return self.value in text.split()
        return False

    @classmethod
    def parse(cls, raw: str | "Trigger") -> "Trigger":
        if isinstance(raw, Trigger):
            return raw
        text = str(raw)
        for kind in ("regex", "keyword", "command", "on_tool", "classifier"):
            prefix = f"{kind}:"
            if text.startswith(prefix):
                return cls(kind, text[len(prefix):])
        if text in {"always", "*", ""}:
            return cls("always")
        raise ValueError(
            f"Unknown trigger {text!r}. Use always, regex:<pattern>, keyword:<word>, "
            f"command:/name, on_tool:<tool>, or classifier:<label>.")

    def __str__(self) -> str:
        return self.kind if not self.value else f"{self.kind}:{self.value}"


# --- context fragments -----------------------------------------------------

#: Eviction order. Lower survives longer. These are the *only* values the
#: allocator understands, so a plugin cannot smuggle in a priority that quietly
#: outranks a pinned fragment.
PRIORITY_PINNED = 0        # never evicted, never compacted
PRIORITY_REQUIREMENT = 10  # the current turn's obligations
PRIORITY_PROJECT = 20      # project instructions, conventions
PRIORITY_SKILL = 30        # active skill text
PRIORITY_MEMORY = 40       # remembered decisions
PRIORITY_REFERENCE = 50    # file contents, docs
PRIORITY_OPPORTUNISTIC = 70  # nice to have, dropped first

VALID_PRIORITIES = {PRIORITY_PINNED, PRIORITY_REQUIREMENT, PRIORITY_PROJECT,
                    PRIORITY_SKILL, PRIORITY_MEMORY, PRIORITY_REFERENCE,
                    PRIORITY_OPPORTUNISTIC}


@dataclass(frozen=True)
class ContextFragment:
    """A piece of content a plugin proposes to place in the model's context.

    A fragment never enters the message list because a plugin said it might be
    useful. It enters because a *classifier* scored it relevant for this turn.
    The trigger is only the cheap first stage.

    ``tokens`` is the plugin's estimate; the core re-measures with the active
    estimator before admitting anything, so a plugin cannot understate its cost
    to buy its way into a crowded context.
    """

    source: str          # "skill:refactor", "mcp:github", "file:src/app.py"
    content: str
    tokens: int
    priority: int
    trigger: Trigger
    pinned: bool = False
    provenance: str = ""  # which plugin contributed it
    #: guaranteed fragments are admitted by priority until the budget is spent;
    #: opportunistic ones compete on classifier score alone.
    guaranteed: bool = True

    def __post_init__(self) -> None:
        if not self.source or not isinstance(self.source, str):
            raise ValueError("A context fragment needs a source.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError(f"Context fragment {self.source!r} has no content.")
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Unknown priority {self.priority}. Valid values: "
                f"{sorted(VALID_PRIORITIES)}")
        if self.pinned and self.priority != PRIORITY_PINNED:
            raise ValueError(
                "A pinned fragment must use PRIORITY_PINNED, so 'pinned' cannot "
                "disagree with the eviction order.")
        # A trigger may be written the way a manifest writes it, as a string.
        # Frozen, so normalise through object.__setattr__.
        if not isinstance(self.trigger, Trigger):
            object.__setattr__(self, "trigger", Trigger.parse(self.trigger))

    @property
    def effective_priority(self) -> int:
        return PRIORITY_PINNED if self.pinned else self.priority


# --- hook verdicts ---------------------------------------------------------

@dataclass(frozen=True)
class HookVerdict:
    """What a ``PRE_TOOL`` hook decided about a call.

    ``allow``, ``deny``, or ``rewrite``. A denial is surfaced to the model as a
    tool error rather than swallowed, so a hook that blocks something leaves a
    trace the user and the model can both see.
    """

    action: str  # allow | deny | rewrite
    reason: str = ""
    arguments: Optional[dict[str, Any]] = None
    plugin: str = ""

    def __post_init__(self) -> None:
        if self.action not in {"allow", "deny", "rewrite"}:
            raise ValueError("Hook action must be allow, deny, or rewrite.")
        if self.action == "deny" and not self.reason:
            # A silent denial is indistinguishable from a hang to the user.
            raise ValueError("A hook denial must say why.")
        if self.action == "rewrite" and not isinstance(self.arguments, dict):
            raise ValueError("A rewrite verdict must supply the new arguments.")


ALLOW = HookVerdict("allow")


# --- contribution records --------------------------------------------------

@dataclass
class McpServerSpec:
    """An MCP server a plugin asks the host to supervise."""

    name: str
    command: list[str]
    #: A remote tool is net-risk and cannot be downgraded by the plugin: the
    #: data crosses a process boundary to code the user did not write.
    risk: str = "net"
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    startup_timeout_s: float = 20.0


@dataclass
class SubagentSpec:
    """A specialist agent a plugin contributes.

    Surfaces as a *mode* of the existing delegation tool, never as a new tool,
    so the agent loop needs no branch to know about it.
    """

    name: str
    system_prompt: str
    tools: tuple[str, ...]
    budget_tokens: int = 16000
    description: str = ""


@dataclass
class CommandSpec:
    """A slash command a plugin contributes.

    The model/tier override is the point: an operator can pin a cheap model for
    a mechanical command and an expensive one for a judgement call without
    editing config or the plugin.
    """

    name: str
    description: str
    model: str = ""
    tier: str = ""
    argument_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookSet:
    """The two-phase hook functions a plugin exposes.

    Each is optional. A plugin that only wants to observe supplies
    ``on_agent_event``, which is the original fire-and-forget listener and
    cannot affect the run.
    """

    pre_tool: Optional[Callable[..., Any]] = None
    post_tool: Optional[Callable[..., Any]] = None
    on_final: Optional[Callable[..., Any]] = None
    on_event: Optional[Callable[[Any], None]] = None
