"""The Context Plane.

> The main model's context must stay as close to empty as possible while it
> remains sufficient.

Everything context-bearing reaches the model's message list through this module.
Prompts, skill text, memory, tool output, file contents, MCP descriptions,
project instructions — all of it arrives as a `ContextFragment` and is admitted,
scored, and placed here. The rule this module exists to enforce is simple:

    no module may write into the message list except through context_plane.py

which is checked by a test that walks the source. The point is not tidiness. It
is that context is the only thing an operator cannot see the cost of, and if
content can enter by the back door, "how much am I paying for context" has no
answer.

Admission is two-stage, and the ordering is the design:

1. **Trigger matching** — regex, keyword, or a tool/command name. Microseconds,
   no tokens spent. A trigger says "worth scoring", never "include".
2. **A relevance classifier** — the local classifier machinery, no foundation
   model. A fragment enters because a classifier scored it relevant *for this
   turn*.

So a fragment never reaches the context because a plugin said it might be
useful. It reaches it because something local decided it was.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from adaptive_harness.plugins.types import (
    ContextFragment,
    PRIORITY_OPPORTUNISTIC,
    PRIORITY_PINNED,
    PRIORITY_REQUIREMENT,
    Trigger,
)

#: How much of the window is held back for the model's own final answer. A
#: request that fills the window completely has nowhere to put a summary.
RESERVED_FOR_SUMMARY_FRACTION = 0.10

#: The default share of the window the plane will fill before refusing more.
DEFAULT_FILL_FRACTION = 0.75


@dataclass
class BudgetReport:
    """What the allocator did this turn, and why.

    This is the object `/context` renders. An operator who cannot see where
    their context went cannot manage it, and an unmanageable context is a cost
    they stop paying attention to.
    """

    capacity: int
    reserved: int
    available: int
    admitted: List[str] = field(default_factory=list)
    rejected: List[tuple[str, str]] = field(default_factory=list)
    evicted: List[tuple[str, str]] = field(default_factory=list)
    used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "reserved": self.reserved,
            "available": self.available,
            "used": self.used,
            "admitted": list(self.admitted),
            "rejected": [{"source": source, "reason": reason}
                         for source, reason in self.rejected],
            "evicted": [{"source": source, "reason": reason}
                        for source, reason in self.evicted],
        }


class ContextPlane:
    """Holds the turn's context fragments and decides what the model sees.

    The plane is per-agent state, not a global: two agents in a swarm have
    different workspaces, different turns, and different budgets.
    """

    def __init__(self, capacity: int, *, fill: float = DEFAULT_FILL_FRACTION,
                 relevance: Optional[Callable[[ContextFragment, str], float]] = None,
                 estimator: Optional[Callable[[List[Dict[str, Any]]], int]] = None) -> None:
        self.capacity = int(capacity)
        self.fill = fill
        self.reserved = int(self.capacity * RESERVED_FOR_SUMMARY_FRACTION)
        self.available = max(1, int(self.capacity * fill) - self.reserved)
        #: Fragments contributed so far, keyed by source so a plugin that
        #: re-contributes the same source replaces rather than duplicates.
        self.fragments: Dict[str, ContextFragment] = {}
        #: Scores from the most recent relevance pass, for `/context`.
        self.last_scores: Dict[str, float] = {}
        self._relevance = relevance
        self._estimator = estimator
        self.last_report: Optional[BudgetReport] = None

    # -- contribution ------------------------------------------------------

    def contribute(self, fragment: ContextFragment) -> None:
        """Record a fragment. It is not in context until the next admission."""
        if not isinstance(fragment, ContextFragment):
            raise TypeError("contribute() takes a ContextFragment")
        self.fragments[fragment.source] = fragment

    def contribute_many(self, fragments: Iterable[ContextFragment]) -> None:
        for fragment in fragments:
            self.contribute(fragment)

    def forget(self, source: str) -> bool:
        return self.fragments.pop(source, None) is not None

    def pinned(self) -> List[ContextFragment]:
        return [fragment for fragment in self.fragments.values() if fragment.pinned]

    # -- admission ---------------------------------------------------------

    def select(self, *, text: str = "", tools_used: Sequence[str] = (),
               commands: Sequence[str] = ()) -> List[ContextFragment]:
        """Stage one: which fragments are worth scoring this turn.

        Purely local. No classifier runs here and no model is consulted; that is
        what makes it free.
        """
        matched: List[ContextFragment] = []
        for fragment in self.fragments.values():
            if fragment.pinned or fragment.trigger.matches(
                    text=text, tools_used=tuple(tools_used), commands=tuple(commands)):
                matched.append(fragment)
        return matched

    def score(self, fragments: Sequence[ContextFragment], text: str) -> Dict[str, float]:
        """Stage two: how relevant is each surviving fragment, right now.

        With no classifier configured, a fragment scores by whether it is
        guaranteed. That is the honest default: the plane then admits what a
        plugin marked as durable and defers what it marked opportunistic, rather
        than inventing a relevance signal it does not have.

        A classifier that raises is treated as absent rather than fatal. It is
        a local model that failed to load, or a plugin backend that crashed, and
        neither may take down a request.
        """
        if self._relevance is None:
            return {fragment.source: (1.0 if fragment.guaranteed else 0.4)
                    for fragment in fragments}
        scores: Dict[str, float] = {}
        for fragment in fragments:
            try:
                scores[fragment.source] = float(self._relevance(fragment, text))
            except Exception:  # noqa: BLE001 - a broken scorer falls back
                scores[fragment.source] = 1.0 if fragment.guaranteed else 0.4
        return scores

    def allocate(self, *, text: str = "", tools_used: Sequence[str] = (),
                 commands: Sequence[str] = (),
                 fixed_tokens: int = 0) -> BudgetReport:
        """Admit what fits, and report everything that did not, and why.

        Pinned fragments and the current turn's obligations are admitted first,
        whatever they cost. Then guaranteed fragments by priority. Then
        opportunistic ones by classifier score, best first, until full. Nothing
        is admitted without a reason and nothing is dropped without one either.
        """
        report = BudgetReport(capacity=self.capacity, reserved=self.reserved,
                              available=max(0, self.available - fixed_tokens))
        budget = report.available

        matched = self.select(text=text, tools_used=tools_used, commands=commands)
        if not matched:
            self.last_report = report
            return report

        scores = self.score(matched, text)
        self.last_scores = dict(scores)

        # Pinned and current-turn first, then guaranteed by priority, then
        # opportunistic by score. Pinned fragments are never dropped: a
        # fragment marked pinned and then evicted is a plugin whose contract
        # quietly does not hold.
        ordered = sorted(
            matched,
            key=lambda fragment: (
                0 if fragment.pinned else 1,
                fragment.effective_priority if fragment.guaranteed else 1000 - scores.get(fragment.source, 0.0),
                fragment.effective_priority,
                fragment.source,
            ),
        )

        admitted: List[ContextFragment] = []
        for fragment in ordered:
            cost = max(1, self._measure(fragment))
            if fragment.pinned or cost <= budget:
                if cost > budget and not fragment.pinned:
                    report.rejected.append((fragment.source, "would exceed the context budget"))
                    continue
                admitted.append(fragment)
                report.admitted.append(fragment.source)
                budget -= cost
                continue
            reason = ("opportunistic and the budget is full"
                      if not fragment.guaranteed
                      else f"needs {cost} tokens but only {budget} remain")
            report.rejected.append((fragment.source, reason))

        report.used = report.available - budget
        self.last_report = report
        return report

    def _measure(self, fragment: ContextFragment) -> int:
        """The real cost of a fragment, not the plugin's claim about it.

        A plugin that understates its own size would otherwise buy its way into
        a crowded context, so the content is measured and the declared figure is
        only ever a floor. Taking the larger of the two means a plugin cannot
        lie downwards, and an over-declared fragment merely reserves more room
        than it uses — which costs a little headroom, not correctness.
        """
        measured = 0
        if self._estimator is not None:
            try:
                measured = int(self._estimator([
                    {"role": "user", "content": fragment.content}]))
            except Exception:  # noqa: BLE001 - fall through to the local estimate
                measured = 0
        if measured <= 0:
            measured = len(fragment.content) // 4
        return max(1, measured, int(fragment.tokens or 0))

    # -- rendering ---------------------------------------------------------

    def render(self, admitted: Sequence[ContextFragment]) -> str:
        """Flatten admitted fragments into the block that enters context.

        Each is attributed to the plugin that contributed it, so anything the
        model reads can be traced back to whoever put it there.
        """
        if not admitted:
            return ""
        parts = []
        for fragment in admitted:
            origin = fragment.provenance or "core"
            parts.append(f"[context · {fragment.source} · from {origin}]\n{fragment.content}")
        return "\n\n".join(parts)

    def as_messages(self, admitted: Sequence[ContextFragment]) -> List[Dict[str, Any]]:
        """Admitted fragments as message dicts, ready for the request."""
        rendered = self.render(admitted)
        if not rendered:
            return []
        return [{"role": "user", "content": rendered}]

    def describe(self) -> Dict[str, Any]:
        """Everything the plane is holding, for `/context`."""
        return {
            "capacity": self.capacity,
            "reserved": self.reserved,
            "available": self.available,
            "held": {source: {"priority": fragment.effective_priority,
                              "pinned": fragment.pinned,
                              "tokens": self._measure(fragment),
                              "trigger": str(fragment.trigger),
                              "provenance": fragment.provenance}
                     for source, fragment in self.fragments.items()},
            "scores": dict(self.last_scores),
            "last": self.last_report.to_dict() if self.last_report else None,
        }


def build_plane(capacity: int, fragments: Sequence[ContextFragment] = (),
                **kwargs) -> ContextPlane:
    """A plane pre-loaded with fragments, for the common case."""
    plane = ContextPlane(capacity, **kwargs)
    plane.contribute_many(fragments)
    return plane


def project_instruction_files(workspace: Path | str) -> List[ContextFragment]:
    """Project-level instructions, in precedence order.

    `AGENTS.md` and `.harness/instructions/*.md` are the conventional carriers
    of project knowledge. They are pinned, because a project instruction the
    harness silently dropped would be worse than no project instructions at all.
    """
    root = Path(workspace)
    fragments: List[ContextFragment] = []
    conventions = root / "AGENTS.md"
    if conventions.is_file():
        try:
            fragments.append(ContextFragment(
                source="project:AGENTS.md",
                content=conventions.read_text(encoding="utf-8", errors="replace"),
                tokens=0, priority=PRIORITY_PINNED, trigger=Trigger.parse("always"),
                pinned=True, provenance="project"))
        except OSError:
            pass
    instructions = root / ".harness" / "instructions"
    if instructions.is_dir():
        for path in sorted(instructions.glob("*.md")):
            try:
                fragments.append(ContextFragment(
                    source=f"project:{path.name}",
                    content=path.read_text(encoding="utf-8", errors="replace"),
                    tokens=0, priority=PRIORITY_PINNED, trigger=Trigger.parse("always"),
                    pinned=True, provenance="project"))
            except OSError:
                continue
    return fragments
