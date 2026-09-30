"""Rendering the prompt list.

`prompts list` is the audit view: the thing a user opens to see every instruction
the models receive. It is only useful if a row is *readable* on one line.

Three things break that, and each is fixed here rather than in the command, so
they can be tested:

- a prompt that starts on a second line previews as a fragment of the middle.
  The whole prompt is collapsed to one line first.
- a hard cut at 90 characters produces a word half-finished and, at a narrow
  terminal, wraps into the next row and destroys the column. The cut is at a
  word boundary and is width-aware.
- 41 flat rows in alphabetical order give no sense of which ones can actually
  fire, and which are dead. They are grouped by what triggers them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

#: Narrower than this and the preview column is not worth having; the name and
#: the size still are.
MIN_PREVIEW_WIDTH = 20

#: The name column, sized to the longest name plus a gutter.
NAME_GUTTER = 2

#: Shown next to a prompt whose text is overridden, and counted in the width
#: budget so an overridden row is not the one that wraps.
OVERRIDE_FLAG = " *overridden"

#: How each group is headed. The heading is the useful part: it says when the
#: prompt can reach a model, which a flat alphabetical list does not.
GROUPS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Always in context", "composed into the system prompt of every run",
     ("system.default", "system.tool_guidance", "system.workspace_line",
      "system.mode_line", "system.preferences_line", "system.exemplar_notice",
      "swarm.workspace_limit")),
    ("Mid-run injections", "appended as messages while a run is in flight",
     ("harness.continuation", "harness.missing_file_changes",
      "harness.verification_repair", "operator.steer")),
    ("Classifier interventions", "a local classifier can override the model with these",
     ("intervention.drift", "intervention.hallucination", "intervention.looping",
      "intervention.stalled", "intervention.stop_circling",
      "intervention.termination_verdict")),
    ("Domain guidance", "selected by operational mode",
     ("domain.guidance.coding", "domain.guidance.research",
      "domain.guidance.science", "domain.guidance.audit", "domain.guidance.plan")),
    ("Swarm", "subagent roles and per-assignment instructions",
     ("system.swarm_coordinator", "swarm.role.coder", "swarm.role.read_only",
      "swarm.role.security", "swarm.context_header",
      "swarm.instruction.architect.plan", "swarm.instruction.coder.implement",
      "swarm.instruction.qa.plan", "swarm.instruction.qa.verify",
      "swarm.instruction.security.verify")),
    ("Research", "the autonomous research swarm",
     ("research.director", "research.division.theory",
      "research.division.empirical", "research.division.adversarial",
      "research.division.literature")),
    ("Supporting models", "sent to a model other than the main one",
     ("tool_filter.summarize", "classifier.local_http", "classifier.ollama",
      "classifier.openrouter")),
]


@dataclass(frozen=True)
class Row:
    """One prompt, ready to print."""

    name: str
    preview: str
    size: int
    overridden: bool

    @property
    def marker(self) -> str:
        return "*" if self.overridden else " "


def collapse(text: str) -> str:
    """The whole prompt on one line.

    A preview taken from ``splitlines()[0]`` is a fragment whenever the prompt
    opens with a blank line or a second sentence, which is most of them.
    """
    return " ".join(str(text).split())


def truncate(text: str, width: int) -> str:
    """Cut at a word boundary and say so, rather than mid-word."""
    if width <= 0 or len(text) <= width:
        return text
    cut = text[: max(1, width - 1)]
    space = cut.rfind(" ")
    if space > width // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"


def terminal_width(console=None, default: int = 100) -> int:
    """The width to lay out for, from the console that will print it.

    Rich wraps to *its own* width, so the preview has to be sized to that and
    not to the terminal directly: a `console.width` that differs from the
    terminal produces a preview that is then wrapped anyway, which is the
    ragged column this replaced. A pipe or a file gets the default rather than
    the degenerate 80-column default Rich would otherwise pick.
    """
    if console is not None:
        try:
            width = console.width
        except Exception:  # noqa: BLE001 - a probe must not break the listing
            width = 0
        if width and width >= 40:
            return int(width)
        return default
    raw = os.environ.get("COLUMNS")
    if raw and raw.isdigit():
        return max(40, int(raw))
    try:
        size = os.get_terminal_size().columns
    except OSError:
        return default
    return size if size >= 40 else default


def build_rows(registry, width: int | None = None, console=None) -> List[Row]:
    """One row per prompt, previewed to fit the terminal."""
    layout = terminal_width(console) if width is None else width
    name_width = max((len(name) for name in registry.names()), default=20)
    name_width = max(name_width, 20)
    # 2 indent + name + 2 gutter + ">6c  " (the size column) + the override flag.
    size_budget = 8 if layout - (2 + name_width + 2) > MIN_PREVIEW_WIDTH + 8 else 0
    fixed = 2 + name_width + 2 + size_budget + len(OVERRIDE_FLAG)
    preview_width = max(MIN_PREVIEW_WIDTH, layout - fixed)

    # Below this width the name column alone fills the line, so the size column
    # is dropped rather than the preview being cut to an ellipsis -- a prompt
    # list that shows 40 ellipses and no content is not an audit view.
    show_size = layout - (2 + name_width + 2 + len(OVERRIDE_FLAG)) > MIN_PREVIEW_WIDTH

    rows: List[Row] = []
    for name in sorted(registry.names()):
        text = registry.get(name)
        rows.append(Row(name=name,
                        preview=truncate(collapse(text), preview_width),
                        size=len(text) if show_size else 0,
                        overridden=registry.is_overridden(name)))
    return rows


def group_rows(rows: Iterable[Row]) -> List[Tuple[str, str, List[Row]]]:
    """Grouped by trigger, with anything unrecognised kept rather than lost.

    A new prompt that nobody has grouped yet must still be listed. A prompt that
    is invisible because someone forgot a table row is worse than one that is
    merely out of place.
    """
    by_name: Dict[str, Row] = {row.name: row for row in rows}
    claimed: set[str] = set()
    groups: List[Tuple[str, str, List[Row]]] = []
    for heading, blurb, names in GROUPS:
        members = [by_name[name] for name in names if name in by_name]
        claimed.update(name for name in names if name in by_name)
        if members:
            groups.append((heading, blurb, members))
    ungrouped = [row for name, row in sorted(by_name.items()) if name not in claimed]
    if ungrouped:
        groups.append(("Other", "not yet grouped by trigger", ungrouped))
    return groups


def render(registry, width: int | None = None, console=None) -> List[str]:
    """The whole listing as plain lines, so it can be printed or asserted on."""
    rows = build_rows(registry, width, console)
    layout = terminal_width(console) if width is None else width
    groups = group_rows(rows)
    name_width = max((len(row.name) for row in rows), default=20)
    lines: List[str] = []
    for index, (heading, blurb, members) in enumerate(groups):
        if index:
            lines.append("")
        head = heading if len(heading) <= layout else truncate(heading, layout)
        lines.append(head)
        if blurb:
            lines.append(f"    {truncate(blurb, max(0, layout - 4))}")
        for row in members:
            flag = OVERRIDE_FLAG if row.overridden else ""
            size_field = f"{row.size:>6}c  " if row.size else ""
            lines.append(f"  {row.name.ljust(name_width)}  "
                         f"{size_field}{row.preview}{flag}")
    return lines
