"""Hierarchical compaction: map, then reduce, over the old cohort.

Single-pass summarization loses detail irreversibly. If the oldest twenty turns
are replaced by one note, whatever that note happened to keep is gone for the
rest of the run, and a later question about it can only be answered from the
session store rather than from the model's own context.

Map→reduce fixes that by folding in generations. The oldest cohort is
summarized; the summaries are then summarized with each other, and so on until
the result fits a fixed budget. Each generation is a smaller loss than the next,
and the loss is bounded and predictable rather than whatever one pass happened
to drop.

Three properties this preserves from the single-pass version it replaces,
because they are what made that version safe:

- only *complete* turns are folded, so a tool call is never separated from its
  result;
- protected regions -- pinned fragments, the system message, the tail -- are
  exempt;
- **an error is never summarized.** Sacred invariant 5. A failed tool call's
  text is the single most valuable thing to keep verbatim, because it is what
  the model needs in order not to repeat the mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

#: A cohort is folded when it has at least this many messages. Folding a
#: two-message cohort loses more structure than it saves.
MIN_COHORT = 6

#: How many summaries are reduced together per generation. Larger means fewer
#: generations and more loss per one; this is a balance, not a constant of nature.
FAN_IN = 4

#: The budget a generation's output is allowed. The final generation is the one
#: that matters; earlier ones exist to make the final one lossy-but-bounded.
GENERATION_BUDGET = 2400

#: A summary longer than this is truncated, so one enormous tool result cannot
#: become one enormous summary that then becomes one enormous next summary.
MAX_SUMMARY_CHARS = 2400


@dataclass
class Cohort:
    """A run of complete turns eligible for folding."""

    messages: List[Dict[str, Any]]
    #: Which fold this cohort is, counting from 0. Reported so a user can see
    #: how much history has been through the machine.
    generation: int = 0

    def __len__(self) -> int:
        return len(self.messages)


@dataclass
class CompactionReport:
    """What the fold did, for the budget surface and the transcript."""

    generations: int = 0
    messages_before: int = 0
    messages_after: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    cohorts: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"generations": self.generations,
                "messages_before": self.messages_before,
                "messages_after": self.messages_after,
                "tokens_before": self.tokens_before,
                "tokens_after": self.tokens_after,
                "cohort_sizes": list(self.cohorts)}


def complete_turns(messages: Sequence[Dict[str, Any]], start: int,
                   stop: int) -> List[int]:
    """Indices in ``[start, stop)`` that form whole turns.

    A turn is a user message plus everything the model and its tools produced in
    answer. Folding must not split one, or the result is an assistant message
    with tool calls and no results -- a history the provider rejects.
    """
    indices: List[int] = []
    index = start
    while index < stop:
        if messages[index].get("role") != "user":
            index += 1
            continue
        end = index + 1
        while end < stop and messages[end].get("role") != "user":
            end += 1
        indices.extend(range(index, end))
        index = end
    return indices


def _turn_digest(turn: Sequence[Dict[str, Any]], limit: int = MAX_SUMMARY_CHARS) -> str:
    """One turn, rendered for a later summary to work from."""
    lines: List[str] = []
    for message in turn:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user":
            lines.append(f"asked: {content[:300]}")
        elif role == "assistant" and message.get("tool_calls"):
            names = ", ".join(
                str((call.get("function") or {}).get("name", "?"))
                for call in message["tool_calls"])
            lines.append(f"called: {names}")
        elif role == "tool":
            # An error is kept verbatim and flagged. Losing it is the one thing
            # that must never happen: it is what stops the model repeating the
            # mistake it just made.
            if message.get("content", "").startswith("ERROR") or not _looks_successful(message):
                lines.append(f"tool error (verbatim, do not lose): {content[:600]}")
            else:
                lines.append(f"tool ok: {content[:200]}")
        else:
            lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)[:limit]


def _looks_successful(message: Dict[str, Any]) -> bool:
    content = str(message.get("content") or "")
    return not content.lstrip().startswith("ERROR")


def fold(turns: Sequence[Sequence[Dict[str, Any]]],
         summarizer: Callable[[str, int], str],
         *, budget: int = GENERATION_BUDGET,
         max_generations: int = 6) -> tuple[List[Dict[str, Any]], int]:
    """Map, then reduce, until the result fits ``budget``.

    ``summarizer`` takes the text of a cohort and its generation number and
    returns its summary. Returning the text unchanged is a valid, lossless
    outcome -- a summary that cannot compress is not an error.
    """
    summaries: List[Dict[str, Any]] = []
    generation = 0
    while True:
        # Map: one summary per turn (or per existing summary, on later passes).
        mapped: List[Dict[str, Any]] = []
        for turn in turns:
            text = "\n".join(turn) if turn and isinstance(turn[0], str) else _turn_digest(turn)
            # No pre-truncation: handing a summarizer a slice of its input is how
            # a lossless fold silently becomes a lossy one.
            mapped.append({"role": "user",
                           "content": f"Summary of earlier work (fold {generation + 1}):\n"
                                      f"{summarizer(text, generation)}"})
        # Reduce: fold summaries together, FAN_IN at a time, until one remains or
        # they all fit.
        while len(mapped) > 1:
            nxt: List[Dict[str, Any]] = []
            for start in range(0, len(mapped), FAN_IN):
                group = mapped[start:start + FAN_IN]
                if len(group) == 1:
                    nxt.extend(group)
                    continue
                joined = "\n\n".join(str(item.get("content", "")) for item in group)
                reduced = summarizer(joined, generation)
                # A summarizer that cannot compress must not make the summary
                # *larger* than what went in: the "fold N" header and the joined
                # text would otherwise grow without bound, and the hierarchy
                # would be making the context worse every generation.
                header = f"Summary of earlier work (fold {generation + 1}):\n"
                # A summarizer that cannot compress must not make the summary
                # *larger* than what went in. But it must not silently drop the
                # tail either: the honest outcome is to say so and keep the
                # material, because losing turns is worse than a summary that
                # did not shrink.
                if len(header) + len(reduced) >= len(joined):
                    reduced = (f"{joined}\n(the fold above could not compress this "
                               f"cohort, so it is kept in full)")
                nxt.append({"role": "user", "content": header + reduced})
            mapped = nxt
        # The final guard, for the single-summary case where the reduce loop
        # never ran. Say what happened rather than truncating silently: a cohort
        # that could not be compressed stays, with a note saying so.
        if len(mapped) == 1 and len(str(mapped[0].get("content", ""))) > budget:
            mapped[0] = {"role": "user", "content": (
                f"{mapped[0]['content']}\n"
                f"(this cohort could not be compressed to the budget and is kept "
                f"in full rather than truncated)")}
        total = sum(len(str(item.get("content", ""))) for item in mapped)
        summaries = mapped
        if total <= budget or generation >= max_generations:
            break
        # Still too large: the summaries are themselves the input to another
        # generation, which is the point of the hierarchy.
        turns = [[str(item.get("content", ""))] for item in mapped]
        generation += 1
    return summaries, generation


def compact_hierarchically(messages: List[Dict[str, Any]], *,
                           protected: Iterable[int],
                           head: int, tail: int,
                           budget: int = GENERATION_BUDGET,
                           summarizer: Optional[Callable[[str, int], str]] = None,
                           estimate: Optional[Callable[[Any], int]] = None,
                           ) -> tuple[List[Dict[str, Any]], CompactionReport]:
    """Replace the old cohort with a bounded, folded summary.

    ``head`` and ``tail`` bound the region being folded: the head is the system
    message and anything before it, the tail is the most recent messages, which
    are never summarized because they are what the model is working on now.
    """
    report = CompactionReport(messages_before=len(messages),
                              messages_after=len(messages))
    protected_set = set(protected)
    region = [index for index in range(head, max(head, len(messages) - tail))
              if index not in protected_set]
    if not region:
        return messages, report

    # Only whole turns may be folded.
    safe = [index for index in complete_turns(messages, head,
                                              max(head, len(messages) - tail))
            if index in set(region)]
    if len(safe) < MIN_COHORT:
        return messages, report

    turns: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for index in safe:
        current.append(messages[index])
        if index == safe[-1] or messages[index].get("role") == "assistant":
            turns.append(current)
            current = []
    if current:
        turns.append(current)
    turns = [turn for turn in turns if len(turn) >= 1]
    if not turns:
        return messages, report

    if summarizer is None:
        # The default is a *bounded* reduction, not an invention. It keeps the
        # errors verbatim (they are the part that must survive), then the
        # requests, then as much of the outcome text as fits. A summarizer that
        # invents an account of the work would be the exact failure this whole
        # module exists to prevent, so the default only ever discards.
        def summarizer(text: str, generation: int) -> str:
            lines = [line for line in text.splitlines() if line.strip()]
            errors = [line for line in lines if line.startswith("tool error")]
            asked = [line for line in lines if line.startswith("asked:")]
            other = [line for line in lines
                     if not line.startswith(("tool error", "asked:"))]
            kept: list[str] = []
            for group in (errors, asked, other):
                for line in group:
                    if len("\n".join(kept)) + len(line) > budget:
                        return "\n".join(kept) if kept else text[:budget]
                    kept.append(line)
            return "\n".join(kept)

    summaries, generations = fold(turns, summarizer, budget=budget)
    if not summaries:
        return messages, report

    compacted = (messages[:head] + summaries
                 + messages[max(head, len(messages) - tail):])
    report.generations = generations
    report.messages_after = len(compacted)
    report.cohorts = [len(turn) for turn in turns]
    if estimate is not None:
        try:
            report.tokens_before = int(estimate(messages))
            report.tokens_after = int(estimate(compacted))
        except Exception:  # noqa: BLE001 - the report is best-effort
            pass
    return compacted, report
