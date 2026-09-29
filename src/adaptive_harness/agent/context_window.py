"""Conditional request compaction with a pristine saved conversation."""

from __future__ import annotations

import json
from typing import Any

from adaptive_harness.agent.compaction import OPERATOR_PREFIX, compact_tool_output
from adaptive_harness.agent.hierarchical import compact_hierarchically
from adaptive_harness.llm.providers import context_window


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Estimate the tokens a request would use.

    Delegates to the active backend in `tokenizer`, which is a pluggable
    estimator rather than a fixed `len//4`. The signature is unchanged, so
    nothing that calls this had to move when the backends arrived.
    """
    from adaptive_harness.agent.tokenizer import count

    return count(messages, tools)


def prepare_context(messages: list[dict[str, Any]], model: str,
                    tools: list[dict[str, Any]] | None = None, *, provider: str = "openrouter",
                    limit: int | None = None, threshold: float = 0.75) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Only compact a request after the estimated context exceeds the threshold.

    System directives, active diffs, and recent tool pairs are preserved. The
    source messages are never modified, so saved sessions remain exact.
    """
    capacity = context_window(model, provider, limit)
    original = estimate_tokens(messages, tools)
    info = {"used_tokens": original, "capacity": capacity,
            "utilization": original / capacity, "compacted_tokens": 0, "compacted": False}
    if original / capacity <= threshold:
        return messages, info

    copied = [dict(message) for message in messages]
    protected = {index for index, message in enumerate(copied)
                 if message.get("role") in {"system", "developer"}}
    # Keep the latest eight entries and any active file diff verbatim.
    protected.update(range(max(0, len(copied) - 8), len(copied)))
    for index, message in enumerate(copied):
        content = str(message.get("content") or "")
        # An operator steer is a user message sitting between two assistant
        # tool-calling turns, so it lands squarely inside the old-turn window
        # that the second pass replaces with a summary. Without this it would be
        # reduced to nothing a few steps after the user typed it.
        if (content.startswith(OPERATOR_PREFIX) or "Diff:\n" in content
                or ("--- a/" in content and "+++ b/" in content)):
            protected.add(index)
    for index, message in enumerate(copied):
        if index in protected or message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        if len(content) > 1200:
            copied[index]["content"] = compact_tool_output(str(message.get("name") or "run_bash"),
                                                              content, limit=1200)

    used = estimate_tokens(copied, tools)
    if used / capacity > threshold:
        # Map->reduce over the old cohort, replacing a single lossy pass. A
        # second attempt is made if one fold was not enough.
        # The foldable region starts after the protected head, which is the
        # system message and anything pinned before it.
        head = (max(protected) + 1) if protected else 0
        tail = 8
        for _ in range(2):
            folded, report = compact_hierarchically(
                copied, protected=protected, head=head, tail=tail,
                estimate=lambda items: estimate_tokens(items, tools))
            if not report.generations and report.messages_after == report.messages_before:
                break
            copied = folded
            if estimate_tokens(copied, tools) / capacity <= threshold:
                break

    compacted = estimate_tokens(copied, tools)
    info["compacted_tokens"] = max(0, original - compacted)
    info["compacted"] = info["compacted_tokens"] > 0
    info["used_tokens"] = compacted
    info["utilization"] = compacted / capacity
    return copied, info
