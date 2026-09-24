"""Conditional request compaction with a pristine saved conversation."""

from __future__ import annotations

import json
from typing import Any

from adaptive_harness.agent.compaction import compact_tool_output
from adaptive_harness.llm.providers import context_window


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    # A conservative character estimate until the provider reports usage.
    serialized = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False, default=str)
    return max(1, (len(serialized) + 3) // 4)


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
        if "Diff:\n" in content or ("--- a/" in content and "+++ b/" in content):
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
        # Summarize only complete old turns, avoiding dangling tool-call pairs.
        cutoff = max(1, len(copied) - 8)
        first_old_user = next((i for i in range(1, cutoff) if copied[i].get("role") == "user"), None)
        if first_old_user is not None:
            end = max((i for i in range(first_old_user + 1, cutoff)
                       if copied[i].get("role") == "user"), default=cutoff)
            old = copied[first_old_user:end]
            if old and not any(i in protected for i in range(first_old_user, end)):
                requests = [str(item.get("content") or "")[:180] for item in old if item.get("role") == "user"]
                outcomes = [str(item.get("content") or "")[:220] for item in old
                            if item.get("role") == "assistant" and not item.get("tool_calls")]
                note = "Earlier conversation summary (original turns retained in session storage):\n"
                note += "Requests: " + " | ".join(requests[-6:]) + "\nOutcomes: " + " | ".join(outcomes[-6:])
                copied[first_old_user:end] = [{"role": "user", "content": note[:2400]}]

    compacted = estimate_tokens(copied, tools)
    info["compacted_tokens"] = max(0, original - compacted)
    info["compacted"] = info["compacted_tokens"] > 0
    info["used_tokens"] = compacted
    info["utilization"] = compacted / capacity
    return copied, info
