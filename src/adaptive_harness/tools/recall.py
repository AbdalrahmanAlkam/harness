"""Retrieval of tool output that the filter compressed out of the transcript."""

from __future__ import annotations

from typing import Any

from adaptive_harness.agent.tool_filter import ToolOutputArchive
from adaptive_harness.tools.base import Tool, ToolResult


class ReadFullOutputTool(Tool):
    """Read tool output that a previous step compressed out of the context.

    When a large, low-value tool result is filtered, the transcript keeps a short
    summary plus a token. If the summary turns out not to be enough, this returns
    the original text in full, so filtering is never lossy.
    """

    name = "read_full_output"
    description = ("Read the complete output of an earlier tool call that was "
                   "compressed to save context. Pass the token shown in that "
                   "message, for example OUTPUT-3.")
    parameters = {
        "type": "object",
        "properties": {
            "token": {"type": "string", "description": "The OUTPUT-n token from the earlier message."},
        },
        "required": ["token"],
    }

    def __init__(self, archive: ToolOutputArchive, *, limit: int = 20_000):
        self.archive = archive
        self.limit = limit

    def execute(self, **kwargs: Any) -> ToolResult:
        token = str(kwargs.get("token") or "").strip()
        if not token:
            return ToolResult(success=False, output="", error="A token is required, e.g. OUTPUT-1")
        archived = self.archive.fetch(token)
        if archived is None:
            # The archive is bounded, so an old token is genuinely gone. Say so
            # plainly and tell the model how to get the text again.
            return ToolResult(
                success=False, output="",
                error=(f"Unknown or expired token {token}. It was filtered earlier in this "
                       f"session but the archive no longer holds it; re-run the tool to see "
                       f"its output again."))
        truncated = len(archived) > self.limit
        if truncated:
            head = self.limit * 2 // 3
            tail = self.limit - head - 60
            # Say so up front. The notice in the transcript promises the full
            # output, and quietly returning a slice would make the model reason
            # about text it never saw.
            banner = (f"[showing the first {head:,} and last {tail:,} of "
                      f"{len(archived):,} characters — this read is capped]\n\n")
            body = (banner + archived[:head].rstrip()
                    + f"\n… {len(archived) - head - tail:,} characters omitted …\n"
                    + archived[-tail:].lstrip())
        else:
            body = archived
        return ToolResult(success=True, output=body,
                          metadata={"token": token, "truncated": truncated,
                                    "total_chars": len(archived)})
