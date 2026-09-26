"""Recover explicitly named files from a model's fenced code response."""

from __future__ import annotations

import re

from adaptive_harness.llm.mock_client import ToolCall


_FENCE = re.compile(r"```[^\n`]*\n(.*?)\n```", re.S)
_HEADER = re.compile(r"^\s*(?:(?://|#|<!--)\s*)?path:\s*([^\s>]+)(?:\s*-->)?\s*$", re.I)
_INLINE = re.compile(r"^\s*[^\s`]+\s+(?://|#)\s*path:\s*([^\s]+)\s*$", re.I)
_EDIT_INTENT = re.compile(r"\b(?:create|write|build|implement|edit|modify|make|add|fix|refactor|generate)\b", re.I)


def requests_file_changes(prompt: str) -> bool:
    if not _EDIT_INTENT.search(prompt):
        return False
    if not re.search(r"^\s*(?:explain|describe|review|inspect|read|show|how|why)\b", prompt, re.I):
        return True
    # "Review and fix" requests implementation even though its first verb
    # is observational. "Show how to fix" remains read-only.
    return bool(re.search(r"\b(?:and|then)\s+(?:also\s+)?(?:create|write|build|implement|edit|modify|make|add|fix|refactor|generate)\b",
                          prompt, re.I))


def extract_file_calls(response: str, prompt: str) -> list[ToolCall]:
    """Accept only fenced code with an explicit path line or fence header."""
    if not requests_file_changes(prompt):
        return []
    calls: list[ToolCall] = []
    for index, block in enumerate(_FENCE.finditer(response), start=1):
        header = response[block.start():response.find("\n", block.start())].removeprefix("```")
        lines = block.group(1).splitlines(keepends=True)
        match = _INLINE.match(header) or (_HEADER.match(lines[0]) if lines else None)
        if not match:
            continue
        path = match.group(1).strip('"\'')
        if not _INLINE.match(header):
            lines = lines[1:]
        content = "".join(lines)
        if not content.strip() or len(content) > 200_000 or len(calls) >= 20:
            continue
        calls.append(ToolCall(id=f"code_fallback_{index}", name="write_file",
                              arguments={"path": path, "content": content}))
    return calls
