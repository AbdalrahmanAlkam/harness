"""High-impact tool call detection for the last authorization boundary."""
from __future__ import annotations

import re


class ToolRiskClassifier:
    PATTERNS = (
        r"\brm\s+(?:-[\w]*r[\w]*f[\w]*|-[\w]*f[\w]*r[\w]*)\b",
        r"\bgit\s+(?:reset\s+--hard|clean\s+-[\w]*f[\w]*d)",
        r"\b(?:mkfs|fdisk|wipefs)\b",
        r"\bdd\s+.*\bof=/dev/",
        r"\b(?:drop\s+(?:database|table)|truncate\s+table)\b",
    )

    def evaluate(self, tool_name: str, arguments: dict) -> str | None:
        if tool_name != "run_bash":
            return None
        command = str(arguments.get("command", ""))
        return command if any(re.search(pattern, command, re.IGNORECASE) for pattern in self.PATTERNS) else None
