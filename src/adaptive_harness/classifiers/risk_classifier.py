"""High-impact tool call detection for the last authorization boundary."""
from __future__ import annotations

import re


class ToolRiskClassifier:
    PATTERNS = (
        r"\bgit\s+(?:reset\s+--hard|clean\s+(?:-[\w]*f[\w]*|--force)|checkout\s+--\s+|restore\s+\.\s*(?:$|[;&|]))",
        r"\b(?:mkfs|fdisk|wipefs)\b",
        r"\bdd\s+.*\bof=/dev/",
        r"\b(?:drop\s+(?:database|table)|truncate\s+table)\b",
    )

    def evaluate(self, tool_name: str, arguments: dict) -> str | None:
        if tool_name != "run_bash":
            return None
        command = str(arguments.get("command", ""))
        for match in re.finditer(r"\brm\s+([^;&|\n]{1,120})", command, re.IGNORECASE):
            args = match.group(1).split()
            flags = [arg for arg in args if arg.startswith("-")]
            if any(flag in ("--recursive", "--force") or
                   (flag.startswith("-") and ("r" in flag[1:] or "f" in flag[1:])) for flag in flags):
                return command
        return command if any(re.search(pattern, command, re.IGNORECASE) for pattern in self.PATTERNS) else None
