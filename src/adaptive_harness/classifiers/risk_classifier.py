"""High-impact tool call detection for the last authorization boundary."""
from __future__ import annotations

import re


class ToolRiskClassifier:
    @staticmethod
    def catastrophic_intent(text: str) -> bool:
        return bool(re.search(r"\b(?:format|wipe)\s+(?:a\s+|the\s+)?(?:disk|drive)\b|"
                              r"\bdrop\s+(?:external\s+)?database\b|"
                              r"\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)[a-z]*\s+/(?:\s|$|[;&|])",
                              text, re.I))
    PATTERNS = (
        r"\bgit\s+(?:reset\s+--hard|clean\s+(?:-[\w]*f[\w]*|--force)|checkout\s+--\s+|restore\s+\.\s*(?:$|[;&|]))",
        r"\b(?:mkfs|fdisk|wipefs)\b",
        r"\bdd\s+.*\bof=/dev/",
        r"\b(?:drop\s+(?:database|table)|truncate\s+table)\b",
    )

    def evaluate(self, tool_name: str, arguments: dict, *, catastrophic_only: bool = False) -> str | None:
        if tool_name != "run_bash":
            return None
        command = str(arguments.get("command", ""))
        if catastrophic_only:
            disasters = (
                r"\bsudo\b", r"\b(?:mkfs|fdisk|wipefs)\b",
                r"\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)[a-z]*\s+/\s*(?:$|[;&|])",
                r"\b(?:drop\s+database|drop\s+table)\b",
            )
            return command if any(re.search(pattern, command, re.I) for pattern in disasters) else None
        for match in re.finditer(r"\brm\s+([^;&|\n]{1,120})", command, re.IGNORECASE):
            args = match.group(1).split()
            flags = [arg for arg in args if arg.startswith("-")]
            if any(flag in ("--recursive", "--force") or
                   (flag.startswith("-") and ("r" in flag[1:] or "f" in flag[1:])) for flag in flags):
                return command
        return command if any(re.search(pattern, command, re.IGNORECASE) for pattern in self.PATTERNS) else None
