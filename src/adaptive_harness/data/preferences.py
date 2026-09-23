"""Private memory for reusable, non-destructive clarification answers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from adaptive_harness.data.config import DEFAULT_CONFIG_DIR, _private_write


class ClarificationMemory:
    LIMIT = 100
    TOPICS = ("database", "linter", "formatter", "test framework", "programming language")

    def __init__(self, directory: Path | str | None = None):
        self.path = Path(directory or DEFAULT_CONFIG_DIR).expanduser() / "preferences.json"

    @staticmethod
    def _key(question: str) -> str:
        normalized = " ".join(question.casefold().split())
        if "prefer" in normalized or "which" in normalized:
            for topic in ClarificationMemory.TOPICS:
                if topic in normalized:
                    return topic
        return normalized.strip(" ?.!:")

    @staticmethod
    def _reusable(question: str, context: str | None, answer: str | None = None) -> bool:
        text = f"{question} {context or ''}".casefold()
        if any(word in text for word in ("destructive", "delete", "remove", "overwrite", "permission",
                                         "execute", "run command", "wipe", "format disk",
                                         "what should i work on", "no identifiable task target",
                                         "clarify the goal", "api key", "password", "access token",
                                         "secret", "credential")):
            return False
        if answer and (answer.casefold().startswith(("abort", "cancel")) or len(answer) > 500):
            return False
        return True

    def load(self) -> dict[str, str]:
        if not self.path.exists() or self.path.is_symlink():
            return {}
        try:
            os.chmod(self.path, 0o600)
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return {str(key): str(value) for key, value in data.items()
                    if isinstance(key, str) and isinstance(value, str)}
        except (OSError, ValueError):
            return {}

    def lookup(self, question: str, context: str | None = None) -> str | None:
        if not self._reusable(question, context):
            return None
        return self.load().get(self._key(question))

    def remember(self, question: str, answer: str, context: str | None = None) -> bool:
        if not self._reusable(question, context, answer):
            return False
        data = self.load()
        data[self._key(question)] = answer.strip()
        _private_write(self.path, json.dumps(dict(list(data.items())[-self.LIMIT:]),
                                                 indent=2, ensure_ascii=False) + "\n")
        return True

    def guidance(self) -> str:
        entries = list(self.load().items())[-20:]
        return "\n".join(f"Known user decision — {question}: {answer}" for question, answer in entries)
