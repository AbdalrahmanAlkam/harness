"""Private local preferences and prompt history for the terminal UI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import re


DEFAULT_CONFIG_DIR = Path.home() / ".config" / "adaptive-harness"
DEFAULT_MODEL = "stealth/space-bunny-alpha"
DEFAULT_MODEL_SELECTION = "manual"
DEFAULT_SWARM_MODE = "auto"


def _private_write(path: Path, content: str) -> None:
    """Atomically replace a private file without ever creating it world-readable."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ConfigManager:
    """Stores the API key and theme in a user-owned JSON file."""

    def __init__(self, directory: Path | str | None = None):
        self.directory = Path(directory or DEFAULT_CONFIG_DIR).expanduser()
        self.path = self.directory / "config.json"
        self.last_error: str | None = None

    def load(self) -> dict[str, str]:
        self.last_error = None
        if not self.path.exists() and not self.path.is_symlink():
            return {}
        try:
            if self.path.is_symlink():
                raise ValueError("Configuration file is a symbolic link")
            os.chmod(self.path, 0o600)
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Configuration must be a JSON object")
            return {str(key): str(item) for key, item in value.items() if isinstance(item, str)}
        except (OSError, ValueError, UnicodeError) as exc:
            self.last_error = f"Could not read TUI configuration: {exc}"
            return {}

    def update(self, **changes: str | None) -> dict[str, str]:
        settings = self.load()
        for key, value in changes.items():
            if value is None:
                settings.pop(key, None)
            else:
                settings[key] = value
        _private_write(self.path, json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
        return settings

    def save_key(self, key: str) -> None:
        if not key.strip():
            raise ValueError("API key cannot be empty")
        self.update(api_key=key.strip())

    def clear_key(self) -> None:
        self.update(api_key=None)

    def save_theme(self, theme: str) -> None:
        self.update(theme=theme)


class PromptHistoryStore:
    """Keeps a private, bounded history of submitted task prompts."""

    LIMIT = 500

    def __init__(self, directory: Path | str | None = None):
        self.directory = Path(directory or DEFAULT_CONFIG_DIR).expanduser()
        self.path = self.directory / "prompt_history.txt"
        self.entries = self.load()

    def load(self) -> list[str]:
        if not self.path.exists() or self.path.is_symlink():
            return []
        try:
            os.chmod(self.path, 0o600)
            return [line for line in self.path.read_text(encoding="utf-8").splitlines() if line][-self.LIMIT:]
        except (OSError, UnicodeError):
            return []

    def record(self, prompt: str) -> bool:
        value = prompt.strip().replace("\n", " ").replace("\r", " ")
        if not value or value.startswith("/"):
            return False
        if re.search(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{12,}|(?:api[_-]?key|token)\s*[=:]\s*\S+", value, re.I):
            return False
        if self.entries and self.entries[-1] == value:
            return False
        updated = [*self.entries, value][-self.LIMIT:]
        _private_write(self.path, "\n".join(updated) + "\n")
        self.entries = updated
        return True
