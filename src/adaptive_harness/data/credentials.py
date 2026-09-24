"""Atomic, private per-provider credential storage."""

from __future__ import annotations

import json
import os
from pathlib import Path

from adaptive_harness.data.config import DEFAULT_CONFIG_DIR, _private_write
from adaptive_harness.llm.providers import PROVIDERS


class CredentialsManager:
    def __init__(self, directory: Path | str | None = None):
        self.path = Path(directory or DEFAULT_CONFIG_DIR).expanduser() / "credentials.json"

    def load(self) -> dict[str, str]:
        if not self.path.exists() and not self.path.is_symlink():
            return {}
        if self.path.is_symlink():
            raise ValueError("Credentials file must not be a symbolic link")
        os.chmod(self.path, 0o600)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Credentials file must contain a JSON object")
        return {key: value for key, value in payload.items()
                if key in PROVIDERS and key != "local" and isinstance(value, str) and value}

    def set(self, provider: str, key: str) -> None:
        if provider not in PROVIDERS or provider == "local":
            raise ValueError(f"Unknown key provider: {provider}")
        if not key.strip():
            raise ValueError("API key cannot be empty")
        data = self.load()
        data[provider] = key.strip()
        _private_write(self.path, json.dumps(data, indent=2) + "\n")

    def clear(self, provider: str) -> None:
        if provider not in PROVIDERS or provider == "local":
            raise ValueError(f"Unknown key provider: {provider}")
        data = self.load()
        data.pop(provider, None)
        _private_write(self.path, json.dumps(data, indent=2) + "\n")
