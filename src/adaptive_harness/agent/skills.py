"""Discover user-selected SKILL.md guidance in the workspace and user config."""
from __future__ import annotations

from pathlib import Path


class SkillCatalog:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()

    def discover(self) -> dict[str, Path]:
        roots = [self.workspace / ".harness" / "skills", Path.home() / ".config" / "adaptive-harness" / "skills"]
        result: dict[str, Path] = {}
        for root in roots:
            if root.is_dir():
                for path in sorted(root.glob("*/SKILL.md")):
                    result.setdefault(path.parent.name, path)
        return result

    def read(self, name: str) -> str:
        path = self.discover().get(name)
        if path is None:
            raise ValueError(f"Unknown skill: {name}")
        content = path.read_text(encoding="utf-8")
        if len(content) > 30_000:
            raise ValueError(f"Skill is too large: {name}")
        return content
