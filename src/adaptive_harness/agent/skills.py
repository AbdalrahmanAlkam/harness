"""Built-in and local custom skill discovery."""
from __future__ import annotations

import json
from pathlib import Path

from adaptive_harness.skills.registry import BUILTIN_BY_NAME, BaseSkill


_SAFE_DEFAULT_TOOLS = ("read_file", "search_files", "list_directory")
_KNOWN_TOOLS = frozenset({"read_file", "search_files", "list_directory", "edit_file", "write_file",
                          "run_pytest", "run_bash", "calculate", "run_python_repl", "verify_equation",
                          "check_convergence", "plot_terminal", "web_search"})
_KNOWN_INVARIANTS = frozenset({"edit_validated", "tests_green", "evidence_checked", "git_checked",
                               "symbolic_checked", "numerical_checked", "sources_checked",
                               "execution_checked", "dependency_checked"})

class SkillCatalog:
    def __init__(self, workspace: str | Path, config_dir: str | Path | None = None):
        self.workspace = Path(workspace).resolve()
        self.config_dir = Path(config_dir).expanduser() if config_dir is not None else Path.home() / ".config" / "adaptive-harness"

    def discover(self) -> dict[str, Path]:
        roots = (self.workspace / "skills", self.workspace / ".harness" / "skills", self.config_dir / "skills")
        result: dict[str, Path] = {}
        for root in roots:
            if root.is_dir():
                for path in sorted(root.glob("*/SKILL.md")):
                    if path.is_symlink() or path.parent.is_symlink() or path.parent.name in BUILTIN_BY_NAME:
                        continue
                    result.setdefault(path.parent.name, path)
        return result

    def all(self) -> dict[str, BaseSkill]:
        catalog = dict(BUILTIN_BY_NAME)
        for name, path in self.discover().items():
            try:
                content = path.read_text(encoding="utf-8")
                if not content.strip() or len(content) > 30_000:
                    continue
                manifest_path = path.with_name("skill.json")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
                if not isinstance(manifest, dict):
                    continue
                tools = manifest.get("tools", _SAFE_DEFAULT_TOOLS)
                invariants = manifest.get("invariants", ())
                if (not isinstance(tools, (list, tuple)) or not tools or
                        not set(tools).issubset(_KNOWN_TOOLS) or
                        not isinstance(invariants, (list, tuple)) or
                        not set(invariants).issubset(_KNOWN_INVARIANTS)):
                    continue
                catalog[name] = BaseSkill(name=name,
                    title=str(manifest.get("title", name.replace("_", " ").title())),
                    category=str(manifest.get("category", "Custom")),
                    trigger=str(manifest.get("trigger", name.replace("_", " "))),
                    instructions=content, tools=tuple(tools), invariants=tuple(invariants),
                    icon=str(manifest.get("icon", "✦")), source=str(path))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
                continue
        return catalog

    def get(self, name: str) -> BaseSkill:
        try:
            return self.all()[name]
        except KeyError as exc:
            raise ValueError(f"Unknown skill: {name}") from exc

    def read(self, name: str) -> str:
        return self.get(name).instructions
