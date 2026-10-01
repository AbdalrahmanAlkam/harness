"""Installing, removing, and listing plugins.

The core ships with none, so this is how capabilities arrive. Three rules shape
it:

- **A local directory is installed by copying it, not by referencing it.** A
  plugin installed from a git checkout stops working when the checkout moves,
  and the failure is at run time rather than at install time. A copy is
  self-contained and survives being moved, archived, or deleted.
- **Installing runs the validator, which does not execute the plugin.** Deciding
  whether to trust something and running it are different acts.
- **A broken install is removed, not left half-working.** A plugin that will
  not load is reported by name; removing it is one command.

There is no privileged tier. An official plugin and a community plugin install
into the same directory and are trusted exactly as much, because a plugin that
deserves more trust deserves to be in the core, and one that does not is
whatever it says on its tin.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from adaptive_harness.data.config import DEFAULT_CONFIG_DIR
from adaptive_harness.plugins.lifecycle import validate


class InstallError(Exception):
    """An install could not be completed. The message says why, in words."""


@dataclass
class InstalledPlugin:
    """One plugin present in the user plugin directory."""

    name: str
    version: str
    directory: Path
    description: str = ""
    permissions: tuple[str, ...] = ()
    valid: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "version": self.version,
                "description": self.description,
                "permissions": list(self.permissions),
                "valid": self.valid, "error": self.error}


@dataclass
class Registry:
    """The installed plugin directory, and the operations over it."""

    directory: Path = field(default_factory=lambda: Path(DEFAULT_CONFIG_DIR) / "plugins")

    def ensure(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory

    def list(self) -> List[InstalledPlugin]:
        """Everything installed, including the ones that will not load.

        A plugin that fails to load is *listed* rather than hidden: a user who
        installed something and cannot see it has no way to diagnose it, and a
        silent no-op is the failure mode this whole system exists to avoid.
        """
        if not self.directory.is_dir():
            return []
        found: List[InstalledPlugin] = []
        for child in sorted(self.directory.iterdir()):
            if not child.is_dir():
                continue
            if not child.exists():
                # A directory that vanished between the listing and now --
                # usually a concurrent uninstall. Skipping it is right; failing
                # would make an unrelated listing command blow up.
                continue
            report = validate(child)
            found.append(InstalledPlugin(
                name=report.name or child.name,
                version=report.version,
                directory=child,
                description=_description_of(child),
                permissions=report.permissions,
                valid=report.ok,
                # `errors` is a list; the first is the headline reason and the
                # rest are usually a cascade from it.
                error="; ".join(report.errors) if report.errors else "",
            ))
        return found

    def get(self, name: str) -> Optional[InstalledPlugin]:
        return next((item for item in self.list() if item.name == name), None)

    def install(self, source: Path | str, *, overwrite: bool = False) -> Path:
        """Copy a plugin directory into place, after validating it."""
        origin = Path(source).expanduser().resolve()
        if not origin.is_dir():
            raise InstallError(f"No plugin directory at {origin}")
        report = validate(origin)
        if not report.ok:
            raise InstallError(
                f"{origin.name} did not validate, so it was not installed:\n{report.render()}")
        target = self.ensure() / report.name
        if target.exists():
            if not overwrite:
                raise InstallError(
                    f"{report.name} is already installed. Pass --overwrite to replace it.")
            shutil.rmtree(target)
        shutil.copytree(origin, target, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache"))
        return target

    def uninstall(self, name: str) -> bool:
        target = self.directory / name
        if not target.is_dir():
            return False
        shutil.rmtree(target)
        return True

    def is_installed(self, name: str) -> bool:
        return (self.directory / name).is_dir()


def _description_of(directory: Path) -> str:
    """Read a manifest's description without failing on a broken one."""
    import json

    for manifest in directory.glob("*.plugin.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        if isinstance(payload, dict):
            return str(payload.get("description", ""))
    return ""


#: The plugins we publish, and the one-line reason each exists. The first
#: release ships none of them: this is the staged path, and a user installs
#: only what they want rather than inheriting a toolbelt they never chose.
#:
#: `min_harness` is enforced at install time, so a plugin that needs a newer
#: core says so instead of failing mysteriously later.
OFFICIAL: List[dict] = [
    {"name": "web", "summary": "Fetch a page as markdown, and search if you have a key.",
     "why": "Research mode can search but had nothing to read a result with.",
     "min_harness": "2.0"},
    {"name": "lsp", "summary": "Definitions, references, hover and diagnostics, with no "
                                "language server required.",
     "why": "Answer 'where is this defined' without configuring a toolchain.",
     "min_harness": "2.0"},
    {"name": "tasklist", "summary": "Live task state, where 'done' needs observed evidence.",
     "why": "Shows progress on a long run, and cannot be marked done by assertion.",
     "min_harness": "2.0"},
    {"name": "sandbox", "summary": "Confine a shell command, and say which backend "
                                    "actually confined it.",
     "why": "Profiles restrict tools; this restricts the process.",
     "min_harness": "2.0"},
    {"name": "agents-md", "summary": "Load the project's own instructions.",
     "why": "Project knowledge should not have to be retyped every session.",
     "min_harness": "2.0"},
    {"name": "todo-scan", "summary": "Find TODO/FIXME markers.",
     "why": "A worked example, and the smallest useful plugin there is.",
     "min_harness": "2.0"},
]


def find_official(name: str, search_paths) -> Optional[Path]:
    """Locate an official plugin by name in the given search paths."""
    for root in search_paths:
        candidate = Path(root) / name
        if candidate.is_dir():
            return candidate
    return None
