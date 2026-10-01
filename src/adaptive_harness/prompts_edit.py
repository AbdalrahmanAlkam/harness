"""Editing and resetting prompt overrides.

`prompts show` tells you what the model will be sent. `prompts edit` lets you
change it, and `prompts reset` puts it back. Both are the same operation seen
from two directions: an override lives in a JSON file, and the interesting part
is *which* file.

Overrides merge from three places -- the user config, the project, and
`ADAPTIVE_PROMPTS_FILE` -- with later winning. So "reset this prompt" has a
trap: removing it from the file you happen to be looking at does not un-override
it if another file also overrides the same name. This module therefore reports
every file that overrides a prompt, and a reset that leaves it overridden says
so rather than reporting a success that did not happen.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

class PromptEditError(Exception):
    """An edit could not be completed. The message says what to do instead."""


def _config_dir() -> Path:
    """The user configuration directory, read at call time."""
    from adaptive_harness.data.config import DEFAULT_CONFIG_DIR

    return Path(DEFAULT_CONFIG_DIR)


@dataclass(frozen=True)
class OverrideFile:
    """One place overrides can live, in precedence order (later wins)."""

    path: Path
    label: str
    exists: bool

    def read(self) -> Dict[str, str]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PromptEditError(
                f"{self.path} is not valid JSON: {exc}\n"
                f"Fix or move it, then retry.") from exc
        if not isinstance(data, dict):
            raise PromptEditError(f"{self.path} must contain a JSON object of name -> text.")
        return {str(key): value for key, value in data.items() if isinstance(value, str)}

    def write(self, data: Dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The same private, atomic write the credentials use: an interrupted
        # edit must not leave a prompts file that fails to parse, because every
        # prompt in it would then be silently dropped.
        if self.path.parent.exists():
            self.path.parent.chmod(0o700)
        # A suffix on the *name*, not `with_suffix`: replacing the extension of
        # "from-env.json" yields "from-env.tmp123.json", which is not what any
        # caller expects and leaves a confusing name behind on failure.
        temp = self.path.with_name(f"{self.path.name}.tmp{os.getpid()}")
        try:
            temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            os.chmod(temp, 0o600)
            os.replace(temp, self.path)
        except OSError as exc:
            temp.unlink(missing_ok=True)
            raise PromptEditError(f"Could not write {self.path}: {exc}") from exc

    @property
    def is_ignored(self) -> bool:
        return str(self.path) == "-"


def override_files(workspace: Path | str | None = None) -> List[OverrideFile]:
    """The override files, in the order they are merged.

    The order is the registry's, so what this reports is what actually wins.
    """
    files = [
        OverrideFile(_config_dir() / "prompts.json", "user", False),
    ]
    if workspace:
        files.append(OverrideFile(Path(workspace) / ".harness" / "prompts.json",
                                 "project", False))
    env_file = os.getenv("ADAPTIVE_PROMPTS_FILE")
    if env_file:
        if env_file == "-":
            files.append(OverrideFile(Path("-"), "stdin (read-only)", True))
        else:
            files.append(OverrideFile(Path(env_file), "ADAPTIVE_PROMPTS_FILE", False))
    return [OverrideFile(f.path, f.label, f.is_ignored or f.path.is_file())
            for f in files]


def _named_env_file() -> Optional[Path]:
    """The file ADAPTIVE_PROMPTS_FILE names, when it names a real one."""
    value = os.getenv("ADAPTIVE_PROMPTS_FILE")
    return Path(value) if value and value != "-" else None


def _stdin_is_the_source(workspace: Path | str | None = None) -> bool:
    """Whether the *last* override source is stdin, and so cannot be written.

    Only the highest-precedence source matters: stdin being present but
    overridden by a writable project file is not a reason to refuse.
    """
    entries = [entry for entry in override_files(workspace) if entry.exists or entry.is_ignored]
    return bool(entries) and entries[-1].is_ignored


def sources_for(name: str, workspace: Path | str | None = None) -> List[OverrideFile]:
    """Every file that currently overrides this prompt, in precedence order."""
    found: List[OverrideFile] = []
    for entry in override_files(workspace):
        if entry.is_ignored:
            continue
        try:
            if name in entry.read():
                found.append(entry)
        except PromptEditError:
            continue  # reported by the caller if it matters
    return found


def effective_source(name: str, workspace: Path | str | None = None) -> Optional[Path]:
    """The path whose override is the one actually in effect, if any."""
    found = sources_for(name, workspace)
    return found[-1].path if found else None


def write_override(name: str, text: str, workspace: Path | str | None = None,
                  target: Optional[Path] = None) -> Path:
    """Set one prompt's override, and say which file it went to."""
    if target is None and _stdin_is_the_source(workspace):
        # Writing to a file the user is not reading from would look like it
        # worked and change nothing at all, which is the worst outcome here.
        raise PromptEditError(
            "ADAPTIVE_PROMPTS_FILE is '-', so overrides are being read from stdin "
            "and cannot be written. Unset it to edit prompts.")
    # A named override file is a deliberate choice by the user and outranks the
    # usual "edit whatever is in effect" rule -- they pointed the harness at it.
    destination = (target or _named_env_file() or effective_source(name, workspace)
                   or _default_target(workspace))
    if str(destination) == "-":
        raise PromptEditError(
            "ADAPTIVE_PROMPTS_FILE is '-', so overrides are being read from stdin and "
            "cannot be written. Unset it to edit prompts.")
    entry = OverrideFile(destination, "explicit", True)
    data = entry.read()
    data[name] = text
    entry.write(data)
    return destination


def _default_target(workspace: Path | str | None = None) -> Path:
    """Where a *new* override goes when the prompt has none yet.

    Inside a project the override belongs to the project, so it is shared with
    the team rather than being one person's local preference. Outside one, it is
    the user's.
    """
    if workspace:
        return Path(workspace) / ".harness" / "prompts.json"
    return _config_dir() / "prompts.json"


def reset_override(name: str, workspace: Path | str | None = None,
                   *, everywhere: bool = False) -> tuple[bool, str]:
    """Remove a prompt's override. Returns (changed, explanation).

    ``everywhere`` is the default behaviour for a single name, because leaving a
    lower-precedence copy behind would mean the reset silently did nothing. A
    partial reset is available for the case where two files are in play on
    purpose, and it says plainly when the prompt is still overridden.
    """
    changed = False
    files = override_files(workspace)
    for entry in files:
        if entry.is_ignored:
            continue
        try:
            data = entry.read()
        except PromptEditError as exc:
            return False, str(exc)
        if name not in data:
            continue
        del data[name]
        entry.write(data)
        changed = True
        if not everywhere:
            break
    if not changed:
        return False, f"{name} is not overridden."
    remaining = [entry.path for entry in sources_for(name, workspace)]
    if remaining:
        return True, (f"Removed from the highest-precedence file, but {name} is still "
                      f"overridden in: " + ", ".join(str(path) for path in remaining))
    return True, f"Reset {name} to the built-in text."


def reset_all(workspace: Path | str | None = None, *, everywhere: bool = True) -> tuple[int, str]:
    """Remove every override. Returns how many were removed."""
    removed = 0
    for entry in override_files(workspace):
        if entry.is_ignored:
            continue
        try:
            data = entry.read()
        except PromptEditError as exc:
            return removed, str(exc)
        if not data:
            continue
        removed += len(data)
        entry.write({} if everywhere else data)
    if removed == 0:
        return 0, "Nothing was overridden."
    return removed, f"Reset {removed} prompt(s) to the built-in text."


# --- editing in place -------------------------------------------------------

def editor_command() -> List[str]:
    """The editor to open, or an error saying how to choose one."""
    for variable in ("VISUAL", "EDITOR"):
        value = os.environ.get(variable)
        if value:
            return value.split()
    for candidate in ("nano", "vim", "vi"):
        found = shutil.which(candidate)
        if found:
            return [found]
    raise PromptEditError(
        "No editor found. Set $EDITOR (for example EDITOR=nano) and retry.")


def edit_in_editor(path: Path, initial: str) -> str:
    """Open ``path`` in the user's editor with ``initial`` and return the result.

    The text is written to a scratch file first rather than the real one, so a
    cancelled edit -- which is the common case -- cannot truncate the prompt.
    """
    scratch = path.with_name(path.name + f".edit{os.getpid()}")
    scratch.write_text(initial, encoding="utf-8")
    command = editor_command() + [str(scratch)]
    try:
        result = subprocess.run(command)
    except OSError as exc:
        scratch.unlink(missing_ok=True)
        raise PromptEditError(f"Could not run the editor: {exc}") from exc
    finally:
        if scratch.exists():
            text = scratch.read_text(encoding="utf-8")
            scratch.unlink(missing_ok=True)
        else:
            text = initial
    if result.returncode != 0:
        raise PromptEditError(
            f"The editor exited with status {result.returncode}. Nothing was changed.")
    return text


def edit_prompt(name: str, current: str, workspace: Path | str | None = None) -> tuple[Path, bool]:
    """Open a prompt for editing. Returns (where it was written, whether it changed).

    An unchanged prompt is not written: a save with no edit should not create an
    override that then shadows a future upgrade to the built-in text.
    """
    destination = effective_source(name, workspace) or _default_target(workspace)
    edited = edit_in_editor(destination, current)
    if edited == current:
        return destination, False
    write_override(name, edited, workspace, target=destination)
    return destination, True
