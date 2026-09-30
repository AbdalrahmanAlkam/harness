"""Editing and resetting prompt overrides.

An override lives in a JSON file, and the interesting part is *which* file:
overrides merge from three places with later winning, so "reset this prompt"
has a trap. Removing it from the file you happen to be looking at does not
un-override it if another file also overrides the same name. A reset that
quietly did nothing is worse than one that says so.

The tests use real temp directories and a scripted editor -- no mocking of the
file layer, because the atomicity and the precedence are the point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_harness.prompts import PromptRegistry
from adaptive_harness.prompts_edit import (
    OverrideFile,
    PromptEditError,
    effective_source,
    edit_prompt,
    override_files,
    reset_all,
    reset_override,
    sources_for,
    write_override,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A project directory, with no user config and no env overrides."""
    project = tmp_path / "project"
    (project / ".harness").mkdir(parents=True)
    return project


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No real user config, no env override: only the project file is in play.

    The config directory is patched where it is *defined*, since the module
    reads it at call time rather than binding it at import.
    """
    from adaptive_harness.data import config

    monkeypatch.setattr(config, "DEFAULT_CONFIG_DIR", tmp_path / "user-config")
    monkeypatch.delenv("ADAPTIVE_PROMPTS_FILE", raising=False)


def _prompt_file(workspace: Path) -> Path:
    return workspace / ".harness" / "prompts.json"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# --- where an override goes -------------------------------------------------


def test_an_edit_inside_a_project_lands_in_the_project(workspace: Path):
    """A project's overrides belong to the project, so the team shares them
    rather than one person carrying a local preference."""
    written = write_override("system.default", "CHANGED", workspace)
    assert written == _prompt_file(workspace)
    assert json.loads(written.read_text())["system.default"] == "CHANGED"


def test_an_edit_outside_a_project_lands_in_the_user_config(tmp_path: Path):
    written = write_override("system.default", "CHANGED", None)
    assert written.name == "prompts.json", (
        "with no project, the override must still be written somewhere that persists")
    assert written.exists()


def test_an_existing_override_is_replaced_in_place(workspace: Path):
    _write(_prompt_file(workspace), {"system.default": "OLD", "other.prompt": "KEEP"})
    write_override("system.default", "NEW", workspace)
    data = json.loads(_prompt_file(workspace).read_text())
    assert data["system.default"] == "NEW"
    assert data["other.prompt"] == "KEEP", "an unrelated prompt was dropped"


def test_the_file_is_private(tmp_path: Path):
    """It can hold a house style or an internal instruction, and is 0600 like
    every other file the harness writes for the user."""
    written = write_override("system.default", "CHANGED", None)
    assert written.stat().st_mode & 0o077 == 0


# --- precedence -------------------------------------------------------------


def test_the_effective_source_is_the_highest_precedence_copy(workspace: Path, tmp_path: Path):
    user_file = tmp_path / "user-config" / "prompts.json"
    _write(user_file, {"system.default": "FROM USER"})
    _write(_prompt_file(workspace), {"system.default": "FROM PROJECT"})

    assert effective_source("system.default", workspace) == _prompt_file(workspace)
    assert len(sources_for("system.default", workspace)) == 2


def test_an_unoverridden_prompt_has_no_source(workspace: Path):
    assert effective_source("system.default", workspace) is None
    assert sources_for("system.default", workspace) == []


def test_editing_an_overridden_prompt_targets_the_copy_in_effect(workspace: Path, tmp_path: Path):
    """The copy that is actually in effect is the one edited, not whichever file
    happens to exist -- otherwise the edit looks like it worked and the old
    override still wins."""
    _write(tmp_path / "user-config" / "prompts.json", {"system.default": "FROM USER"})
    _write(_prompt_file(workspace), {"system.default": "FROM PROJECT"})
    written = write_override("system.default", "NEW", workspace)
    assert written == _prompt_file(workspace), (
        "editing must not silently write to a file that is not in effect")
    # And the losing copy is untouched, so the precedence stays honest.
    assert json.loads((tmp_path / "user-config" / "prompts.json").read_text())[
        "system.default"] == "FROM USER"


# --- reset ------------------------------------------------------------------


def test_reset_restores_the_built_in_text(workspace: Path):
    write_override("system.default", "CHANGED", workspace)
    changed, message = reset_override("system.default", workspace)
    assert changed
    assert "Reset" in message
    assert json.loads(_prompt_file(workspace).read_text()) == {}


def test_resetting_something_not_overridden_says_so(workspace: Path):
    changed, message = reset_override("system.default", workspace)
    assert not changed
    assert "not overridden" in message


def test_reset_clears_every_copy_and_says_when_one_remains(workspace: Path, tmp_path: Path):
    """The trap: two files override the same name, so removing one does not
    un-override it. Reporting success there would be a lie."""
    _write(tmp_path / "user-config" / "prompts.json", {"system.default": "FROM USER"})
    _write(_prompt_file(workspace), {"system.default": "FROM PROJECT"})

    changed, message = reset_override("system.default", workspace)
    assert changed
    assert "still overridden" in message, (
        "a partial reset was reported as though the prompt were restored")
    assert "still overridden" in message


def test_reset_clears_every_file_when_asked(workspace: Path, tmp_path: Path):
    _write(tmp_path / "user-config" / "prompts.json", {"system.default": "FROM USER"})
    _write(_prompt_file(workspace), {"system.default": "FROM PROJECT"})
    changed, message = reset_override("system.default", workspace, everywhere=True)
    assert changed and "still overridden" not in message
    assert sources_for("system.default", workspace) == []


def test_reset_all_clears_both_files(workspace: Path, tmp_path: Path):
    _write(tmp_path / "user-config" / "prompts.json", {"a": "1", "b": "2"})
    _write(_prompt_file(workspace), {"c": "3"})
    count, message = reset_all(workspace)
    assert count == 3
    assert json.loads((tmp_path / "user-config" / "prompts.json").read_text()) == {}
    assert json.loads(_prompt_file(workspace).read_text()) == {}


def test_reset_all_on_a_clean_install_says_nothing_was_overridden(workspace: Path):
    count, message = reset_all(workspace)
    assert count == 0
    assert "Nothing" in message


# --- editing in place -------------------------------------------------------


def test_edit_saves_a_changed_prompt(workspace: Path, monkeypatch: pytest.MonkeyPatch):
    script = workspace / "editor.sh"
    script.write_text('#!/bin/sh\nprintf "EDITED TEXT" > "$1"\n', encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(script))

    destination, changed = edit_prompt("system.default", "ORIGINAL", workspace)
    assert changed
    assert json.loads(destination.read_text())["system.default"] == "EDITED TEXT"


def test_an_unchanged_edit_writes_no_override(workspace: Path, monkeypatch: pytest.MonkeyPatch):
    """A save with no edit must not shadow a future upgrade to the built-in."""
    script = workspace / "editor.sh"
    script.write_text('#!/bin/sh\ncat "$1" > /dev/null\n', encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(script))

    _destination, changed = edit_prompt("system.default", "ORIGINAL", workspace)
    assert not changed
    assert not _prompt_file(workspace).exists(), (
        "an override was written for a prompt nobody changed")


def test_a_failed_editor_changes_nothing(workspace: Path, monkeypatch: pytest.MonkeyPatch):
    """The common case is a cancelled edit, and it must not truncate the prompt."""
    script = workspace / "editor.sh"
    script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(script))

    with pytest.raises(PromptEditError):
        edit_prompt("system.default", "ORIGINAL", workspace)
    assert not _prompt_file(workspace).exists()


def test_no_editor_at_all_is_reported_rather_than_crashing(workspace: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(PromptEditError) as excinfo:
        edit_prompt("system.default", "ORIGINAL", workspace)
    assert "EDITOR" in str(excinfo.value)


# --- a broken override file -------------------------------------------------


def test_a_corrupt_override_file_is_reported_rather_than_ignored(workspace: Path):
    """Silently ignoring it would mean the prompts the user wrote silently stop
    applying, and the built-in text silently comes back."""
    _prompt_file(workspace).write_text("{not json", encoding="utf-8")
    entry = OverrideFile(_prompt_file(workspace), "project", True)
    with pytest.raises(PromptEditError) as excinfo:
        entry.read()
    assert "not valid JSON" in str(excinfo.value)


def test_a_reset_against_a_corrupt_file_says_so(workspace: Path):
    _prompt_file(workspace).write_text("{not json", encoding="utf-8")
    changed, message = reset_override("system.default", workspace)
    assert not changed
    assert "not valid JSON" in message


# --- the registry agrees with what the editor wrote -------------------------


def test_an_edited_prompt_is_what_the_registry_serves(workspace: Path):
    """The point of the whole command: change the text, and the model gets the
    changed text."""
    write_override("system.default", "CUSTOM SYSTEM PROMPT", workspace)
    assert PromptRegistry.for_workspace(workspace).get("system.default") == "CUSTOM SYSTEM PROMPT"


def test_after_a_reset_the_registry_serves_the_built_in_text(workspace: Path):
    built_in = PromptRegistry().get("system.default")
    write_override("system.default", "CUSTOM", workspace)
    reset_override("system.default", workspace)
    assert PromptRegistry.for_workspace(workspace).get("system.default") == built_in


def test_the_project_override_is_found_from_the_command_line_root(workspace: Path):
    """`prompts list` runs from the project directory, so it must see the
    project's own overrides rather than the user-only view."""
    write_override("system.default", "PROJECT TEXT", workspace)
    registry = PromptRegistry.for_workspace(workspace)
    assert registry.is_overridden("system.default")
    assert registry.get("system.default") == "PROJECT TEXT"


def test_stdin_overrides_cannot_be_written_to(workspace: Path, monkeypatch: pytest.MonkeyPatch):
    """ADAPTIVE_PROMPTS_FILE=- reads from stdin, so writing there is refused
    rather than silently doing nothing."""
    monkeypatch.setenv("ADAPTIVE_PROMPTS_FILE", "-")
    assert any(entry.is_ignored for entry in override_files(workspace))
    with pytest.raises(PromptEditError) as excinfo:
        write_override("system.default", "X", workspace)
    assert "stdin" in str(excinfo.value)


def test_an_env_override_file_is_a_valid_target(workspace: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    env_file = tmp_path / "from-env.json"
    monkeypatch.setenv("ADAPTIVE_PROMPTS_FILE", str(env_file))
    write_override("system.default", "ENV TEXT", workspace)
    assert json.loads(env_file.read_text())["system.default"] == "ENV TEXT"
