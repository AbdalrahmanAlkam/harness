"""Isolation and explicit patch-review safety tests."""

from pathlib import Path
import subprocess

import pytest

from adaptive_harness.workspace.worktree import WorktreeError, WorktreeManager
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.widgets import DiffReviewModal
from textual.widgets import RichLog


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Harness Test")
    git(tmp_path, "config", "user.email", "harness@example.invalid")
    (tmp_path / "app.py").write_text("answer = 1\n")
    git(tmp_path, "add", "app.py")
    git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def test_worktree_isolates_changes_and_applies_reviewed_patch(repo: Path):
    manager = WorktreeManager(repo)
    task = manager.create()
    assert task.path.is_dir() and task.workspace == task.path
    (task.path / "app.py").write_text("answer = 2\n")
    (task.path / "new.py").write_text("created = True\n")
    assert (repo / "app.py").read_text() == "answer = 1\n"
    patch = manager.patch(task)
    assert "+answer = 2" in patch and "new.py" in patch
    assert manager.apply(task).startswith("Reviewed patch")
    assert (repo / "app.py").read_text() == "answer = 2\n"
    assert (repo / "new.py").read_text() == "created = True\n"
    assert not task.path.exists()
    assert task.branch not in git(repo, "branch", "--list", "harness/*")


def test_worktree_abort_removes_artifacts_without_touching_main(repo: Path):
    manager = WorktreeManager(repo)
    task = manager.create()
    (task.path / "app.py").write_text("answer = 999\n")
    manager.abort(task)
    assert (repo / "app.py").read_text() == "answer = 1\n"
    assert git(repo, "status", "--porcelain") == ""
    assert not task.path.exists() and not (repo / ".harness").exists()


def test_review_does_not_apply_over_dirty_main(repo: Path):
    manager = WorktreeManager(repo)
    task = manager.create()
    (task.path / "app.py").write_text("answer = 2\n")
    (repo / "app.py").write_text("answer = 3\n")
    with pytest.raises(WorktreeError, match="uncommitted"):
        manager.apply(task)
    assert (repo / "app.py").read_text() == "answer = 3\n"
    assert task.path.exists()
    manager.abort(task)


def test_agent_tools_are_reanchored_to_isolated_worktree(repo: Path):
    app = AdaptiveHarnessApp(workspace_root=str(repo), db_path=repo.parent / "ui.db",
                             config_dir=repo.parent / "config")
    manager = WorktreeManager(repo)
    task = manager.create()
    app.agent.set_workspace(task.workspace)
    result = app.agent.tools["write_file"].execute(path="app.py", content="answer = 7\n")
    assert result.success
    assert (task.path / "app.py").read_text() == "answer = 7\n"
    assert (repo / "app.py").read_text() == "answer = 1\n"
    app.agent.set_workspace(repo)
    manager.abort(task)


def test_isolation_and_swarm_modes_select_complex_edits(repo: Path):
    app = AdaptiveHarnessApp(workspace_root=str(repo), db_path=repo.parent / "ui.db",
                             config_dir=repo.parent / "config")
    assert not app._should_isolate("read file app.py")
    assert app._should_isolate("Refactor the architecture across modules and add tests")
    assert app._should_swarm("Fix a concurrency deadlock and refactor the architecture")
    app.swarm_mode = "off"
    assert not app._should_swarm("Fix a concurrency deadlock")
    app.isolation_mode = "off"
    assert not app._should_isolate("Refactor the architecture")
    app.isolation_mode = "on"
    assert app._should_isolate("read file app.py")


def test_worktree_refuses_symlinked_storage(repo: Path):
    (repo / ".harness").symlink_to(repo.parent, target_is_directory=True)
    with pytest.raises(WorktreeError, match="symlink"):
        WorktreeManager(repo).create()


@pytest.mark.anyio
async def test_tui_isolated_task_opens_review_and_merge_gate(repo: Path):
    app = AdaptiveHarnessApp(workspace_root=str(repo), db_path=repo.parent / "ui.db",
                             config_dir=repo.parent / "config")
    manager = WorktreeManager(repo)
    task = manager.create()
    (task.path / "app.py").write_text("answer = 42\n")
    patch = manager.patch(task)
    async with app.run_test(size=(115, 30)) as pilot:
        app._worktree_ready(manager, task, patch)
        await pilot.pause()
        assert isinstance(app.screen, DiffReviewModal)
        assert (repo / "app.py").read_text() == "answer = 1\n"
        assert "+answer = 42" in app.screen.patch
        assert "M app.py (+1, -1)" in app.screen._summary()
        await pilot.press("enter")
        await pilot.pause()
        assert (repo / "app.py").read_text() == "answer = 42\n"
        assert app._review_task is None


@pytest.mark.anyio
async def test_tui_review_escape_discards_worktree(repo: Path):
    app = AdaptiveHarnessApp(workspace_root=str(repo), db_path=repo.parent / "ui.db",
                             config_dir=repo.parent / "config")
    manager = WorktreeManager(repo)
    task = manager.create()
    (task.path / "app.py").write_text("answer = 999\n")
    async with app.run_test(size=(115, 30)) as pilot:
        app._worktree_ready(manager, task, manager.patch(task))
        await pilot.pause()
        assert isinstance(app.screen, DiffReviewModal)
        await pilot.press("escape")
        await pilot.pause()
        assert app._review_task is None
        assert (repo / "app.py").read_text() == "answer = 1\n"
        assert not task.path.exists()


@pytest.mark.anyio
async def test_diff_modal_scrolls_large_patch(repo: Path):
    app = AdaptiveHarnessApp(workspace_root=str(repo), db_path=repo.parent / "ui.db",
                             config_dir=repo.parent / "config")
    patch = "diff --git a/app.py b/app.py\n" + "\n".join(f"+line {index}" for index in range(120))
    async with app.run_test(size=(90, 20)) as pilot:
        app.push_screen(DiffReviewModal(patch, "sample"))
        await pilot.pause()
        view = app.screen.query_one("#diff-body", RichLog)
        assert view.can_focus
        await pilot.press("down", "down", "down")
        await pilot.pause()
        assert view.scroll_y > 0
        await pilot.press("escape")


@pytest.mark.anyio
async def test_tui_isolation_and_swarm_commands(repo: Path):
    app = AdaptiveHarnessApp(workspace_root=str(repo), db_path=repo.parent / "ui.db",
                             config_dir=repo.parent / "config")
    async with app.run_test(size=(100, 26)):
        app._handle_slash_command("/isolation on")
        app._handle_slash_command("/swarm off")
        assert app.isolation_mode == "on" and app.swarm_mode == "off"
