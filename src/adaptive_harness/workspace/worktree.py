"""Create disposable linked worktrees and apply reviewed patches safely."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import uuid


class WorktreeError(RuntimeError):
    """A worktree operation could not be completed safely."""


@dataclass(frozen=True)
class WorktreeTask:
    id: str
    branch: str
    path: Path
    workspace: Path
    base_commit: str


class WorktreeManager:
    """Keep agent edits separate until an explicit review decision.

    ``workspace`` may be a repository subdirectory; tools receive the matching
    subdirectory in the linked worktree. A merge applies the reviewed patch to
    the main checkout without silently committing or rewriting its history.
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.repo = self._git("rev-parse", "--show-toplevel", cwd=self.workspace).stdout.strip()
        if not self.repo:
            raise WorktreeError("The workspace is not inside a Git repository")
        self.repo_root = Path(self.repo).resolve()
        self.relative_workspace = self.workspace.relative_to(self.repo_root)
        self.worktrees_dir = self.repo_root / ".harness" / "worktrees"

    @staticmethod
    def _git(*args: str, cwd: Path, input_data: str | None = None) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(["git", "-C", str(cwd), *args], input=input_data,
                                    text=True, capture_output=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorktreeError(f"Git operation failed: {exc}") from exc
        if result.returncode:
            raise WorktreeError((result.stderr or result.stdout).strip() or "Git operation failed")
        return result

    def create(self) -> WorktreeTask:
        if self.worktrees_dir.parent.is_symlink() or self.worktrees_dir.is_symlink():
            raise WorktreeError("The .harness worktree directory must not be a symlink")
        task_id = uuid.uuid4().hex[:10]
        branch = f"harness/task-{task_id}"
        path = self.worktrees_dir / f"task-{task_id}"
        base_commit = self._git("rev-parse", "HEAD", cwd=self.repo_root).stdout.strip()
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._git("worktree", "add", "-b", branch, str(path), base_commit, cwd=self.repo_root)
        except Exception:
            # A timed-out or interrupted Git process may leave a branch or a
            # partially registered worktree. Only clean up our unique task ID.
            for command in (("worktree", "remove", "--force", str(path)), ("branch", "-D", branch)):
                try:
                    subprocess.run(["git", "-C", str(self.repo_root), *command],
                                   capture_output=True, timeout=15, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._prune_empty_dirs()
            raise
        return WorktreeTask(task_id, branch, path, path / self.relative_workspace, base_commit)

    def patch(self, task: WorktreeTask) -> str:
        self._validate(task)
        # The index is private to this worktree. Staging captures added and
        # deleted files as well as tracked edits, including binary changes.
        self._git("add", "-A", cwd=task.path)
        return self._git("diff", "--cached", "--binary", "--no-color", "--no-ext-diff", "HEAD", cwd=task.path).stdout

    def apply(self, task: WorktreeTask) -> str:
        patch = self.patch(task)
        if not patch:
            self.abort(task)
            return "No changes to apply"
        # Applying a patch against an already dirty checkout risks clobbering
        # user work. Require a clean main checkout and unchanged HEAD.
        current = self._git("rev-parse", "HEAD", cwd=self.repo_root).stdout.strip()
        if current != task.base_commit:
            raise WorktreeError("The main branch advanced during review; resolve it before applying this patch")
        status = self._git("status", "--porcelain", "--untracked-files=normal", "--", ".",
                           ":(exclude).harness/worktrees", cwd=self.repo_root).stdout
        if status.strip():
            raise WorktreeError("The main workspace has uncommitted changes; clean it before applying the reviewed patch")
        self._git("apply", "--check", "--binary", "-", cwd=self.repo_root, input_data=patch)
        self._git("apply", "--binary", "-", cwd=self.repo_root, input_data=patch)
        self.abort(task)
        return "Reviewed patch applied to the main workspace"

    def abort(self, task: WorktreeTask) -> None:
        self._validate(task)
        self._git("worktree", "remove", "--force", str(task.path), cwd=self.repo_root)
        self._git("branch", "-D", task.branch, cwd=self.repo_root)
        self._prune_empty_dirs()

    def _validate(self, task: WorktreeTask) -> None:
        if task.path.parent != self.worktrees_dir or not task.path.name.startswith("task-"):
            raise WorktreeError("Task does not belong to this worktree manager")
        if not task.path.is_dir():
            raise WorktreeError("Task worktree no longer exists")

    def _prune_empty_dirs(self) -> None:
        for path in (self.worktrees_dir, self.worktrees_dir.parent):
            try:
                path.rmdir()
            except OSError:
                break
