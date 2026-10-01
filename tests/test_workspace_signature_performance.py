"""The shell's mutation evidence must stay correct, and must stop being slow.

Every `run_bash` and `run_pytest` call brackets the command with a workspace
signature so the harness can tell a command that changed the tree from one that
merely printed. That check ran twice per shell call and walked the whole tree
each time, which was the largest fixed cost on the agent's hot path.

The optimisation is only legitimate if it still detects what it is supposed to
detect, so the tests below pin both halves: the fast path stays correct for
every kind of change, and the slow path is genuinely not reached when nothing
happened.
"""

from __future__ import annotations

import time
from pathlib import Path

from adaptive_harness.agent.agent import (
    _workspace_changed,
    _workspace_quick_stamp,
    _workspace_signature,
    _walk_workspace,
    _WORKSPACE_SKIP_DIRS,
)


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("original", encoding="utf-8")
    return tmp_path


def _before(root: Path):
    return _workspace_signature(root), _workspace_quick_stamp(root)


# --- correctness first: an undetected change would under-report a mutation ---


def test_nothing_changed_is_not_reported_as_a_change(tmp_path: Path):
    root = _workspace(tmp_path)
    signature, stamp = _before(root)
    assert _workspace_changed(root, signature, stamp) is False


def test_a_new_file_is_detected(tmp_path: Path):
    root = _workspace(tmp_path)
    signature, stamp = _before(root)
    (root / "sub" / "b.py").write_text("new", encoding="utf-8")
    assert _workspace_changed(root, signature, stamp) is True


def test_a_new_nested_file_is_detected(tmp_path: Path):
    root = _workspace(tmp_path)
    signature, stamp = _before(root)
    (root / "deep" / "deeper").mkdir(parents=True)
    (root / "deep" / "deeper" / "c.py").write_text("new", encoding="utf-8")
    assert _workspace_changed(root, signature, stamp) is True


def test_a_new_empty_directory_is_detected(tmp_path: Path):
    """A directory with no files in it is still something the agent created."""
    root = _workspace(tmp_path)
    signature, stamp = _before(root)
    (root / "fresh").mkdir()
    assert _workspace_changed(root, signature, stamp) is True


def test_a_deletion_is_detected(tmp_path: Path):
    root = _workspace(tmp_path)
    signature, stamp = _before(root)
    (root / "sub" / "a.py").unlink()
    assert _workspace_changed(root, signature, stamp) is True


def test_deleting_a_whole_directory_is_detected(tmp_path: Path):
    import shutil

    root = _workspace(tmp_path)
    signature, stamp = _before(root)
    shutil.rmtree(root / "sub")
    assert _workspace_changed(root, signature, stamp) is True


# --- and the performance claim, so the fast path cannot silently regress ---


def test_the_unchanged_case_avoids_the_full_tree_walk(tmp_path: Path, monkeypatch):
    """The whole point: no deep walk when nothing appeared or disappeared."""
    root = _workspace(tmp_path)
    signature, stamp = _before(root)

    def explode(_root: Path):
        raise AssertionError("the full signature was recomputed for an unchanged workspace")

    monkeypatch.setattr("adaptive_harness.agent.agent._workspace_signature", explode)
    assert _workspace_changed(root, signature, stamp) is False


def test_dependency_directories_are_pruned_not_merely_filtered(tmp_path: Path):
    """`rglob` descends into .venv and node_modules and discards them
    afterwards, so the skip list saved stat calls but not traversal."""
    (tmp_path / ".venv" / "lib" / "nested").mkdir(parents=True)
    for index in range(50):
        (tmp_path / ".venv" / "lib" / "nested" / f"f{index}.py").write_text("x", encoding="utf-8")
    (tmp_path / "real.py").write_text("y", encoding="utf-8")

    walked = [path for path in _walk_workspace(root=tmp_path)]
    assert all(part not in _WORKSPACE_SKIP_DIRS for path in walked for part in path.parts)
    assert sorted(_workspace_signature(tmp_path)) == ["real.py"]


def test_the_unchanged_check_is_materially_cheaper_than_the_walk(tmp_path: Path):
    """Guards the optimisation against being quietly reverted."""
    root = _workspace(tmp_path)
    for index in range(200):
        (root / f"file{index}.py").write_text("x", encoding="utf-8")
    signature, stamp = _before(root)

    start = time.perf_counter()
    for _ in range(20):
        assert _workspace_changed(root, signature, stamp) is False
    fast = (time.perf_counter() - start) / 20

    start = time.perf_counter()
    for _ in range(5):
        _workspace_signature(root)
    slow = (time.perf_counter() - start) / 5

    assert fast < slow, f"the fast path ({fast:.4f}s) is not cheaper than the walk ({slow:.4f}s)"


def test_the_signature_records_file_content_state(tmp_path: Path):
    """The recorded evidence must be the file's own mtime and size, since that
    is what a later comparison is trusted to mean."""
    root = _workspace(tmp_path)
    signature = _workspace_signature(root)
    assert "sub/a.py" in signature
    before = signature["sub/a.py"]
    (root / "sub" / "a.py").write_text("much longer content than before", encoding="utf-8")
    after = _workspace_signature(root)["sub/a.py"]
    assert after != before
