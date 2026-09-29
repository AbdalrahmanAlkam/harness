"""Regressions for paths where the harness could assert something untrue.

Every test here corresponds to a defect that was reproduced before it was fixed.
They are grouped by the failure mode they guard, because the shared theme is
that a research tool must fail closed: an unverified result reported as verified
is worse than an error, because a reader cannot tell the difference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.agent.agent import _synthesize_missing_tool_results
from adaptive_harness.research.paper import typst_raw_block
from adaptive_harness.tools.bash import RunBashTool
from adaptive_harness.tools.file_ops import EditFileTool
from adaptive_harness.tools.lean import LeanVerifier, RunLeanProofTool


class _Call:
    def __init__(self, call_id: str, name: str = "run_bash") -> None:
        self.id = call_id
        self.name = name


# --- the tool loop must not leave a declared tool_call unanswered ----------


def test_an_unanswered_tool_call_is_backfilled():
    """A dangling tool_call makes every later request in the session fail.

    The assistant message declares every call in a batch up front. An overseer
    stop or a clarification abort can exit the loop early, and OpenAI-compatible
    endpoints reject the whole request when a declared call has no result.
    """
    calls = [_Call("c1"), _Call("c2"), _Call("c3")]
    synthesized = _synthesize_missing_tool_results(calls, {"c1"})
    assert [item["tool_call_id"] for item in synthesized] == ["c2", "c3"]
    assert all(item["role"] == "tool" for item in synthesized)
    assert all("CANCELLED" in item["content"] for item in synthesized)


def test_a_fully_answered_batch_gets_no_synthetic_results():
    calls = [_Call("c1"), _Call("c2")]
    assert _synthesize_missing_tool_results(calls, {"c1", "c2"}) == []


# --- a worker must not be able to write outside its lease ------------------


def test_edit_file_refuses_a_path_outside_the_allowlist(tmp_path: Path):
    owned = tmp_path / "proofs" / "prop-01.py"
    sibling = tmp_path / "proofs" / "prop-02.py"
    owned.parent.mkdir(parents=True)
    owned.write_text("original\n", encoding="utf-8")
    sibling.write_text("sibling artifact\n", encoding="utf-8")

    tool = EditFileTool(workspace_root=tmp_path, allowed_paths=["proofs/prop-01.py"])
    result = tool.execute(path="proofs/prop-02.py", target_text="sibling",
                          replacement_text="hijacked")
    assert not result.success
    assert "Write refused" in (result.error or "")
    assert sibling.read_text(encoding="utf-8") == "sibling artifact\n"

    allowed = tool.execute(path="proofs/prop-01.py", target_text="original",
                           replacement_text="mine")
    assert allowed.success, allowed.error


def test_a_worker_shell_cannot_redirect_onto_a_sibling_artifact(tmp_path: Path):
    (tmp_path / "proofs").mkdir()
    (tmp_path / "proofs" / "prop-01.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "proofs" / "prop-02.py").write_text("print(2)\n", encoding="utf-8")
    tool = RunBashTool(workspace_root=tmp_path, allowed_write_paths=["proofs/prop-01.py"])

    for command in ("cat > proofs/prop-02.py", "echo x > proofs/prop-02.py",
                    "cp proofs/prop-01.py proofs/prop-02.py",
                    "mv proofs/prop-01.py proofs/prop-02.py",
                    "tee proofs/prop-02.py"):
        result = tool.execute(command, timeout_seconds=5)
        assert not result.success, f"{command!r} should have been refused"
    assert (tmp_path / "proofs" / "prop-02.py").read_text(encoding="utf-8") == "print(2)\n"


def test_a_worker_can_still_execute_its_own_artifact(tmp_path: Path):
    """The write allow-list must not stop a worker running its own decider."""
    (tmp_path / "proofs").mkdir()
    (tmp_path / "proofs" / "prop-01.py").write_text("print(42)\n", encoding="utf-8")
    tool = RunBashTool(workspace_root=tmp_path, allowed_write_paths=["proofs/prop-01.py"])

    assert tool.execute("python proofs/prop-01.py", timeout_seconds=30).success
    assert tool.execute("cat proofs/prop-01.py", timeout_seconds=10).success
    assert tool.execute("echo done > /dev/null", timeout_seconds=10).success


# --- the Lean tool must stay inside the workspace --------------------------


def test_lean_proof_cannot_read_outside_the_workspace(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.lean").write_text("theorem t : 1 = 1 := rfl\n", encoding="utf-8")

    tool = RunLeanProofTool(workspace_root=workspace, lean_dir="proofs")
    for path in (str(outside / "secret.lean"), "../outside/secret.lean"):
        result = tool.execute(file_path=path)
        assert not result.success
    assert not list(outside.glob("*.receipt.json"))


def test_lean_proof_cannot_overwrite_a_file_outside_the_workspace(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.txt"
    victim.write_text("ORIGINAL", encoding="utf-8")

    tool = RunLeanProofTool(workspace_root=workspace, lean_dir="proofs")
    result = tool.execute(lean_code="theorem pwn : True := trivial\n",
                          theorem_name=str(outside / "victim.lean"))
    assert not result.success
    assert victim.read_text(encoding="utf-8") == "ORIGINAL"


def test_an_empty_lean_file_is_not_certified(tmp_path: Path):
    """Lean exits 0 on a file with no declaration, so exit code alone lies."""
    empty = tmp_path / "Empty.lean"
    empty.write_text("", encoding="utf-8")
    assert not LeanVerifier().verify(empty).success

    comments = tmp_path / "Comments.lean"
    comments.write_text("-- just a comment\n", encoding="utf-8")
    assert not LeanVerifier().verify(comments).success


# --- model-authored text must not become Typst markup ---------------------


def test_a_quote_in_a_raw_block_cannot_close_the_string_literal():
    """A model-authored source containing `"` used to escape into markup.

    The payload could then typeset a forged `#lean-box` certification badge in a
    paper that never ran Lean on the claim.
    """
    hostile = 'theorem t : Nat := 1\n#raw("x") as a string")\n= FAKE'
    rendered = typst_raw_block(hostile)
    # Only the two structural quotes are unescaped: the opener and the closer.
    # Every quote from the payload is emitted as \", so it cannot terminate the
    # literal and hand the remainder to the Typst parser.
    assert rendered.count('\\"') == 3
    assert rendered.count('"') - rendered.count('\\"') == 2
    assert rendered.startswith('#raw(block: true, lang: none, "')
    assert rendered.endswith('")')
