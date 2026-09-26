"""Lean 4 prover: valid proof certified, placeholders rejected, errors caught.

Covers the three behaviours the formal prover must guarantee, at the level of
the tool's public surface. Cases that genuinely need the compiler are skipped
when Lean is absent, so the suite still means something on a machine without it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_harness.tools.lean import LeanToolchain, RunLeanProofTool

TOOLCHAIN = LeanToolchain()
needs_lean = pytest.mark.skipif(not TOOLCHAIN.available, reason="Lean 4 is not installed")

# A real arithmetic theorem, proved by computation.
VALID_ARITHMETIC = "theorem two_plus_two : 2 + 2 = 4 := by rfl\n"

# A propositional fact discharged by excluded middle. Lean 4 spells negation
# `Not p`; `~p` is Lean 3 syntax and is a syntax error here.
VALID_PROPOSITIONAL = (
    "theorem tautology (p : Prop) : p \\/ Not p := by\n"
    "  exact Classical.em p\n")

# The escape hatches. Lean exits 0 on all three, so only the harness can refuse
# them: `sorry` and `admit` are the same escape hatch, and a bare `axiom` is an
# assumption dressed as a proof.
SORRY_PROOF = "theorem bad : 1 = 1 := by sorry\n"
ADMIT_PROOF = "theorem bad : 1 = 1 := by admit\n"
AXIOM_PROOF = "axiom cheat : 1 = 2\n\ntheorem bad : 1 = 2 := cheat\n"

# Malformed, and a genuine unsolved goal.
SYNTAX_ERROR = "theorem broken : := \n"
UNSOLVED_GOAL = (
    "theorem wrong (n : Nat) : n + 0 = n + 1 := by\n"
    "  induction n with\n"
    "  | zero => rfl\n")


@needs_lean
def test_valid_theorem_passes_and_writes_a_verified_receipt(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(lean_code=VALID_ARITHMETIC, theorem_name="TwoPlusTwo")
    assert result.success, result.error
    assert result.metadata["exit_code"] == 0

    receipt_path = Path(result.metadata["receipt_path"])
    assert receipt_path.is_file()
    assert receipt_path.name == "TwoPlusTwo.lean.receipt.json"

    receipt = json.loads(receipt_path.read_text())
    assert receipt["verified"] is True
    assert receipt["status"] == "LEAN_VERIFIED"
    assert receipt["exit_code"] == 0
    assert receipt["theorem_id"] == "TwoPlusTwo"
    assert receipt["sha256"].startswith("sha256:")
    assert receipt["timestamp"]
    assert receipt["placeholders"] == []


@needs_lean
def test_propositional_theorem_is_accepted(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(lean_code=VALID_PROPOSITIONAL, theorem_name="Tautology")
    assert result.success, result.error
    receipt = json.loads(Path(result.metadata["receipt_path"]).read_text())
    assert receipt["verified"] is True
    # Classical.choice is a legitimate modelling decision, not a hole.
    assert "sorryAx" not in {item["axiom"] for item in receipt["axioms"]}


@needs_lean
@pytest.mark.parametrize("source,token", [
    (SORRY_PROOF, "sorry"),
    (ADMIT_PROOF, "admit"),
    (AXIOM_PROOF, "axiom"),
])
def test_placeholders_are_rejected_with_a_failed_receipt(tmp_path: Path, source, token):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(lean_code=source, theorem_name="Dishonest")
    assert not result.success
    assert token in (result.error or "")

    receipt = json.loads(Path(result.metadata["receipt_path"]).read_text())
    assert receipt["verified"] is False
    assert receipt["status"] == "LEAN_REJECTED"
    assert [item["token"] for item in receipt["placeholders"]] == [token]


@needs_lean
def test_syntax_error_is_caught_cleanly(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(lean_code=SYNTAX_ERROR, theorem_name="Broken")
    assert not result.success
    assert result.metadata["exit_code"] == 1
    receipt = json.loads(Path(result.metadata["receipt_path"]).read_text())
    assert receipt["verified"] is False
    # The diagnostic must be located, not merely reported as a failure.
    assert receipt["errors"]
    assert receipt["errors"][0]["line"] >= 1


@needs_lean
def test_unsolved_goal_is_reported_with_position(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(lean_code=UNSOLVED_GOAL, theorem_name="Unclosed")
    assert not result.success
    receipt = json.loads(Path(result.metadata["receipt_path"]).read_text())
    assert receipt["verified"] is False
    assert receipt["errors"] or receipt["unsolved_goals"]


@needs_lean
def test_a_proof_in_the_proofs_lean_directory_is_placed_there(tmp_path: Path):
    """Inline code lands in proofs/lean/, keeping the two tiers separate."""
    tool = RunLeanProofTool(workspace_root=tmp_path, lean_dir="proofs/lean")
    result = tool.execute(lean_code=VALID_ARITHMETIC, theorem_name="Placed")
    assert result.success, result.error
    assert (tmp_path / "proofs" / "lean" / "Placed.lean").is_file()
    assert (tmp_path / "proofs" / "lean" / "Placed.lean.receipt.json").is_file()


@needs_lean
def test_an_existing_file_can_be_verified_in_place(tmp_path: Path):
    target = tmp_path / "proofs" / "lean" / "Existing.lean"
    target.parent.mkdir(parents=True)
    target.write_text(VALID_ARITHMETIC)
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(file_path=str(target))
    assert result.success, result.error
    assert result.metadata["receipt_path"] == str(target) + ".receipt.json"


@needs_lean
def test_the_timeout_is_honoured(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(lean_code=VALID_ARITHMETIC, theorem_name="Timed", timeout_s=120)
    assert result.success, result.error
    assert tool.verifier.timeout_s == 120


def test_missing_lean_is_reported_without_crashing(tmp_path: Path, monkeypatch):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    monkeypatch.setattr(tool.verifier.toolchain, "lean", None)
    result = tool.execute(lean_code=VALID_ARITHMETIC)
    assert not result.success
    assert result.metadata["lean_available"] is False
    assert "elan" in (result.error or "")
