"""Tests for the Lean 4 machine-checked verification engine.

The suite is written so that the important cases do not require Lean to be
installed: detection, parsing, and adjudication are pure functions and are
always exercised. The cases that genuinely need the compiler are skipped when
no toolchain is present, so a machine without Lean still gets a meaningful run.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.tools.lean import (BENIGN_AXIOMS, LeanDiagnostic, LeanError,
                                         LeanToolchain, LeanVerification, LeanVerifier,
                                         RunLeanProofTool, axiom_audit_source,
                                         declared_names, find_placeholders,
                                         strip_lean_comments)
from adaptive_harness.research.lean_gate import LeanProofGate, theorems_in

TOOLCHAIN = LeanToolchain()
needs_lean = pytest.mark.skipif(not TOOLCHAIN.available,
                                reason="Lean 4 is not installed on this machine")

HONEST_SORRY = "theorem bad : 1 = 1 := by sorry\n"
FALSE_THEOREM = "theorem bad : 1 = 2 := by rfl\n"
VALID_THEOREM = "theorem good : 2 + 2 = 4 := by rfl\n"

# A core-only Gauss proof, verified below. It states the sum in doubling form to
# avoid Nat division, and expands the product explicitly because core Lean has no
# `ring` tactic.
GAUSS_LEAN = '''/-- Sum of the first `n` natural numbers. -/
def sumFirst : Nat → Nat
  | 0 => 0
  | n + 1 => sumFirst n + (n + 1)

/-- Distributing the product twice, then commutativity. -/
theorem expand (k : Nat) : (k + 1) * ((k + 1) + 1) = k * (k + 1) + (k + 1) + (k + 1) := by
  have hA : (k + 1) * ((k + 1) + 1) = (k + 1) * (k + 1) + (k + 1) * 1 := Nat.mul_add _ _ _
  have hB : (k + 1) * 1 = k + 1 := Nat.mul_one (k + 1)
  have hC : (k + 1) * (k + 1) = (k + 1) * k + (k + 1) * 1 := Nat.mul_add (k + 1) k 1
  have hD : (k + 1) * k = k * (k + 1) := (Nat.mul_comm k (k + 1)).symm
  simp only [hA, hB, hC, hD]

/-- Gauss's sum, in the form that avoids division. -/
theorem gauss_two_mul (n : Nat) : 2 * sumFirst n = n * (n + 1) := by
  induction n with
  | zero => rfl
  | succ k ih =>
    calc 2 * (sumFirst k + (k + 1)) = 2 * sumFirst k + 2 * (k + 1) := by omega
      _ = k * (k + 1) + 2 * (k + 1) := by rw [ih]
      _ = k * (k + 1) + (k + 1) + (k + 1) := by omega
      _ = (k + 1) * ((k + 1) + 1) := (expand k).symm
      _ = (k + 1) * (k + 2) := rfl
'''


# -- placeholder detection -------------------------------------------------

def test_placeholders_in_prose_comments_are_ignored():
    assert find_placeholders("-- we avoid sorry here\ntheorem t : 1 = 1 := rfl\n") == []
    assert find_placeholders("/- sorry -/\ntheorem t : 1 = 1 := rfl\n") == []
    assert find_placeholders('/- outer /- inner sorry -/ still -/\ntheorem t := rfl\n') == []


def test_placeholders_inside_strings_are_ignored():
    assert find_placeholders('def msg := "please do not use sorry"\n') == []


def test_real_placeholders_are_found_with_line_numbers():
    assert find_placeholders(HONEST_SORRY) == [(1, "sorry")]
    assert find_placeholders("theorem t : 1 = 1 := by\n  admit\n") == [(2, "admit")]
    hits = find_placeholders("theorem a : 1 = 1 := by sorry\ntheorem b : 1 = 1 := by sorry\n")
    assert [line for line, _ in hits] == [1, 2]


def test_identifiers_containing_a_token_are_not_placeholders():
    assert find_placeholders("theorem sorry_note : 1 = 1 := rfl\n") == []
    assert find_placeholders("def admitted : Nat := 3\n") == []


def test_comment_stripping_preserves_structure():
    stripped = strip_lean_comments("theorem t : Nat := 1 -- trailing\n/- block -/\ndef u := 2\n")
    assert "theorem t" in stripped and "def u" in stripped
    assert "trailing" not in stripped and "block" not in stripped


def test_declared_names_are_extracted_for_the_axiom_audit():
    names = declared_names("theorem foo : True := trivial\nlemma bar : True := trivial\n"
                           "def baz : Nat := 1\nstructure S where\n  x : Nat\n")
    assert {"foo", "bar", "baz"} <= set(names)


def test_axiom_audit_source_appends_print_axioms():
    audited = axiom_audit_source("theorem foo : True := trivial\n")
    assert "#print axioms foo" in audited
    # A source with no declarations is returned unchanged.
    assert axiom_audit_source("-- just a comment\n") == "-- just a comment\n"


# -- toolchain discovery ---------------------------------------------------

def test_toolchain_reports_a_missing_lean_clearly(monkeypatch, tmp_path: Path):
    empty = LeanToolchain()
    empty.lean = None
    with pytest.raises(LeanError) as excinfo:
        empty.require()
    assert "elan" in str(excinfo.value)


def test_toolchain_prefers_lake_env_when_a_lakefile_exists(tmp_path: Path):
    (tmp_path / "lakefile.lean").write_text("package x\n")
    source = tmp_path / "Proof.lean"
    source.write_text(VALID_THEOREM)
    toolchain = LeanToolchain()
    toolchain.lake = toolchain.lake or "/usr/bin/lake"
    toolchain.lean = toolchain.lean or "/usr/bin/lean"
    command = toolchain.command_for(source)
    assert command[0].endswith("lake")
    assert command[1] == "env"


def test_toolchain_uses_plain_lean_without_a_lakefile(tmp_path: Path):
    source = tmp_path / "Proof.lean"
    source.write_text(VALID_THEOREM)
    toolchain = LeanToolchain()
    toolchain.lean = toolchain.lean or "/usr/bin/lean"
    toolchain.lake = None
    command = toolchain.command_for(source)
    assert command[0].endswith("lean")
    assert len(command) == 2


# -- the tool contract -----------------------------------------------------

def test_tool_requires_source_or_path(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    if not TOOLCHAIN.available:
        pytest.skip("Lean 4 is not installed")
    result = tool.execute()
    assert not result.success
    assert "source" in (result.error or "")


def test_missing_file_is_reported_not_crashed(tmp_path: Path):
    verifier = LeanVerifier(TOOLCHAIN)
    result = verifier.verify(tmp_path / "absent.lean")
    assert not result.success
    assert "does not exist" in (result.error or "")


# -- compilation (requires Lean) ------------------------------------------

@needs_lean
def test_valid_proof_passes_with_exit_zero(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(source=VALID_THEOREM, name="Good.lean")
    assert result.success, result.error
    assert result.metadata["exit_code"] == 0
    assert result.metadata["lean_version"]


@needs_lean
def test_sorry_is_rejected_even_though_lean_exits_zero(tmp_path: Path):
    """The whole point of the tool: Lean's exit code is not trustworthy here."""
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(source=HONEST_SORRY, name="Sorry.lean")
    assert not result.success
    assert "placeholder" in (result.error or "")
    assert "sorry" in (result.error or "")


@needs_lean
def test_a_false_theorem_is_rejected_with_position(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(source=FALSE_THEOREM, name="False.lean")
    assert not result.success
    assert "1 = 2" in (result.error or "") or "not definitionally equal" in (result.error or "")


@needs_lean
def test_unsolved_goals_are_reported_with_context(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    source = "theorem bad (n : Nat) : n + 0 = n + 1 := by\n  induction n with\n  | zero => rfl\n"
    result = tool.execute(source=source, name="Goal.lean")
    assert not result.success
    payload = result.output
    assert "unsolved" in payload.lower() or "not definitionally equal" in payload.lower()


@needs_lean
def test_syntax_errors_report_line_and_column(tmp_path: Path):
    verifier = LeanVerifier(TOOLCHAIN)
    result = verifier.verify_source("theorem bad : := \n", name="Broken.lean", directory=tmp_path)
    assert not result.success
    assert result.errors
    first = result.errors[0]
    assert first.line >= 1
    assert first.column >= 1
    assert first.severity == "error"


@needs_lean
def test_gauss_sum_is_machine_checked_without_sorry(tmp_path: Path):
    """The end-to-end promise: a real theorem certified by Lean itself."""
    verifier = LeanVerifier(TOOLCHAIN)
    result = verifier.verify_source(GAUSS_LEAN, name="Gauss.lean", directory=tmp_path)
    assert result.success, result.render()
    assert result.exit_code == 0
    assert not result.sorry_dependencies
    assert not result.placeholders
    # The kernel's own audit must show only foundational axioms.
    assert {item.axiom for item in result.axioms} <= BENIGN_AXIOMS
    assert any(item.declaration == "gauss_two_mul" for item in result.axioms)


@needs_lean
def test_gauss_proof_hash_is_recorded_and_stable(tmp_path: Path):
    verifier = LeanVerifier(TOOLCHAIN)
    first = verifier.verify_source(GAUSS_LEAN, name="Gauss.lean", directory=tmp_path)
    second = verifier.verify_source(GAUSS_LEAN, name="Gauss.lean", directory=tmp_path)
    assert first.sha256 == second.sha256
    assert first.sha256.startswith("sha256:")


@needs_lean
def test_verification_can_be_re_run_from_a_path(tmp_path: Path):
    tool = RunLeanProofTool(workspace_root=tmp_path)
    written = tool.execute(source=GAUSS_LEAN, name="Stored.lean")
    assert written.success, written.error
    stored = tmp_path / "lean" / "Stored.lean"
    assert stored.is_file()
    again = tool.execute(path=str(stored))
    assert again.success, again.error


@needs_lean
def test_sorry_is_caught_before_compiling(tmp_path: Path):
    """A placeholder must never reach the compiler.

    The rejected source is still written to disk, because preserving the exact
    text that was refused is what makes the rejection auditable; what must not
    happen is a compilation, which is why the exit code is absent.
    """
    tool = RunLeanProofTool(workspace_root=tmp_path)
    result = tool.execute(source=HONEST_SORRY, name="Never.lean")
    assert not result.success
    assert result.metadata["exit_code"] is None
    assert result.metadata["lean_version"] == ""
    assert "placeholder" in (result.error or "")
    # The artifact is retained for inspection, with its refusal recorded.
    assert (tmp_path / "lean" / "Never.lean").is_file()
