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
from adaptive_harness.research.claim import Verdict
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


# -- the formalisation library --------------------------------------------

def test_every_library_proof_is_accepted_by_lean(tmp_path: Path):
    """No proof may ship unverified.

    The library holds hand-written Lean, so it can rot: a tactic change in a new
    Lean release would silently make a published theorem unprovable. This test
    compiles every entry and fails loudly if one stops verifying.
    """
    from adaptive_harness.research.formalise import LIBRARY
    from adaptive_harness.research.lean_gate import LeanProofGate

    gate = LeanProofGate(tmp_path / "lean")
    for formalisation in LIBRARY:
        gate.write(formalisation.lean_id, formalisation.source)
    receipts = gate.verify_all()
    assert len(receipts) == len(LIBRARY)
    for receipt, formalisation in zip(receipts, sorted(LIBRARY, key=lambda f: f.lean_id)):
        assert receipt.certified, f"{formalisation.lean_id} did not verify: {receipt.errors[:2]}"
        # Certification already implies no sorry; assert the axiom set directly
        # so a future status change cannot hide a sorryAx dependency.
        assert "sorryAx" not in receipt.axioms
        assert set(receipt.axioms) <= BENIGN_AXIOMS
    ok, detail, _ = gate.clearance()
    assert ok, detail


def test_formalisation_selection_is_honest_about_no_match():
    from adaptive_harness.research.formalise import plan_formalisations

    chosen, notes = plan_formalisations("gauss sum of natural numbers")
    assert [item.lean_id for item in chosen] == ["LEAN-GAUSS"]
    none, notes = plan_formalisations("zzz qqq unmatchable")
    assert none == ()
    assert "No formalisation" in notes


def test_formalisations_are_matched_to_propositions():
    from adaptive_harness.research.claim import Proposition
    from adaptive_harness.research.formalise import formalisations_for

    unrelated = (Proposition("PROP-99", "theorem", "unrelated"),)
    assert formalisations_for(unrelated) == ()
    supported = (Proposition("PROP-05", "lemma", "imbalance"),)
    assert [item.lean_id for item in formalisations_for(supported)] == ["LEAN-BALANCED"]


def test_gate_refuses_clearance_without_lean(tmp_path: Path, monkeypatch):
    """An unavailable toolchain must fail clearance, never pass vacuously."""
    from adaptive_harness.research.lean_gate import LeanProofGate

    gate = LeanProofGate(tmp_path / "lean")
    gate.write("Gauss", GAUSS_LEAN)
    # Simulate a machine with no Lean by detaching the toolchain's binary.
    monkeypatch.setattr(gate.toolchain, "lean", None)
    ok, detail, _ = gate.clearance()
    assert not ok
    assert "unavailable" in detail


def test_gate_refuses_clearance_with_no_proofs(tmp_path: Path):
    from adaptive_harness.research.lean_gate import LeanProofGate

    gate = LeanProofGate(tmp_path / "lean")
    gate.verify_all()
    ok, detail, _ = gate.clearance()
    assert not ok
    assert "no Lean proof" in detail


# -- the formal tier inside the swarm --------------------------------------

@needs_lean
def test_proven_claim_requires_a_certified_lean_proof(tmp_path: Path):
    """An asserted theorem may not be published without the formal tier."""
    from adaptive_harness.research import ResearchSwarm, SwarmConfig

    swarm = ResearchSwarm("gauss sum of the first n natural numbers", root=tmp_path,
                          config=SwarmConfig(max_cycles=1))
    swarm._synthesize()
    swarm._formalise()
    swarm._adjudicate()
    assert swarm.claims.headline is Verdict.PROVEN
    ok, detail, _ = swarm._evaluate_formal()
    assert ok, detail
    assert "sorryAx" in detail


@needs_lean
def test_refuted_claim_needs_no_lean_proof(tmp_path: Path):
    """A refutation is certified by its exact witness, not by a Lean proof."""
    from adaptive_harness.research import ResearchSwarm, SwarmConfig

    swarm = ResearchSwarm("quadratic expansion", root=tmp_path,
                          config=SwarmConfig(max_cycles=1,
                                             claim="(x + y)**2 == x**2 + y**2",
                                             claim_symbols=("x", "y")))
    swarm._synthesize()
    swarm._formalise()
    swarm._adjudicate()
    assert swarm.claims.headline is Verdict.DISPROVEN
    ok, detail, _ = swarm._evaluate_formal()
    assert ok, detail
    assert "no theorem is asserted" in detail


@needs_lean
def test_lean_files_land_in_proofs_lean_and_not_in_the_sympy_gate(tmp_path: Path):
    """The two tiers must not contaminate each other."""
    from adaptive_harness.research import ResearchSwarm
    from adaptive_harness.research.formalise import LIBRARY

    swarm = ResearchSwarm("gauss sum of the first n natural numbers", root=tmp_path)
    swarm._formalise()
    lean_files = list(swarm.workspace.lean_dir.glob("*.lean"))
    assert lean_files
    assert {path.name for path in lean_files} <= {f"{f.lean_id}.lean" for f in LIBRARY}
    # The SymPy gate globs proofs/*.py only, so Lean files are invisible to it.
    assert all(path.suffix == ".py" for path in swarm.proofs.scripts())
    assert all(path.suffix == ".lean" for path in lean_files)


@needs_lean
def test_paper_carries_the_lean_boxes_and_listings(tmp_path: Path):
    from adaptive_harness.research import ResearchSwarm

    swarm = ResearchSwarm("gauss sum of the first n natural numbers, and balanced routing", root=tmp_path)
    outcome = swarm.run()
    assert outcome.solved, outcome.render()
    paper = Path(swarm.workspace.paper_typ).read_text()
    assert "lean-box" in paper
    assert "Formal Foundations" in paper
    assert "proofs/lean/LEAN-GAUSS.lean" in paper
    assert "sorryAx" in paper
    # The appendix carries the listings so a reader can reproduce the check.
    assert "Lean 4 Listings" in paper
    assert "gauss_two_mul" in paper
