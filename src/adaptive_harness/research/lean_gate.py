"""The two-tier mathematical proof standard.

Tier 1 is computational: a SymPy script that reduces an expression to a closed
form. Tier 2 is logical: a Lean 4 file that the kernel itself checks. SymPy can
confirm that two expressions are equal; only Lean can confirm that a *deduction*
is valid.

The tiers are deliberately not interchangeable. A SymPy script that never exits 0
is a broken derivation and fails the gate; a Lean file that reports an error, an
unsolved goal, or a dependency on ``sorryAx`` is an unproved claim and fails the
gate. Pre-compilation clearance runs this over every theorem bound for the paper,
so an unproved statement cannot reach a reader as a published result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from adaptive_harness.tools.lean import (BENIGN_AXIOMS, LeanToolchain, LeanVerification,
                                         LeanVerifier, declared_names)

LEAN_DIRNAME = "lean"
RECEIPT_PREFIX = "LEAN-PROOF"

# Lean declarations that assert a proposition, as opposed to definitions.
_THEOREM_KEYWORDS = ("theorem", "lemma", "corollary", "example")


@dataclass(frozen=True)
class LeanProofReceipt:
    """An immutable record that Lean certified one formal proof."""

    proof_id: str
    name: str
    path: str
    sha256: str
    status: str
    lean_version: str = ""
    duration_ms: float = 0.0
    exit_code: int | None = None
    theorems: tuple[str, ...] = ()
    axioms: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    verified_at: str = ""

    @property
    def certified(self) -> bool:
        return self.status == "LEAN_VERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return {"proof_id": self.proof_id, "name": self.name, "path": self.path,
                "hash": self.sha256, "status": self.status, "lean_version": self.lean_version,
                "duration_ms": round(self.duration_ms, 2), "exit_code": self.exit_code,
                "theorems": list(self.theorems), "axioms": list(self.axioms),
                "errors": list(self.errors), "verified_at": self.verified_at}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def theorems_in(source: str) -> list[str]:
    """Names of the propositions a Lean file asserts."""
    from adaptive_harness.tools.lean import declarations
    return [name for keyword, name in declarations(source)
            if keyword in _THEOREM_KEYWORDS]


@dataclass
class LeanProofGate:
    """Verify every ``.lean`` file under a proofs directory.

    ``mathlib_available`` is recorded so the paper can state honestly whether the
    proofs were checked against core Lean alone or against a Mathlib environment.
    """

    proof_dir: Path
    timeout_s: float = 300.0
    toolchain: LeanToolchain | None = None
    receipts: list[LeanProofReceipt] = field(default_factory=list)
    _counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.proof_dir = Path(self.proof_dir)
        self.proof_dir.mkdir(parents=True, exist_ok=True)
        self.toolchain = self.toolchain or LeanToolchain()

    @property
    def available(self) -> bool:
        return bool(self.toolchain and self.toolchain.available)

    @property
    def lean_version(self) -> str:
        return self.toolchain.version() if self.toolchain else ""

    def scripts(self) -> list[Path]:
        return sorted(self.proof_dir.glob("*.lean"))

    def write(self, name: str, source: str) -> Path:
        """Write a Lean source into the gate's directory and return its path."""
        target = self.proof_dir / (name if name.endswith(".lean") else f"{name}.lean")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        return target

    def verify_file(self, path: str | Path) -> LeanProofReceipt:
        """Machine-check one file and mint its receipt."""
        target = Path(path).resolve()
        index = max(1, len(self.scripts()))
        if not target.is_file():
            return LeanProofReceipt(f"{RECEIPT_PREFIX}-{index:03d}", target.stem, str(target),
                                    "", "LEAN_MISSING", verified_at=_now())
        source = target.read_text(encoding="utf-8", errors="replace")
        digest = f"sha256:{hashlib.sha256(source.encode('utf-8')).hexdigest()}"
        names = tuple(theorems_in(source))
        position = sorted(self.scripts()).index(target) + 1 if target in self.scripts() else index
        proof_id = f"{RECEIPT_PREFIX}-{position:03d}"

        if not self.available:
            # Refusing to certify is the only honest outcome without a toolchain.
            return LeanProofReceipt(proof_id, target.stem, str(target), digest, "LEAN_UNAVAILABLE",
                                    theorems=names, verified_at=_now(),
                                    errors=("Lean 4 was not found, so this proof is unverified; "
                                            "an unverified proof is never certified",))
        verifier = LeanVerifier(self.toolchain, timeout_s=self.timeout_s)
        result: LeanVerification = verifier.verify(target)
        axioms = tuple(sorted({item.axiom for item in result.axioms}))
        unexpected = [name for name in axioms if name not in BENIGN_AXIOMS]
        errors = tuple(item.message for item in result.errors)
        if result.success and unexpected:
            status = "LEAN_DEPENDS_ON_EXTRA_AXIOMS"
        elif result.success:
            status = "LEAN_VERIFIED"
        else:
            status = "LEAN_REJECTED"
        return LeanProofReceipt(
            proof_id=proof_id, name=target.stem, path=str(target), sha256=digest, status=status,
            lean_version=result.lean_version, duration_ms=result.duration_ms,
            exit_code=result.exit_code, theorems=names, axioms=axioms, errors=errors,
            verified_at=_now())

    def verify_all(self) -> list[LeanProofReceipt]:
        """Verify every Lean file and record the receipts."""
        receipts = [self.verify_file(path) for path in self.scripts()]
        self.receipts = receipts
        return receipts

    @property
    def all_certified(self) -> bool:
        """True only when at least one proof exists and every one is certified."""
        return bool(self.receipts) and all(receipt.certified for receipt in self.receipts)

    def to_index(self, path: str | Path | None = None) -> dict[str, Any]:
        """A machine-readable index of the formal verification state."""
        payload = {
            "lean_version": self.lean_version,
            "mathlib_available": self._mathlib_available(),
            "all_certified": self.all_certified,
            "receipts": [receipt.to_dict() for receipt in self.receipts],
        }
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            import json
            target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def _mathlib_available(self) -> bool:
        """Whether a Mathlib environment can be resolved, for honest reporting."""
        if not self.toolchain or not self.toolchain.lake:
            return False
        import subprocess
        probe = Path(self.proof_dir) / ".mathlib_probe.lean"
        probe.write_text("import Mathlib\n", encoding="utf-8")
        try:
            completed = subprocess.run(
                [self.toolchain.lake, "env", "lean", str(probe)],
                capture_output=True, text=True, timeout=180, check=False,
                cwd=str(self.proof_dir))
            return completed.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
        finally:
            probe.unlink(missing_ok=True)

    def clearance(self) -> tuple[bool, str, tuple[str, ...]]:
        """Pre-compilation clearance: may the paper be typeset?

        The paper is blocked unless every Lean file is certified. An absent
        toolchain or an empty proof set is a *failure*, because publishing a
        formally unverified theorem while claiming a two-tier standard would be
        the exact failure mode this gate exists to prevent.
        """
        if not self.available:
            return False, ("Lean 4 is unavailable, so no proof can be machine-checked; "
                           "the paper's formal claims would be unverified"), ()
        scripts = self.scripts()
        if not scripts:
            return False, "no Lean proof exists, so the formal tier is empty", ()
        receipts = self.receipts or self.verify_all()
        evidence = [f"{receipt.proof_id} {receipt.status}" for receipt in receipts]
        failed = [receipt for receipt in receipts if not receipt.certified]
        if failed:
            detail = "; ".join(f"{item.proof_id} {item.status}"
                               + (f" ({item.errors[0][:120]})" if item.errors else "")
                               for item in failed)
            return False, f"{len(failed)} of {len(receipts)} formal proof(s) not certified: {detail}", tuple(evidence)
        total = sum(len(receipt.theorems) for receipt in receipts)
        return True, (f"all {len(receipts)} Lean file(s) machine-checked by {self.lean_version}, "
                      f"covering {total} theorem(s), with no sorry and no sorryAx"), tuple(evidence)


def render_lean_listing(source: str, max_lines: int = 60) -> str:
    """Trim a Lean source for inclusion in the paper appendix."""
    lines = source.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept.append(f"... ({len(lines) - max_lines} further lines omitted; see the .lean file)")
    return "\n".join(kept)


_LEAN_LISTING = re.compile(r"^\s*(theorem|lemma|def)\s+([A-Za-z_][A-Za-z0-9_'.]*)", re.M)


def summarise_lean_sources(sources: Mapping[str, str]) -> list[str]:
    """One line per Lean file, for a table in the paper."""
    rows: list[str] = []
    for name, source in sources.items():
        theorems = theorems_in(source)
        rows.append(f"{name}: {len(theorems)} theorem(s)"
                    + (f" ({', '.join(theorems[:3])})" if theorems else ""))
    return rows
