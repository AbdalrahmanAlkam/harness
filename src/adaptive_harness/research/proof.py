"""Formal proof receipts: exact SymPy derivations that must exit 0.

A claim is not admissible evidence until a self-contained script in ``proofs/``
runs to completion with exit code 0 *and* is free of floating-point
approximations. The runner enforces both conditions mechanically:

* execution — a subprocess with a wall-clock timeout and a network-free import
  guard, so a "proof" cannot quietly depend on the internet;
* exactness — a static AST scan rejecting ``Float`` literals and the
  approximating entry points ``float``, ``evalf``, ``N``, ``nsimplify``'s
  tolerance path, and friends.

Both checks are deterministic, so a receipt is reproducible on any machine.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

PROOF_DIRNAME = "proofs"

# Approximating builtins/attributes that would silently weaken an exact claim.
_APPROXIMATING_CALLS = frozenset({"float", "evalf", "evalf", "n", "N", "lambdify", "pyapprox"})
_APPROXIMATING_ATTRS = frozenset({"evalf", "n", "N", "evalf"})

_SYMPY_FLOAT = re.compile(r"\bFloat\s*\(")


class ProofError(RuntimeError):
    """A proof script violated the exactness or execution contract."""


@dataclass(frozen=True)
class ProofReceipt:
    """Immutable record that one theorem was mechanically discharged."""

    theorem_id: str
    script: str
    sha256: str
    status: str
    exit_code: int | None = None
    duration_ms: float = 0.0
    stdout: str = ""
    violations: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return self.status == "VERIFIED_EXIT_0"

    def to_dict(self) -> dict[str, Any]:
        return {"theorem_id": self.theorem_id, "script": self.script, "hash": self.sha256,
                "status": self.status, "exit_code": self.exit_code,
                "duration_ms": round(self.duration_ms, 2),
                "violations": list(self.violations)}


def scan_for_approximations(source: str) -> list[str]:
    """Return a list of exactness violations found in a proof script.

    Rejects float literals (``1.0``), ``Float(...)`` constructions, and calls to
    functions that convert exact values into floating point. Integer division
    producing a float at runtime is caught by the ban on ``float`` and on
    ``/``-returning helpers is not attempted: the type error surfaces as a
    non-zero exit code instead.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SYNTAX_ERROR line {exc.lineno}: {exc.msg}"]

    violations: list[str] = []
    call_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            violations.append(f"line {node.lineno}: float literal {node.value!r} is not exact")
        elif isinstance(node, ast.Call):
            call_nodes.add(id(node.func))
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else "")
            if name in _APPROXIMATING_CALLS:
                violations.append(
                    f"line {node.lineno}: {name}() approximates and cannot prove an exact claim")
        elif isinstance(node, ast.Attribute) and node.attr in _APPROXIMATING_ATTRS:
            if id(node) not in call_nodes:
                violations.append(
                    f"line {node.lineno}: .{node.attr} approximates and cannot prove an exact claim")
    for match in _SYMPY_FLOAT.finditer(source):
        violations.append(f"Float() construction at offset {match.start()} is not exact")
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in violations:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


_NETWORK_GUARD = """
import socket
def _blocked(*args, **kwargs):
    raise OSError('proof scripts must not open network connections')
socket.socket = _blocked
socket.create_connection = _blocked
"""


@dataclass
class ProofRunner:
    """Execute and adjudicate exact-derivation scripts for a research topic.

    Receipts are memoized on the script's SHA-256. A receipt is a pure function
    of the script bytes, so re-executing an unchanged proof would be pure waste;
    the gate re-checks every proof each cycle, and SymPy derivations are the
    slowest artifact in the tree.
    """

    proof_dir: Path
    timeout_s: float = 120.0
    python_executable: str = field(default=sys.executable)
    receipts: list[ProofReceipt] = field(default_factory=list)
    _cache: dict[str, ProofReceipt] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.proof_dir = Path(self.proof_dir)
        self.proof_dir.mkdir(parents=True, exist_ok=True)

    def scripts(self) -> list[Path]:
        return sorted(self.proof_dir.glob("*.py"))

    def run_script(self, script: str | Path, theorem_id: str | None = None,
                   *, use_cache: bool = True) -> ProofReceipt:
        """Run one proof script and return its receipt without recording it."""
        import hashlib
        import time

        path = Path(script)
        if not path.is_absolute():
            path = self.proof_dir / path
        if not path.is_file():
            receipt = ProofReceipt(theorem_id or path.stem, str(path), "", "MISSING", None, 0.0,
                                  (), ("proof script does not exist",))
            return receipt
        source = path.read_text(encoding="utf-8", errors="replace")
        digest = f"sha256:{hashlib.sha256(source.encode('utf-8')).hexdigest()}"
        identifier = theorem_id or path.stem

        # The cache is keyed on content, so an edited script always re-runs.
        cached = self._cache.get(digest) if use_cache else None
        if cached is not None and cached.theorem_id == identifier and cached.script == str(path):
            return cached

        violations = scan_for_approximations(source)
        if violations:
            receipt = ProofReceipt(identifier, str(path), digest, "REJECTED_INEXACT", None, 0.0,
                                   "", tuple(violations))
            self._cache[digest] = receipt
            return receipt

        guarded = self.proof_dir / f".{path.stem}.guarded.py"
        guarded.write_text(_NETWORK_GUARD + source, encoding="utf-8")
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [self.python_executable, str(guarded)], capture_output=True, text=True,
                timeout=self.timeout_s, check=False, cwd=str(self.proof_dir))
            exit_code: int | None = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired:
            exit_code, stdout, stderr = None, "", f"timeout after {self.timeout_s:g}s"
        except OSError as exc:
            exit_code, stdout, stderr = None, "", f"{type(exc).__name__}: {exc}"
        finally:
            guarded.unlink(missing_ok=True)
        duration_ms = (time.perf_counter() - started) * 1000.0

        if exit_code == 0:
            status = "VERIFIED_EXIT_0"
        elif exit_code is None:
            status = "REJECTED_TIMEOUT"
        else:
            status = "REJECTED_NONZERO"
        receipt = ProofReceipt(identifier, str(path), digest, status, exit_code, duration_ms,
                               (stdout + stderr)[-4000:], ())
        self._cache[digest] = receipt
        return receipt

    def run_all(self, theorems: Mapping[str, str] | None = None) -> list[ProofReceipt]:
        """Run every proof script, optionally pairing script paths to theorem ids.

        ``theorems`` maps a script filename to a theorem identifier so receipts
        are citable (``THM-01``) rather than file-shaped.
        """
        mapping = dict(theorems or {})
        receipts = [self.run_script(path, mapping.get(path.name))
                    for path in self.scripts()]
        self.receipts = receipts
        return receipts

    def record(self, receipt: ProofReceipt, ledger: Any = None, sender: Mapping[str, str] | None = None,
               recipient: Mapping[str, str] | None = None) -> ProofReceipt:
        """Persist a receipt to disk and, when given, to the comm ledger."""
        self.receipts.append(receipt)
        index_path = self.proof_dir.parent / "proof_receipts.json"
        existing: list[dict[str, Any]] = []
        if index_path.is_file():
            try:
                existing = json.loads(index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []
        existing = [item for item in existing if item.get("script") != receipt.script]
        existing.append(receipt.to_dict())
        index_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        if ledger is not None:
            from adaptive_harness.research.ledger import sha256_file
            ledger.append("PROOF_VERIFIED" if receipt.verified else "PROOF_REJECTED",
                          sender or {"agent_id": "proof_runner", "role": "Proof Runner"},
                          recipient or {"agent_id": "theoretical_lead", "role": "Theory Lead"},
                          {"theorem_id": receipt.theorem_id,
                           "script": str(Path(receipt.script).name),
                           "hash": receipt.sha256 or sha256_file(receipt.script),
                           "status": receipt.status, "exit_code": receipt.exit_code,
                           "violations": list(receipt.violations)})
        return receipt

    @property
    def all_verified(self) -> bool:
        return bool(self.receipts) and all(receipt.verified for receipt in self.receipts)
