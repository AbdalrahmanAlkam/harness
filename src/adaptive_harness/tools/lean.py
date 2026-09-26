"""Lean 4 execution and machine-checked verification.

Lean is the harness's definitive epistemic engine: SymPy can compute, but only
Lean checks a *logical* derivation. This module wraps it behind the tool
interface and enforces the one invariant that matters.

The "zero sorry" invariant
--------------------------
``sorry`` is Lean's escape hatch. A developer can write
``theorem foo : 1 = 2 := by sorry`` and **Lean exits 0** — it emits a
*warning*, not an error, and stamps the declaration as depending on the
``sorryAx`` axiom. A verifier that trusts the exit code alone would certify a
false theorem. Three independent checks are therefore applied, and all three
must pass:

1. the source contains no ``sorry`` token (comments stripped, so prose about
   ``sorry`` does not trip it);
2. the compiler emitted no "declaration uses 'sorry'" diagnostic;
3. an axiom audit shows no declared theorem depends on ``sorryAx``.

Check 3 is the strongest of the three because it is emitted by the kernel
itself rather than inferred from text.

Toolchain
---------
Lean is located on ``PATH``, in ``~/.elan/bin``, or in ``~/.local/bin``. When a
``lakefile`` is present the file is checked with ``lake env lean`` so that
Mathlib and other dependencies resolve; otherwise plain ``lean`` is used, which
means core-only proofs verify with no extra download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Sequence

from adaptive_harness.tools.base import Tool, ToolResult

# Placeholders and escape hatches that let Lean accept an unproved claim.
# `sorry` is the canonical one and `admit` is its alias. A bare `axiom`
# declaration is also refused: a theorem that leans on an assumed constant is
# not proved, and the axiom-dependency check would only catch it after the fact
# with a less obvious message. Note `axioms` (plural) is deliberately not a
# match, so `#print axioms` in a source is not a false positive.
PLACEHOLDER_TOKENS = ("sorry", "admit", "axiom")
SORRY_AXIOM = "sorryAx"

# The token a placeholder is reported under, so the message names what the
# author actually wrote rather than always saying "sorry".
_TOKEN_LABEL = {"sorry": "sorry", "admit": "admit", "axiom": "unproved axiom declaration"}

# Lean's own foundational axioms. A proof depending only on these is accepted;
# anything else beyond them (choice, classical logic) is reported but not fatal,
# because it is a legitimate modelling decision rather than a hole.
BENIGN_AXIOMS = frozenset({"propext", "Quot.sound", "Classical.choice"})

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_LEAN_SEARCH_PATHS = (
    Path.home() / ".elan" / "bin",
    Path.home() / ".local" / "bin",
    Path("/usr/local/bin"),
    Path("/usr/bin"),
)

_DIAGNOSTIC = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<severity>error|warning|info):\s*(?P<message>.*)$")

_DECLARATION = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+|nonrec\s+)*"
    r"(theorem|lemma|def|abbrev|example|instance|structure)\s+"
    r"([A-Za-z_][A-Za-z0-9_'.]*)", re.M)

# Lean's three escape hatches, and the axioms they leave behind.
_ESCAPES = ("sorry", "admit", "sorryAx", "Classical.choice", "propext", "Quot.sound")


class LeanError(RuntimeError):
    """Lean could not be located, or could not be executed."""


def strip_lean_comments(source: str) -> str:
    """Remove ``--`` line comments and nestable ``/- -/`` block comments.

    The zero-sorry scan must not fire on prose that merely mentions the word, so
    comments are removed before the token search. String literals are left alone:
    ``sorry`` inside a string is not a placeholder.
    """
    out: list[str] = []
    index = 0
    length = len(source)
    depth = 0
    while index < length:
        if depth == 0 and source.startswith("--", index):
            newline = source.find("\n", index)
            index = length if newline == -1 else newline
            continue
        if source.startswith("/-", index):
            depth += 1
            index += 2
            continue
        if depth > 0 and source.startswith("-/", index):
            depth -= 1
            index += 2
            continue
        if depth == 0 and source[index] == '"':
            # Blank the literal rather than keeping it: `sorry` inside a string is
            # data, never a placeholder, and the scan below must not see it.
            out.append(" ")
            index += 1
            while index < length:
                if source[index] == "\\" and index + 1 < length:
                    index += 2
                    continue
                if source[index] == '"':
                    index += 1
                    break
                index += 1
            continue
        out.append(source[index] if depth == 0 else " ")
        index += 1
    return "".join(out)


def find_placeholders(source: str) -> list[tuple[int, str]]:
    """Locate unproven-placeholder tokens, with 1-based line numbers."""
    stripped = strip_lean_comments(source)
    hits: list[tuple[int, str]] = []
    for number, line in enumerate(stripped.splitlines(), start=1):
        for token in PLACEHOLDER_TOKENS:
            if re.search(rf"(?<![\w.]){re.escape(token)}(?![\w'])", line):
                hits.append((number, token))
                break
    return hits


def describe_placeholders(placeholders: Iterable[tuple[int, str]]) -> str:
    """Human-readable summary of what was found and where."""
    return ", ".join(f"line {line} ({_TOKEN_LABEL.get(token, token)})"
                     for line, token in placeholders)


def declarations(source: str) -> list[tuple[str, str]]:
    """Top-level declarations as ``(keyword, name)`` pairs, comments stripped."""
    return [(match.group(1), match.group(2))
            for match in _DECLARATION.finditer(strip_lean_comments(source))]


def declared_names(source: str) -> list[str]:
    """Top-level declaration names, used to drive the axiom audit."""
    return [name for _, name in declarations(source)]


@dataclass(frozen=True)
class LeanDiagnostic:
    """One structured message from the Lean compiler."""

    file: str
    line: int
    column: int
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "line": self.line, "column": self.column,
                "severity": self.severity, "message": self.message}


@dataclass(frozen=True)
class LeanAxiom:
    """One axiom a declaration was found to depend on."""

    declaration: str
    axiom: str

    def to_dict(self) -> dict[str, Any]:
        return {"declaration": self.declaration, "axiom": self.axiom}


@dataclass(frozen=True)
class LeanVerification:
    """Outcome of one machine-checked compilation."""

    success: bool
    exit_code: int | None
    source_path: str
    sha256: str
    lean_version: str = ""
    launcher: str = "lean"
    duration_ms: float = 0.0
    diagnostics: tuple[LeanDiagnostic, ...] = ()
    unsolved_goals: tuple[str, ...] = ()
    axioms: tuple[LeanAxiom, ...] = ()
    placeholders: tuple[tuple[int, str], ...] = ()
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def errors(self) -> tuple[LeanDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")

    @property
    def warnings(self) -> tuple[LeanDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "warning")

    @property
    def sorry_dependencies(self) -> tuple[LeanAxiom, ...]:
        return tuple(item for item in self.axioms if item.axiom == SORRY_AXIOM)

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "exit_code": self.exit_code,
                "source_path": self.source_path, "hash": self.sha256,
                "lean_version": self.lean_version, "launcher": self.launcher,
                "duration_ms": round(self.duration_ms, 2),
                "diagnostics": [item.to_dict() for item in self.diagnostics],
                "unsolved_goals": list(self.unsolved_goals),
                "axioms": [item.to_dict() for item in self.axioms],
                "placeholders": [{"line": line, "token": token}
                                 for line, token in self.placeholders],
                "error": self.error}

    def render(self) -> str:
        lines = [f"Lean verification {'PASSED' if self.success else 'FAILED'} "
                 f"({self.lean_version or 'unknown'}, exit {self.exit_code})"]
        for item in self.errors:
            lines.append(f"  {item.file}:{item.line}:{item.column}: {item.message}")
        for goal in self.unsolved_goals:
            lines.append(f"  unsolved goal: {goal}")
        for item in self.sorry_dependencies:
            lines.append(f"  {item.declaration} depends on {SORRY_AXIOM}")
        for line, token in self.placeholders:
            lines.append(f"  line {line}: unproven placeholder '{token}'")
        return "\n".join(lines)


@dataclass
class LeanToolchain:
    """Locate and invoke a usable Lean installation."""

    lean: str | None = field(default=None, init=False)
    lake: str | None = field(default=None, init=False)
    search_path: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        candidates: list[Path] = [Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep)
                                  if entry]
        candidates.extend(_LEAN_SEARCH_PATHS)
        self.search_path = tuple(str(path) for path in candidates)
        self.lean = shutil.which("lean", path=os.pathsep.join(self.search_path))
        self.lake = shutil.which("lake", path=os.pathsep.join(self.search_path))

    @property
    def available(self) -> bool:
        return self.lean is not None

    def version(self) -> str:
        if not self.lean:
            return ""
        try:
            probe = subprocess.run([self.lean, "--version"], capture_output=True, text=True,
                                   timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return probe.stdout.strip() if probe.returncode == 0 else ""

    def require(self) -> str:
        if not self.lean:
            raise LeanError(
                "Lean 4 was not found. Install it with `curl https://elan.lean-lang.org/elan-init.sh "
                "-sSf | sh` (adds elan to ~/.elan/bin) or your distribution's package manager, "
                "then re-run the verification step.")
        return self.lean

    def command_for(self, source: Path) -> list[str]:
        """Prefer ``lake env lean`` so a lakefile's dependencies resolve."""
        if self.lake:
            root = self._lake_root(source)
            if root is not None:
                return [self.lake, "env", "lean", str(source)]
        return [self.require(), str(source)]

    @staticmethod
    def _lake_root(source: Path) -> Path | None:
        for directory in [source.parent, *source.parents]:
            if (directory / "lakefile.lean").is_file() or (directory / "lakefile.toml").is_file():
                return directory
            if directory.parent == directory:
                break
        return None


def _parse_diagnostics(text: str, default_file: str) -> list[LeanDiagnostic]:
    found: list[LeanDiagnostic] = []
    for line in text.splitlines():
        match = _DIAGNOSTIC.match(line.strip())
        if match:
            found.append(LeanDiagnostic(
                file=match.group("file"), line=int(match.group("line")),
                column=int(match.group("col")), severity=match.group("severity"),
                message=match.group("message").strip()))
        elif line.strip().startswith(("error:", "warning:")):
            found.append(LeanDiagnostic(file=default_file, line=0, column=0,
                                        severity=line.split(":", 1)[0],
                                        message=line.split(":", 1)[1].strip()))
    return found


def _collect_unsolved_goals(text: str) -> list[str]:
    """Capture the goal state Lean prints after an unsolved-goal error."""
    goals: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "unsolved goals" in line:
            for follower in lines[index + 1:]:
                stripped = follower.strip()
                if not stripped or _DIAGNOSTIC.match(stripped):
                    break
                if stripped.startswith(("error:", "warning:")):
                    break
                goals.append(stripped)
    return goals


_AXIOM_LINE = re.compile(r"'([^']+)' depends on axioms:\s*\[([^\]]*)\]")


def _collect_axioms(text: str) -> list[LeanAxiom]:
    axioms: list[LeanAxiom] = []
    for match in _AXIOM_LINE.finditer(text):
        declaration = match.group(1)
        for name in (item.strip() for item in match.group(2).split(",")):
            if name:
                axioms.append(LeanAxiom(declaration=declaration, axiom=name))
    return axioms


def axiom_audit_source(source: str) -> str:
    """Append ``#print axioms`` for every declaration, for a kernel-side audit.

    Lean reports a declaration's axiom dependencies itself, so this turns "no
    sorry" from a text scan into a fact the kernel vouches for.
    """
    names = declared_names(source)
    if not names:
        return source
    return source.rstrip() + "\n\n" + "\n".join(f"#print axioms {name}" for name in names) + "\n"


class LeanVerifier:
    """Compile Lean sources and adjudicate them under the zero-sorry rule."""

    def __init__(self, toolchain: LeanToolchain | None = None, timeout_s: float = 300.0,
                 audit_axioms: bool = True):
        self.toolchain = toolchain or LeanToolchain()
        self.timeout_s = timeout_s
        self.audit_axioms = audit_axioms

    def verify_source(self, source: str, *, name: str = "Proof.lean",
                      directory: str | Path | None = None) -> LeanVerification:
        """Write ``source`` to disk, compile it, and adjudicate the result."""
        import time

        started = time.perf_counter()
        target_dir = Path(directory) if directory else Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / name
        path.write_text(source, encoding="utf-8")
        return self.verify(path, started=started)

    def verify(self, path: str | Path, *, started: float | None = None) -> LeanVerification:
        """Compile a ``.lean`` file and decide whether the proof is certified."""
        import time

        target = Path(path).resolve()
        begin = started if started is not None else time.perf_counter()
        if not target.is_file():
            return LeanVerification(False, None, str(target), "", error="Lean source does not exist")
        text = target.read_text(encoding="utf-8", errors="replace")
        digest = f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

        # Check 1: no placeholder token in the source itself.
        placeholders = find_placeholders(text)
        if placeholders:
            return LeanVerification(
                False, None, str(target), digest, launcher=self.toolchain.lean or "lean",
                duration_ms=(time.perf_counter() - begin) * 1000.0,
                placeholders=tuple(placeholders),
                error=("Proof contains an unproven placeholder and was rejected: "
                       f"{describe_placeholders(placeholders)}"))

        lean = self.toolchain.require()
        # The axiom audit must be part of the compiled file, so it is appended to
        # a sibling copy that is compiled in place of the original. The recorded
        # digest is always that of the original source, never of the audit.
        audited = axiom_audit_source(text) if self.audit_axioms else text
        compile_target = target
        if audited != text:
            compile_target = target.with_name(f".{target.stem}.audit.lean")
            compile_target.write_text(audited, encoding="utf-8")
        command = self.toolchain.command_for(compile_target)
        exit_code: int | None
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=self.timeout_s, check=False,
                                       cwd=str(self.toolchain._lake_root(compile_target)
                                               or compile_target.parent))
            exit_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired:
            return LeanVerification(False, None, str(target), digest,
                                    launcher=" ".join(command[:2]),
                                    duration_ms=(time.perf_counter() - begin) * 1000.0,
                                    error=f"Lean timed out after {self.timeout_s:g}s")
        except OSError as exc:
            raise LeanError(f"Could not execute Lean: {exc}") from exc
        finally:
            if compile_target != target:
                compile_target.unlink(missing_ok=True)

        combined = f"{stdout}\n{stderr}"
        diagnostics = _parse_diagnostics(combined, target.name)
        goals = _collect_unsolved_goals(combined)
        axioms = _collect_axioms(combined) if self.audit_axioms else []

        # Check 2: the compiler itself flagged a placeholder.
        sorry_diagnostics = [item for item in diagnostics if "uses 'sorry'" in item.message
                             or SORRY_AXIOM in item.message]
        # Check 3: a declaration depends on sorryAx.
        sorry_axioms = [item for item in axioms if item.axiom == SORRY_AXIOM]
        unsupported_axioms = [item for item in axioms if item.axiom not in BENIGN_AXIOMS]

        errors = [item for item in diagnostics if item.severity == "error"]
        failure: str | None = None
        if exit_code != 0:
            failure = ((stderr or stdout).strip() or f"lean exited with code {exit_code}")[-2000:]
        elif sorry_diagnostics:
            failure = ("Proof contains unproven placeholder 'sorry': the compiler reported "
                       f"{sorry_diagnostics[0].message}")
        elif sorry_axioms:
            names = ", ".join(item.declaration for item in sorry_axioms)
            failure = f"Proof contains unproven placeholder 'sorry': {names} depends on {SORRY_AXIOM}"
        elif unsupported_axioms:
            names = ", ".join(sorted({item.axiom for item in unsupported_axioms}))
            failure = f"Proof depends on unsupported axiom(s): {names}"
        elif goals:
            failure = "Proof has unsolved goals: " + "; ".join(goals[:3])
        elif errors:
            failure = errors[0].message

        return LeanVerification(
            success=failure is None, exit_code=exit_code, source_path=str(target), sha256=digest,
            lean_version=self.toolchain.version(), launcher=" ".join(command[:2]),
            duration_ms=(time.perf_counter() - begin) * 1000.0,
            diagnostics=tuple(diagnostics), unsolved_goals=tuple(goals), axioms=tuple(axioms),
            stdout=stdout[-4000:], stderr=stderr[-4000:], error=failure)


class RunLeanProofTool(Tool):
    """Compiles and machine-checks formal Lean 4 theorems and tactics."""

    name = "run_lean_proof"
    description = (
        "Compiles a Lean 4 file or snippet with 'lean' (or 'lake env lean') and verifies there are "
        "zero errors and zero unproven placeholders. A proof is rejected if the source contains "
        "sorry, admit, or a bare axiom declaration, if the compiler reports one, or if any "
        "declaration depends on the sorryAx axiom. Writes a SHA-256 receipt next to the file.")
    parameters = {"type": "object", "properties": {
        "lean_code": {"type": "string",
                      "description": "Complete Lean 4 source, written to proofs/<theorem_name>.lean."},
        "file_path": {"type": "string", "description": "Existing .lean file to verify instead."},
        "theorem_name": {"type": "string",
                         "description": "Stem for the written file and its receipt.",
                         "default": "Proof"},
        "timeout_s": {"type": "integer", "minimum": 1, "default": 60,
                      "description": "Compilation timeout in seconds."},
        "source": {"type": "string", "description": "Alias for lean_code."},
        "path": {"type": "string", "description": "Alias for file_path."},
        "name": {"type": "string", "description": "Alias for theorem_name."},
    }, "required": []}

    def __init__(self, workspace_root: str | Path | None = None, *,
                 toolchain: LeanToolchain | None = None, timeout_s: float = 300.0,
                 audit_axioms: bool = True, lean_dir: str | None = "lean",
                 write_receipts: bool = True):
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.verifier = LeanVerifier(toolchain, timeout_s=timeout_s, audit_axioms=audit_axioms)
        self.lean_dir = lean_dir
        self.write_receipts = write_receipts

    def execute(self, lean_code: str | None = None, file_path: str | None = None,
                theorem_name: str = "Proof", timeout_s: int | None = None,
                source: str | None = None, path: str | None = None,
                name: str | None = None, **kwargs: Any) -> ToolResult:
        # Accept the short aliases so a caller can use either spelling.
        code = lean_code if lean_code is not None else source
        target_path = file_path if file_path is not None else path
        stem = name or theorem_name or "Proof"
        if timeout_s is not None:
            self.verifier.timeout_s = float(timeout_s)
        if not self.verifier.toolchain.available:
            message = ("Lean 4 was not found on PATH, in ~/.elan/bin, or in ~/.local/bin. "
                       "Install it with `curl https://elan.lean-lang.org/elan-init.sh -sSf | sh`.")
            return ToolResult(success=False, output="", error=message,
                              metadata={"lean_available": False})
        directory = self.workspace_root / self.lean_dir if self.lean_dir else self.workspace_root
        try:
            if target_path:
                target = Path(target_path)
                if not target.is_absolute():
                    target = self.workspace_root / target
                result = self.verifier.verify(target)
            elif code:
                filename = stem if stem.endswith(".lean") else f"{stem}.lean"
                result = self.verifier.verify_source(code, name=filename, directory=directory)
            else:
                return ToolResult(success=False, output="",
                                  error="Provide either 'lean_code' (Lean 4 text) or 'file_path' "
                                        "(a .lean file)")
        except LeanError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except (OSError, ValueError) as exc:
            return ToolResult(success=False, output="",
                              error=f"Lean verification failed: {type(exc).__name__}: {exc}")
        receipt_path = self._write_receipt(result) if self.write_receipts else None
        payload = result.to_dict()
        if receipt_path is not None:
            payload["receipt_path"] = str(receipt_path)
        return ToolResult(success=result.success, output=json.dumps(payload, ensure_ascii=False),
                          error=result.error,
                          metadata={"exit_code": result.exit_code,
                                    "lean_version": result.lean_version,
                                    "unsolved_goals": len(result.unsolved_goals),
                                    "sorry_dependencies": len(result.sorry_dependencies),
                                    "axioms": [item.axiom for item in result.axioms],
                                    "receipt_path": str(receipt_path) if receipt_path else None})

    def _write_receipt(self, result: LeanVerification) -> Path | None:
        """Persist the per-file receipt beside the proof, as a durable record.

        The receipt is the artifact an auditor re-reads later, so it carries the
        digest, the verdict, the compiler output, and the time of verification.
        """
        target = Path(result.source_path)
        if not target.is_file():
            return None
        payload = {
            "theorem_id": target.stem,
            "path": str(target),
            "sha256": result.sha256,
            "verified": result.success,
            "status": "LEAN_VERIFIED" if result.success else "LEAN_REJECTED",
            "exit_code": result.exit_code,
            "lean_version": result.lean_version,
            "errors": [item.to_dict() for item in result.errors],
            "unsolved_goals": list(result.unsolved_goals),
            "placeholders": [{"line": line, "token": token} for line, token in result.placeholders],
            "axioms": [item.to_dict() for item in result.axioms],
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
            "timestamp": _utc_now(),
        }
        receipt = target.with_name(target.name + ".receipt.json")
        try:
            receipt.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            return None
        return receipt
