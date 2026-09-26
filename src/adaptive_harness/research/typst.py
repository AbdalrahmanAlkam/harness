"""Typst toolchain resolution and PDF compilation.

Typst is resolved in three escalating steps — a binary on ``PATH``, the
``typst`` PyPI wrapper (which bundles the compiler), then a reported failure.
Compilation treats **any** Typst diagnostic on stderr as a build break, because
a paper that "compiled with warnings" has an unresolved reference or a bad
figure path, and both are defects the invariant gate must reject.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


class TypstError(RuntimeError):
    """Typst could not be resolved, or compilation failed."""


@dataclass(frozen=True)
class CompileResult:
    """Outcome of one ``typst compile`` invocation."""

    success: bool
    pdf_path: Path
    typst_path: str
    typst_version: str
    pages: int | None = None
    size_bytes: int = 0
    warnings: tuple[str, ...] = ()
    error: str | None = None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "pdf_path": str(self.pdf_path),
                "typst_path": self.typst_path, "typst_version": self.typst_version,
                "pages": self.pages, "size_bytes": self.size_bytes,
                "warnings": list(self.warnings), "error": self.error}


# Typst writes progress notes to stderr that are not diagnostics.
_BENIGN_STDERR = ("reading", "writing", "compiling")


def resolve_typst(*, allow_install: bool = True) -> tuple[str, str]:
    """Return ``(executable_prefix, version)`` for a usable Typst toolchain.

    The prefix is empty for a native binary and ``python -m`` for the wrapper.
    """
    binary = shutil.which("typst")
    if binary:
        try:
            probe = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                   timeout=30, check=False)
            if probe.returncode == 0:
                return "", probe.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass

    wrapper = _probe_python_wrapper()
    if wrapper is not None:
        return wrapper, ""

    if not allow_install:
        raise TypstError("Typst is not installed and automatic installation is disabled")
    if _install_python_wrapper():
        wrapper = _probe_python_wrapper()
        if wrapper is not None:
            return wrapper, ""
    raise TypstError(
        "Typst is unavailable. Install it with `cargo install typst-cli`, `brew install typst`, "
        "or `pip install typst`, then re-run the compilation step.")


def _probe_python_wrapper() -> str | None:
    try:
        probe = subprocess.run([sys.executable, "-m", "typst", "--version"],
                               capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode == 0 and probe.stdout.strip():
        return sys.executable
    return None


def _install_python_wrapper() -> bool:
    try:
        completed = subprocess.run([sys.executable, "-m", "pip", "install", "typst"],
                                   capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _count_pages(pdf: Path) -> int | None:
    try:
        data = pdf.read_bytes()
    except OSError:
        return None
    # Uncompressed object streams make the count unreliable; report None rather
    # than guess, since a wrong page count would be a false proof receipt.
    if b"/Count" in data and b"/Type /Page" not in data:
        return None
    import re
    counts = [int(match) for match in re.findall(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data)]
    if counts:
        return max(counts)
    occurrences = data.count(b"/Type /Page\n") + data.count(b"/Type/Page/")
    return occurrences or None


class TypstCompiler:
    """Compile Typst sources into publication-grade PDFs."""

    def __init__(self, *, root: str | Path | None = None, allow_install: bool = True,
                 timeout_s: float = 300.0):
        self.root = Path(root).resolve() if root else None
        self.allow_install = allow_install
        self.timeout_s = timeout_s
        self._prefix: tuple[str, str] | None = None

    @property
    def toolchain(self) -> tuple[str, str]:
        if self._prefix is None:
            self._prefix = resolve_typst(allow_install=self.allow_install)
        return self._prefix

    def compile(self, typ_path: str | Path, pdf_path: str | Path | None = None) -> CompileResult:
        """Compile ``typ_path`` to PDF, treating any warning as a failure."""
        source = Path(typ_path).resolve()
        if not source.is_file():
            raise TypstError(f"Typst source does not exist: {source}")
        target = Path(pdf_path).resolve() if pdf_path else source.with_suffix(".pdf")
        target.parent.mkdir(parents=True, exist_ok=True)
        prefix, version = self.toolchain
        # A native binary needs the executable name; the wrapper is a module.
        launcher = [prefix, "-m", "typst"] if prefix else ["typst"]
        command = [*launcher, "compile", str(source), str(target)]
        if self.root is not None:
            command.extend(["--root", str(self.root)])
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=self.timeout_s, check=False, cwd=str(source.parent))
        except subprocess.TimeoutExpired:
            return CompileResult(False, target, prefix or "typst", version, None, 0, (),
                                 f"Typst compilation timed out after {self.timeout_s:g}s")
        except OSError as exc:
            raise TypstError(f"Could not execute Typst: {exc}") from exc

        warnings = tuple(
            line.strip() for line in (completed.stderr or "").splitlines()
            if line.strip() and not line.strip().lower().startswith(_BENIGN_STDERR)
        )
        exists = target.is_file() and target.stat().st_size > 0
        success = completed.returncode == 0 and exists and not warnings
        if completed.returncode != 0:
            error = ((completed.stderr or completed.stdout or "").strip() or
                     f"typst exited with code {completed.returncode}")[-2000:]
        elif not exists:
            error = "Typst reported success but produced no PDF"
        elif warnings:
            error = f"Typst emitted {len(warnings)} warning(s); a clean build emits none"
        else:
            error = None
        return CompileResult(success, target, prefix or "typst", version,
                             _count_pages(target) if exists else None,
                             target.stat().st_size if exists else 0,
                             warnings, error, completed.stdout or "", completed.stderr or "")
