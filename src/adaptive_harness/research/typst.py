"""Typst toolchain resolution and PDF compilation.

Typst is resolved in three escalating steps — a native binary on ``PATH``, the
``typst`` PyPI package used as an in-process **binding**, then a reported failure.
Compilation treats **any** Typst diagnostic on stderr as a build break, because
a paper that "compiled with warnings" has an unresolved reference or a bad
figure path, and both are defects the invariant gate must reject.

The two engines are not interchangeable, and the difference matters:

* A native binary is driven as a subprocess. ``--root`` is a *search path*: a
  relative ``image("figures/x.png")`` resolves against it.
* The PyPI ``typst`` package is a compiled extension module, not a CLI. It
  cannot be launched with ``python -m typst`` (it ships no ``__main__``), so it
  is driven through a short runner that is still a subprocess — that keeps the
  timeout and crash isolation of the binary path. Its ``root`` is only a
  *containment* gate; relative assets resolve against the input file's own
  directory.

``TypstCompiler`` therefore refuses to guess: when a caller asks for a ``root``
that is not the source's directory, the binding cannot honour that search path,
and silently compiling against a different root would be a wrong proof. The
native binary is always preferred, and the divergence is reported rather than
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
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


@dataclass(frozen=True)
class TypstToolchain:
    """A resolved way of running the Typst compiler.

    ``kind`` is ``"binary"`` when a native executable was found on ``PATH`` and
    ``"binding"`` when the ``typst`` PyPI extension module is used instead. The
    two carry different search-path semantics, so the kind is recorded rather
    than flattened away.
    """

    kind: str
    version: str
    executable: str = ""

    @property
    def label(self) -> str:
        """A short provenance string for proof receipts."""
        return self.executable if self.kind == "binary" else "typst-python"

    @property
    def uses_root_as_search_path(self) -> bool:
        """Whether ``root`` adds a directory that relative assets resolve against."""
        return self.kind == "binary"


# Typst writes progress notes to stderr that are not diagnostics.
_BENIGN_STDERR = ("reading", "writing", "compiling")

#: Pinned so an explicitly requested install is reproducible rather than
#: picking up whatever the index serves that day.
TYPST_BINDING_VERSION = "0.15.0"

#: Prefix marking the runner's single machine-readable line of stdout.
_RESULT_MARKER = "@@adaptive-harness-typst@@"

#: Driver for the binding engine. It is a subprocess so the compile inherits the
#: same timeout and crash isolation as the native binary; the marker line keeps
#: the JSON payload unambiguous even if the compiler prints anything itself.
_BINDING_RUNNER = f"""
import json, sys
import typst

source, target, root = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {{"error": None, "warnings": [], "version": getattr(typst, "__version__", "")}}
try:
    _, warnings = typst.compile_with_warnings(source, target, root=root or None)
    payload["warnings"] = [str(item) for item in (warnings or [])]
except Exception as exc:  # TypstError, or a hard failure inside the extension
    payload["error"] = f"{{type(exc).__name__}}: {{exc}}"
sys.stdout.write("{_RESULT_MARKER}" + json.dumps(payload) + "\\n")
"""


def _probe_binding() -> str | None:
    """Return the binding's version when the ``typst`` extension module imports.

    The module object itself is deliberately not returned: compilation runs in
    a subprocess, so the caller only needs to know that the engine is usable.
    """
    try:
        module = importlib.import_module("typst")
    except Exception:
        return None
    if not callable(getattr(module, "compile_with_warnings", None)):
        return None
    return str(getattr(module, "__version__", ""))


def _install_binding() -> bool:
    """Install the ``typst`` extension module into the running interpreter.

    Only ever reached when the caller passed ``allow_install=True``, which
    requires an explicit opt-in. A shipped CLI mutating the user's environment
    mid-run, with no prompt and no pin, is not acceptable: the version is
    pinned so a run cannot silently pick up a different build, and a failure
    reports rather than retrying.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"typst=={TYPST_BINDING_VERSION}"],
            capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    # A negative probe result is not cached by ``import``, but a stale finder
    # cache can hide a package installed a moment ago.
    importlib.invalidate_caches()
    return True


def resolve_toolchain(*, allow_install: bool = False) -> TypstToolchain:
    """Return a usable Typst toolchain, preferring the native binary."""
    binary = shutil.which("typst")
    if binary:
        try:
            probe = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                   timeout=30, check=False)
            if probe.returncode == 0:
                return TypstToolchain("binary", probe.stdout.strip(), binary)
        except (OSError, subprocess.TimeoutExpired):
            pass

    version = _probe_binding()
    if version is not None:
        return TypstToolchain("binding", version)
    if not allow_install:
        raise TypstError("Typst is not installed and automatic installation is disabled")
    if _install_binding():
        version = _probe_binding()
        if version is not None:
            return TypstToolchain("binding", version)
    raise TypstError(
        "Typst is unavailable. Install it with `cargo install typst-cli`, `brew install typst`, "
        "or `pip install typst`, then re-run the compilation step.")


def resolve_typst(*, allow_install: bool = False) -> tuple[str, str]:
    """Return ``(executable_prefix, version)`` for a usable Typst toolchain.

    The prefix is the native binary's path, and empty for the PyPI binding,
    which is a module rather than a program. Prefer :func:`resolve_toolchain`,
    which reports *which* engine answered; the tuple cannot carry that.
    """
    toolchain = resolve_toolchain(allow_install=allow_install)
    return (toolchain.executable if toolchain.kind == "binary" else ""), toolchain.version


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
    counts = [int(match.group(1)) for match in
              re.finditer(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data)]
    if counts:
        return max(counts)
    occurrences = data.count(b"/Type /Page\n") + data.count(b"/Type/Page/")
    return occurrences or None


class TypstCompiler:
    """Compile Typst sources into publication-grade PDFs."""

    def __init__(self, *, root: str | Path | None = None, allow_install: bool = False,
                 timeout_s: float = 300.0):
        self.root = Path(root).resolve() if root else None
        self.allow_install = allow_install
        self.timeout_s = timeout_s
        self._toolchain: TypstToolchain | None = None

    @property
    def toolchain(self) -> TypstToolchain:
        if self._toolchain is None:
            self._toolchain = resolve_toolchain(allow_install=self.allow_install)
        return self._toolchain

    def compile(self, typ_path: str | Path, pdf_path: str | Path | None = None) -> CompileResult:
        """Compile ``typ_path`` to PDF, treating any warning as a failure."""
        source = Path(typ_path).resolve()
        if not source.is_file():
            raise TypstError(f"Typst source does not exist: {source}")
        target = Path(pdf_path).resolve() if pdf_path else source.with_suffix(".pdf")
        target.parent.mkdir(parents=True, exist_ok=True)
        toolchain = self.toolchain
        if toolchain.kind == "binary":
            return self._compile_binary(toolchain, source, target)
        return self._compile_binding(toolchain, source, target)

    # -- engines -------------------------------------------------------------

    def _run(self, command: list[str], source: Path) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(command, capture_output=True, text=True,
                                  timeout=self.timeout_s, check=False, cwd=str(source.parent))
        except subprocess.TimeoutExpired as exc:
            raise _CompilationTimedOut(
                f"Typst compilation timed out after {self.timeout_s:g}s") from exc

    def _compile_binary(self, toolchain: TypstToolchain, source: Path,
                        target: Path) -> CompileResult:
        command = [toolchain.executable, "compile", str(source), str(target)]
        if self.root is not None:
            command.extend(["--root", str(self.root)])
        try:
            completed = self._run(command, source)
        except _CompilationTimedOut as timed_out:
            return self._failed(target, toolchain, str(timed_out))
        except OSError as exc:
            raise TypstError(f"Could not execute Typst: {exc}") from exc

        warnings = tuple(
            line.strip() for line in (completed.stderr or "").splitlines()
            if line.strip() and not line.strip().lower().startswith(_BENIGN_STDERR)
        )
        error = self._verdict(completed.returncode, warnings, completed.stderr,
                              completed.stdout, target)
        return self._settle(target, toolchain, warnings, error,
                            completed.stdout or "", completed.stderr or "")

    def _compile_binding(self, toolchain: TypstToolchain, source: Path,
                         target: Path) -> CompileResult:
        divergence = self._root_divergence(source)
        if divergence:
            # Compiling anyway could resolve ``image(...)`` to a different file
            # than the native binary would, which for a proof pipeline means a
            # PDF that looks verified but shows the wrong figure.
            return self._failed(target, toolchain, divergence)
        command = [sys.executable, "-c", _BINDING_RUNNER, str(source), str(target),
                   str(self.root) if self.root else ""]
        try:
            completed = self._run(command, source)
        except _CompilationTimedOut as timed_out:
            return self._failed(target, toolchain, str(timed_out))
        except OSError as exc:
            raise TypstError(f"Could not execute Typst: {exc}") from exc

        payload = _parse_binding_stdout(completed.stdout)
        if payload is None:
            detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            return self._failed(target, toolchain,
                                f"Typst binding produced no result: {detail or 'no output'}")

        version = str(payload.get("version") or toolchain.version)
        reported = payload.get("error")
        warnings = tuple(str(item) for item in (payload.get("warnings") or []))
        if reported:
            return self._failed(target, TypstToolchain("binding", version),
                                str(reported)[:2000], warnings)
        if warnings:
            error = f"Typst emitted {len(warnings)} warning(s); a clean build emits none"
        else:
            error = None
        return self._settle(target, TypstToolchain("binding", version), warnings, error, "", "")

    # -- shared result handling ---------------------------------------------

    def _root_divergence(self, source: Path) -> str | None:
        """Report why the binding engine cannot honour this ``root``.

        The binding resolves a relative asset against the input file's own
        directory, while a native binary resolves it against ``--root``. The two
        agree only when the source sits directly in the root.
        """
        if self.root is None or source.parent == self.root:
            return None
        return (f"Typst root {self.root} is not the source directory {source.parent}, so the "
                f"Python binding would resolve relative assets differently than the native "
                f"typst binary; install the native CLI (cargo install typst-cli) to compile "
                f"this document safely")

    @staticmethod
    def _verdict(returncode: int, warnings: tuple[str, ...], stderr: str, stdout: str,
                 target: Path) -> str | None:
        """Decide the reported error for a finished native invocation."""
        exists = target.is_file() and target.stat().st_size > 0
        if returncode != 0:
            return ((stderr or stdout or "").strip() or
                    f"typst exited with code {returncode}")[-2000:]
        if not exists:
            return "Typst reported success but produced no PDF"
        if warnings:
            return f"Typst emitted {len(warnings)} warning(s); a clean build emits none"
        return None

    def _settle(self, target: Path, toolchain: TypstToolchain, warnings: tuple[str, ...],
                error: str | None, stdout: str, stderr: str) -> CompileResult:
        """Turn a finished invocation into a verified result.

        A build that reported any problem is discarded outright rather than
        measured: whatever sits at ``target`` is then either a partial write or
        a leftover from an earlier run, and reporting its page count as this
        build's output would be a false receipt. A build that reported success
        but left no PDF is treated as the failure it is.
        """
        if error is None and not (target.is_file() and target.stat().st_size > 0):
            error = "Typst reported success but produced no PDF"
        if error is not None:
            _discard_partial(target)
            return CompileResult(False, target, toolchain.label, toolchain.version,
                                 None, 0, warnings, error, stdout, stderr)
        return CompileResult(True, target, toolchain.label, toolchain.version,
                             _count_pages(target), target.stat().st_size, warnings,
                             None, stdout, stderr)

    def _failed(self, target: Path, toolchain: TypstToolchain, error: str,
                warnings: tuple[str, ...] = ()) -> CompileResult:
        """A failed build must never leave a stale PDF that looks like a result."""
        _discard_partial(target)
        return CompileResult(False, target, toolchain.label, toolchain.version,
                             None, 0, warnings, error)


class _CompilationTimedOut(RuntimeError):
    """Internal signal so a timeout is a failed build, not a crash."""


def _discard_partial(target: Path) -> None:
    """Remove a half-written or stale PDF so it cannot be mistaken for output."""
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass


def _parse_binding_stdout(stdout: str) -> dict[str, Any] | None:
    """Read the runner's JSON payload, tolerating any compiler chatter."""
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_RESULT_MARKER):
            try:
                payload = json.loads(line[len(_RESULT_MARKER):])
            except ValueError:
                return None
            return payload if isinstance(payload, dict) else None
    return None
