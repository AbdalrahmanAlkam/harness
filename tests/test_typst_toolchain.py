"""Regressions for Typst toolchain resolution and PDF compilation.

The previous resolver probed the ``typst`` PyPI package as if it were a program
(``python -m typst``). That package is a compiled extension module with no
``__main__``, so the probe could never succeed: the resolver installed the
package, believed it worked, and then raised ``TypstError`` anyway. On any
machine without a native ``typst`` binary every paper build failed, which is why
these cases are pinned directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.research import typst as typst_module
from adaptive_harness.research.typst import (
    TypstCompiler,
    TypstError,
    TypstToolchain,
    resolve_toolchain,
    resolve_typst,
)

VALID = "#set page(width: 200pt, height: 100pt)\n= Heading\n\nBody *bold*.\n"
BROKEN = '#set page(width: 200pt, height: 100pt)\n#panic("boom")\n'
# A duplicate label is not an error, but Typst reports it as a warning, and a
# build that emits any warning is a build break for this project.
WARNING = "#set page(width: 200pt, height: 100pt)\n#label(\"dup\")\n#label(\"dup\")\n"


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# -- toolchain discovery ---------------------------------------------------


def test_the_pypi_typst_is_used_as_a_binding_not_as_a_program(monkeypatch, tmp_path: Path):
    """The PyPI package ships no ``__main__``; probing it as a CLI always fails."""
    monkeypatch.setattr(typst_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(typst_module, "_probe_binding", lambda: "0.15.0")

    toolchain = resolve_toolchain(allow_install=False)
    assert toolchain.kind == "binding"
    assert toolchain.version == "0.15.0"
    # A binding has no executable, and must never claim root is a search path.
    assert toolchain.executable == ""
    assert toolchain.uses_root_as_search_path is False
    assert toolchain.label == "typst-python"


def test_a_native_binary_is_preferred_over_the_binding(monkeypatch, tmp_path: Path):
    binary = tmp_path / "typst"
    binary.write_text("#!/bin/sh\necho 'typst 0.13.1'\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(typst_module.shutil, "which", lambda name: str(binary))

    toolchain = resolve_toolchain(allow_install=False)
    assert toolchain.kind == "binary"
    assert toolchain.version == "typst 0.13.1"
    assert toolchain.executable == str(binary)
    assert toolchain.uses_root_as_search_path is True


def test_a_broken_binary_falls_back_instead_of_being_trusted(monkeypatch, tmp_path: Path):
    """A binary on PATH that cannot report a version is not a usable toolchain."""
    binary = tmp_path / "typst"
    binary.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(typst_module.shutil, "which", lambda name: str(binary))
    monkeypatch.setattr(typst_module, "_probe_binding", lambda: "0.15.0")

    assert resolve_toolchain(allow_install=False).kind == "binding"


def test_an_unusable_environment_reports_how_to_install(monkeypatch):
    monkeypatch.setattr(typst_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(typst_module, "_probe_binding", lambda: None)
    monkeypatch.setattr(typst_module, "_install_binding", lambda: False)

    with pytest.raises(TypstError, match="pip install typst"):
        resolve_toolchain(allow_install=True)
    # With installation disabled the message must not promise to install.
    with pytest.raises(TypstError, match="automatic installation is disabled"):
        resolve_toolchain(allow_install=False)


def test_the_legacy_tuple_helpers_still_work():
    """``resolve_typst`` is a public export; the tuple must stay meaningful."""
    prefix, version = resolve_typst(allow_install=False)
    assert version
    assert prefix in ("", ) or Path(prefix).name == "typst"


# -- compiling -------------------------------------------------------------


def test_a_clean_document_produces_a_real_pdf(tmp_path: Path):
    compiler = TypstCompiler(root=tmp_path, allow_install=False)
    result = compiler.compile(_write(tmp_path, "ok.typ", VALID), tmp_path / "ok.pdf")
    if result.error and "unavailable" in result.error:
        pytest.skip("no Typst engine available on this machine")
    assert result.success, result.error
    assert result.size_bytes > 0
    assert (tmp_path / "ok.pdf").is_file()


def test_a_failing_document_leaves_no_pdf_behind(tmp_path: Path):
    """A stale PDF from an earlier run must never be reported as this build's."""
    target = tmp_path / "paper.pdf"
    target.write_bytes(b"%PDF-1.4 stale bytes from a previous successful build")

    compiler = TypstCompiler(root=tmp_path, allow_install=False)
    result = compiler.compile(_write(tmp_path, "bad.typ", BROKEN), target)
    if result.error and "unavailable" in result.error:
        pytest.skip("no Typst engine available on this machine")
    assert not result.success
    assert result.error
    assert result.size_bytes == 0
    assert not target.exists(), "a failed build must delete the stale PDF"


def test_a_warning_is_a_build_break_and_removes_the_output(tmp_path: Path):
    compiler = TypstCompiler(root=tmp_path, allow_install=False)
    result = compiler.compile(_write(tmp_path, "warn.typ", WARNING), tmp_path / "warn.pdf")
    if result.error and "unavailable" in result.error:
        pytest.skip("no Typst engine available on this machine")
    assert not result.success
    assert "warning" in (result.error or "").lower()
    assert result.warnings
    assert not (tmp_path / "warn.pdf").exists()


def test_a_timeout_is_a_failed_build_not_a_crash(tmp_path: Path):
    compiler = TypstCompiler(root=tmp_path, allow_install=False, timeout_s=0.001)
    source = _write(tmp_path, "slow.typ", VALID + "x " * 20000)
    result = compiler.compile(source, tmp_path / "slow.pdf")
    assert not result.success
    assert "timed out" in (result.error or "").lower()


def test_a_missing_source_is_reported_clearly(tmp_path: Path):
    compiler = TypstCompiler(allow_install=False)
    with pytest.raises(TypstError, match="does not exist"):
        compiler.compile(tmp_path / "absent.typ")


def test_a_root_the_binding_cannot_honour_is_refused(tmp_path: Path):
    """Silently compiling against a different root could show the wrong figure."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    source = _write(source_dir, "paper.typ", VALID)

    compiler = TypstCompiler(root=root, allow_install=False)
    if resolve_toolchain(allow_install=False).kind == "binary":
        pytest.skip("a native binary honours --root as a search path")
    result = compiler.compile(source, tmp_path / "paper.pdf")
    assert not result.success
    assert "root" in (result.error or "").lower()
    assert not (tmp_path / "paper.pdf").exists()


def test_the_toolchain_is_resolved_once_and_reused(tmp_path: Path):
    calls = []

    def counting(allow_install: bool = True) -> TypstToolchain:
        calls.append(allow_install)
        return TypstToolchain("binding", "0.15.0")

    compiler = TypstCompiler(root=tmp_path, allow_install=False)
    compiler._toolchain = None
    original = typst_module.resolve_toolchain
    try:
        typst_module.resolve_toolchain = counting
        compiler.toolchain
        compiler.toolchain
    finally:
        typst_module.resolve_toolchain = original
    assert len(calls) == 1
