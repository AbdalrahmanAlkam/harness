"""Language tooling that degrades honestly when no language server is installed.

A real LSP client speaks a JSON-RPC protocol to a background server per
language. That is a large dependency, a process per language, and a machine
that often has no server installed at all. Rather than require it, this plugin
does the part that is always possible and **says which part it did**:

- **syntax**, with the tools that ship with the machine: Python is parsed
  in-process with :mod:`ast`; JavaScript is checked with ``node --check`` when
  ``node`` is on ``PATH``.
- **definitions, references and hover**, by parsing the workspace: an AST walk
  for Python, a bounded regular-expression search for anything else.

None of that is a language server. There is no cross-file type inference, no
"go to definition" through an import that resolves at runtime, no completion.
The tools say so in their output rather than presenting a regex hit as if a
compiler had confirmed it, because the failure this plugin exists to avoid is
the quiet one: a tool that looks like a compiler and is a grep.

**The rule about absence.** ``lsp_status`` enumerates the servers it knows
about and reports each as installed or missing, by name and path. If a server
is missing, the diagnostics tool says so in the same breath as its result, and
the definition tools work anyway because they never needed one. Nothing here
substitutes its own output for a check that did not run.

**The subprocess permission, honestly.** It is used for exactly two things,
both visible in this file: ``node --check`` for JavaScript, and nothing else by
default. A language-server binary is only ever run when the user has named one
explicitly, via ``--server``/``ADAPTIVE_HARNESS_LSP_SERVER``; it is never run
just because one was found on ``PATH``. The one other thing this plugin reads
from the environment is ``PATH``, and only through :func:`shutil.which`, which
is the same lookup the harness itself does when it runs ``python``. No other
variable is read, which is why the manifest does not claim ``env``.

**Testability.** The parser, the server probe and the process runner are all
module-level and replaceable, so a test can assert what the tools do with a
missing server without needing a missing server on the machine.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --- limits ----------------------------------------------------------------

#: Directories a workspace walk never enters. Vendored and generated code
#: would otherwise dominate a symbol search with definitions of its own.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".harness", "dist", "build", ".eggs", "site-packages", ".next", ".cache",
})

#: A file bigger than this is not read for a symbol search. The answer is the
#: same either way -- a definition buried past 512 KB of vendored table is not
#: what was asked for -- and the cost is not.
MAX_FILE_BYTES = 512 * 1024

#: Ceilings on what a search returns. A search with no bound is a way to fill
#: the context window with one symbol's call sites.
MAX_DEFINITIONS = 40
MAX_REFERENCES = 80
MAX_DIAGNOSTICS = 40

#: How far up the tree the plugin looks for a project root, in case of a deep
#: file inside a checkout.
MAX_ROOT_WALK = 12

#: Wall-clock bound on ``node --check``. A file that takes longer than this is
#: pathological, and a hung subprocess is worse than no diagnostic.
SUBPROCESS_TIMEOUT_S = 10.0

#: Markers that identify the root of a project. If none is found above the
#: file, the file's own directory is used and the result says so.
ROOT_MARKERS = (".git", "pyproject.toml", "package.json", "Cargo.toml",
                "go.mod", "setup.py", ".harness")

#: Suffix to language, and the text extensions a symbol search reads for each.
LANGUAGES: Dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".sh": "shell", ".md": "markdown", ".toml": "toml", ".json": "json",
}


def language_of(path: Path) -> str:
    """The language name for a path, or ``"unknown"``."""
    return LANGUAGES.get(path.suffix.lower(), "unknown")


# --- the language-server table --------------------------------------------


class ServerSpec:
    """One language server the plugin knows how to name and find."""

    def __init__(self, key: str, languages: str, binaries: Sequence[str],
                 install: str) -> None:
        self.key = key
        self.languages = languages
        #: Tried in order. A server is usually installed under one of several
        #: names, and reporting only the first is how "not installed" gets
        #: said about something that is.
        self.binaries = tuple(binaries)
        self.install = install

    def locate(self) -> Optional[str]:
        """The path to the server, or ``None`` if it is not installed."""
        for name in self.binaries:
            found = _WHICH(name)
            if found:
                return found
        return None


#: The single seam for executable lookup. Replaced in tests.
_WHICH = shutil.which

SERVERS: Tuple[ServerSpec, ...] = (
    ServerSpec("pylsp", "python", ("pylsp", "python-lsp-server"),
               "pip install python-lsp-server"),
    ServerSpec("pyright", "python,typescript", ("pyright-langserver", "pyright",
                                               "basedpyright-langserver",
                                               "basedpyright"),
               "npm install -g pyright"),
    ServerSpec("typescript-language-server", "typescript",
               ("typescript-language-server",),
               "npm install -g typescript-language-server typescript"),
    ServerSpec("gopls", "go", ("gopls",), "go install golang.org/x/tools/gopls@latest"),
    ServerSpec("rust-analyzer", "rust", ("rust-analyzer",),
               "rustup component add rust-analyzer"),
    ServerSpec("clangd", "c,cpp", ("clangd",), "apt install clangd"),
    ServerSpec("jdtls", "java", ("jdtls", "eclipse.jdt.ls"),
               "install the Eclipse JDT Language Server"),
    ServerSpec("solargraph", "ruby", ("solargraph",), "gem install solargraph"),
    ServerSpec("lua-language-server", "lua", ("lua-language-server",),
               "install Lua Language Server"),
    ServerSpec("deno", "typescript,javascript", ("deno",), "install Deno"),
)

def _explicit_server() -> str:
    """A server the user named, or ``""``.

    The only configuration this plugin has, and the only path on which a
    language-server binary is ever executed. ``shutil.which`` finding one is
    *not* permission to run it: a program that happens to be installed is not a
    program the user asked this tool to drive.
    """
    return (os.environ.get("ADAPTIVE_HARNESS_LSP_SERVER", "") or "").strip()


# --- workspace -------------------------------------------------------------


def _resolve_path(raw: Any) -> Path:
    """Turn a tool argument into an absolute path.

    Absolute input is kept as given; relative input is resolved against the
    current working directory, which is the workspace. The host has already
    confined the argument by the time a tool is called through it; this is the
    same resolution for a handler invoked directly.
    """
    text = str(raw or "").strip()
    if not text:
        return Path.cwd()
    candidate = Path(text).expanduser()
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate)


def project_root(start: Path) -> Path:
    """The nearest ancestor of ``start`` that looks like a project root.

    Falls back to ``start`` itself, and the tools report that they used the
    file's own directory, because a search that quietly scanned a subtree of a
    checkout and reported the result as *the workspace* would be a search whose
    scope the caller cannot see.
    """
    current = start if start.is_dir() else start.parent
    for _ in range(MAX_ROOT_WALK):
        for marker in ROOT_MARKERS:
            if (current / marker).exists():
                return current
        if current.parent == current:
            break
        current = current.parent
    return start if start.is_dir() else start.parent


def _search_files(root: Path, target: Path) -> Tuple[List[Path], bool]:
    """The files a search for ``target``'s symbol should read, and whether it is AST.

    A symbol crosses file boundaries within a *language*, not within one
    extension: ``greet`` defined in a ``.py`` is not referenced from a ``.pyi``
    only incidentally, and a JavaScript caller of a ``.ts`` function is exactly
    what a search should find. So for a non-Python language the search covers
    every suffix of that language, falling back to the file's own suffix when
    the language has none registered -- an unregistered language searches
    exactly what the caller pointed at, and no more.
    """
    language = language_of(target)
    used_ast = language == "python"
    if used_ast:
        return _walk(root, (target.suffix.lower(),)), True
    suffixes = tuple(s for s, lang in LANGUAGES.items() if lang == language)
    if not suffixes:
        return _walk(root, (target.suffix.lower(),)), False
    files = _walk(root, suffixes)
    if not files:
        # A workspace of ``.js`` with a ``.ts`` file opened in the editor: the
        # language matched nothing, so fall back rather than report "not found"
        # for a file that was never read.
        files = _walk(root, (target.suffix.lower(),))
    return files, False


def _walk(root: Path, suffixes: Iterable[str]) -> List[Path]:
    """Every file under ``root`` with one of ``suffixes``, in a stable order.

    Bounded by :data:`MAX_FILE_BYTES` and by the skip list, because a symbol
    search that descends into ``node_modules`` is a search that never returns.
    """
    wanted = {suffix.lower() for suffix in suffixes}
    found: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not any(part in SKIP_DIRS for part in path.parts):
            if path.is_file() and path.suffix.lower() in wanted:
                try:
                    if path.stat().st_size <= MAX_FILE_BYTES:
                        found.append(path)
                except OSError:
                    continue
    return found


def _read(path: Path) -> str:
    """Read a source file, or return ``""`` if it cannot be read.

    A file that will not read is skipped rather than fatal: one unreadable file
    in a workspace of a thousand should not fail a search.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# --- python analysis -------------------------------------------------------


class _PythonIndex(ast.NodeVisitor):
    """Definitions in one parsed Python file.

    Walks rather than reaching for ``symtable``: ``symtable`` sees only
    scopes, and a symbol's definition is a node in the tree with a line number
    attached, which is exactly what a caller wants to jump to.
    """

    def __init__(self, path: Path, root: Path, symbol: str) -> None:
        self.path = path
        self.root = root
        self.symbol = symbol
        self.definitions: List[Dict[str, Any]] = []
        self.references: List[Dict[str, Any]] = []

    # -- helpers
    def _lines(self) -> List[str]:
        return _read(self.path).splitlines()

    def _record_definition(self, name: str, node: ast.AST, kind: str) -> None:
        if name != self.symbol:
            return
        line = getattr(node, "lineno", 0)
        self.definitions.append({
            "file": _relative(self.path, self.root),
            "line": line,
            "kind": kind,
            "signature": _python_signature(node),
            "source": _line_at(self._lines(), line),
            "docstring": _docstring(node),
        })

    # -- visitors
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._record_definition(node.name, node, "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._record_definition(node.name, node, "async function")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._record_definition(node.name, node, "class")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id == self.symbol and isinstance(node.ctx, (ast.Load, ast.Store)):
            self.references.append(self._use(node, node.id))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # ``obj.symbol`` counts as a reference to the symbol.
        if node.attr == self.symbol and isinstance(node.ctx, ast.Load):
            self.references.append(self._use(node, node.attr))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        # An import brings a name into scope, which is a reference to it. The
        # binding is an ``ast.alias`` and never appears as a Name, so without
        # this a symbol's very first use -- the line that brought it in -- is
        # the one line missing from its references.
        module = ("." * (node.level or 0)) + (node.module or "")
        for alias in node.names:
            bound = alias.asname or alias.name
            if bound != self.symbol:
                continue
            self.references.append({
                "file": _relative(self.path, self.root), "line": node.lineno,
                "kind": "import", "name": bound, "from": module,
                "source": _line_at(self._lines(), node.lineno)})
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            if bound != self.symbol:
                continue
            self.references.append({
                "file": _relative(self.path, self.root), "line": node.lineno,
                "kind": "import", "name": bound, "from": alias.name,
                "source": _line_at(self._lines(), node.lineno)})
        self.generic_visit(node)

    def _use(self, node: ast.AST, shown: str) -> Dict[str, Any]:
        line = getattr(node, "lineno", 0)
        return {"file": _relative(self.path, self.root), "line": line,
                "kind": "use", "name": shown,
                "source": _line_at(self._lines(), line)}

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record_definition(target.id, node, "assignment")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if isinstance(node.target, ast.Name):
            self._record_definition(node.target.id, node, "annotated assignment")
        self.generic_visit(node)


def _line_at(lines: Sequence[str], number: int) -> str:
    """The 1-indexed line, trimmed. ``""`` when out of range."""
    if 1 <= number <= len(lines):
        return lines[number - 1].strip()[:200]
    return ""


def _docstring(node: ast.AST) -> str:
    """A node's docstring, or ``""``."""
    try:
        return (ast.get_docstring(node) or "").strip()[:600]
    except TypeError:
        return ""


def _python_signature(node: ast.AST) -> str:
    """A one-line signature for a def or class node.

    Reconstructed from the tree rather than read back from the source, so a
    definition is described the same way whether it was found by the AST or by
    the regex path.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        returns = f" -> {_unparse(node.returns)}" if node.returns else ""
        # ``ast.unparse`` on the arguments node keeps annotations, defaults and
        # ``*``/``**`` markers, which is what makes the signature worth reading.
        # A definition described without its annotations is a different function
        # to the caller, so the fallback keeps them when it can.
        try:
            arguments = ast.unparse(node.args)
        except Exception:  # noqa: BLE001 - a description must never fail a search
            arguments = ", ".join(arg.arg for arg in node.args.args)
        return f"{prefix} {node.name}({arguments}){returns}"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(_unparse(base) for base in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    if isinstance(node, ast.AnnAssign):
        annotation = f": {_unparse(node.annotation)}" if node.annotation else ""
        return f"{getattr(node.target, 'id', '?')}{annotation} = ..."
    if isinstance(node, ast.Assign):
        targets = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name))
        return f"{targets} = ..."
    return ""


def _unparse(node: Optional[ast.AST]) -> str:
    """``ast.unparse`` where available, and something readable where not."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - a description must never fail a search
        return getattr(node, "id", "") or type(node).__name__


def _parse_python(path: Path) -> Optional[ast.AST]:
    """Parse a Python file, or return ``None`` if it does not parse.

    A file with a syntax error is a *result*, not a failure of this tool: the
    caller asked where a symbol is defined, and a file that cannot be parsed
    cannot answer. It is skipped and counted.
    """
    text = _read(path)
    if not text:
        return None
    try:
        return ast.parse(text, filename=str(path))
    except (SyntaxError, ValueError, RecursionError):
        return None


# --- non-python analysis ---------------------------------------------------

#: Definition patterns for the languages with no parser here. Deliberately
#: conservative: a match is reported as a *candidate* line, not as a confirmed
#: definition, because a regex cannot know what a language's grammar says.
_DEF_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)")),
    ("class", re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)")),
    ("assignment", re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[:=]")),
    ("assignment", re.compile(r"^\s*(\w+)\s*:=\s*")),
    ("method", re.compile(r"^\s{2,}(?:(?:public|private|static|async|get|set)\s+)*"
                          r"(\w+)\s*\([^)]*\)\s*\{")),
    ("function", re.compile(r"^\s*func\s+(\w+)")),
    ("type", re.compile(r"^\s*type\s+(\w+)")),
    ("function", re.compile(r"^\s*fn\s+(\w+)")),
    ("struct", re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)")),
    ("class", re.compile(r"^\s*(?:public\s+|final\s+|abstract\s+)*class\s+(\w+)")),
    ("function", re.compile(r"^\s*(?:def|function|defn|sub)\s+(\w+)")),
    ("function", re.compile(r"^\s*(\w+)\s*\(\)\s*\{")),
    ("variable", re.compile(r"^\s*(?:set|local|export)\s+(?:-\w+\s+)*(\w+)[= ]")),
)


def _regex_definitions(path: Path, root: Path, symbol: str) -> List[Dict[str, Any]]:
    """Candidate definitions of ``symbol`` in one non-Python file."""
    found: List[Dict[str, Any]] = []
    for number, line in enumerate(_read(path).splitlines(), start=1):
        if len(line) > 400:
            continue
        for kind, pattern in _DEF_PATTERNS:
            match = pattern.match(line)
            if match and match.group(1) == symbol:
                found.append({"file": _relative(path, root), "line": number,
                              "kind": f"{kind} (regex, unconfirmed)", "signature": "",
                              "source": line.strip()[:200], "docstring": ""})
                break
    return found


def _regex_references(path: Path, root: Path, symbol: str) -> List[Dict[str, Any]]:
    """Every word-boundary occurrence of ``symbol`` in one non-Python file."""
    if not re.fullmatch(r"\w+", symbol):
        return []  # a regex-special symbol would match its own metacharacters
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    found: List[Dict[str, Any]] = []
    for number, line in enumerate(_read(path).splitlines(), start=1):
        if pattern.search(line):
            found.append({"file": _relative(path, root), "line": number,
                          "kind": "text match (unconfirmed)", "name": symbol,
                          "source": line.strip()[:200]})
    return found


# --- diagnostics -----------------------------------------------------------

#: The single seam for running an external program. Replaced in tests.
_RUNNER = subprocess.run


def _node_check(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Run ``node --check`` on a JavaScript file.

    Returns ``(diagnostic, note)``. ``diagnostic`` is ``None`` when the file is
    clean; ``note`` explains an absence -- node not installed, or a process
    that could not be run -- so the caller can say why rather than reporting a
    clean file it never checked.
    """
    node = _WHICH("node")
    if not node:
        return None, ("node is not on PATH, so this JavaScript file was not "
                      "checked at all -- not checked is not the same as clean")
    try:
        outcome = _RUNNER([node, "--check", str(path)], capture_output=True,
                          timeout=SUBPROCESS_TIMEOUT_S, text=True, check=False)
    except subprocess.TimeoutExpired:
        return None, (f"node --check did not finish within {SUBPROCESS_TIMEOUT_S:g}s "
                      f"on {path.name}, so it was not checked")
    except OSError as exc:
        return None, f"node --check could not be run ({exc}); {path.name} was not checked"

    if outcome.returncode == 0:
        return None, ""
    message = (outcome.stderr or outcome.stdout or "").strip()
    first = message.splitlines()[0] if message else f"node --check exited {outcome.returncode}"
    line_number = 0
    # node's error format is "path:line" on the first line, then the source.
    header = message.splitlines()[0] if message else ""
    column = header.rfind(":")
    if column > 0 and header[:column].rstrip(":").isdigit():
        line_number = int(header[:column].rstrip(":"))
    return {"line": line_number, "message": first.strip()[:400], "source": ""}, ""


def _python_diagnostics(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Parse a Python file and return every syntax error it has.

    Uses :func:`ast.parse`, which is what ``python -m py_compile`` runs
    underneath. The subprocess form is not used: it writes a ``__pycache__``
    entry next to the user's source as a side effect of *asking whether the
    file has an error*, and a diagnostic tool that leaves bytecode behind is a
    tool that mutates a repository to report on it.
    """
    text = _read(path)
    if not text:
        return [], f"{path.name} could not be read, so it was not checked"
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return ([{"line": exc.lineno or 0, "column": exc.offset or 0,
                  "message": exc.msg or "syntax error",
                  "source": (exc.text or "").strip()[:200]}]), ""
    except ValueError as exc:
        return [{"line": 0, "column": 0, "message": f"could not parse: {exc}",
                 "source": ""}], ""
    return [], ""


def _server_availability() -> List[Dict[str, Any]]:
    """One row per known language server: installed at this path, or missing."""
    rows: List[Dict[str, Any]] = []
    for spec in SERVERS:
        path = spec.locate()
        rows.append({"server": spec.key, "languages": spec.languages,
                     "installed": bool(path), "path": path or "",
                     "tried": ", ".join(spec.binaries), "install": spec.install})
    return rows


def _configured_server() -> Dict[str, Any]:
    """The server the user named, resolved. Never probed implicitly."""
    named = _explicit_server()
    if not named:
        return {"configured": "", "available": False,
                "note": ("No language server is configured. Set "
                         "ADAPTIVE_HARNESS_LSP_SERVER to the binary you want "
                         "driven; until then this plugin never launches one.")}
    path = _WHICH(named) or (named if Path(named).is_file() else None)
    return {"configured": named, "available": bool(path), "path": path or "",
            "note": ("" if path else
                     f"{named!r} was named but no such program is on PATH")}


# --- rendering -------------------------------------------------------------


def _render_diagnostics(path: Path, language: str, checker: str,
                        diagnostics: Sequence[Dict[str, Any]], note: str,
                        root: Path) -> str:
    """The text a model reads. Provenance first, then the findings."""
    lines = [f"{_relative(path, root)} -- {language}, checked with {checker}"]
    if note:
        lines.append(f"NOTE: {note}")
    if not diagnostics:
        lines.append("No problems found by that check.")
        lines.append(
            "Scope of this claim: a syntax check only. It is not a type check, "
            "not a linter, and not a language server -- it will not see a name "
            "that is misspelled, a signature that does not match its call site, "
            "or an import that does not resolve.")
        return "\n".join(lines)
    lines.append(f"{len(diagnostics)} problem(s):")
    for item in diagnostics[:MAX_DIAGNOSTICS]:
        location = f"line {item.get('line', 0)}" if item.get("line") else "unknown line"
        if item.get("column"):
            location += f", column {item['column']}"
        lines.append(f"  {location}: {item.get('message', '')}")
        if item.get("source"):
            lines.append(f"    {item['source']}")
    if len(diagnostics) > MAX_DIAGNOSTICS:
        lines.append(f"  ... and {len(diagnostics) - MAX_DIAGNOSTICS} more.")
    lines.append("That is everything the check found; it is a syntax check only.")
    return "\n".join(lines)


def _render_definitions(symbol: str, found: Sequence[Dict[str, Any]],
                        root: Path, used_ast: bool, root_reason: str) -> str:
    lines = [f"Definitions of {symbol!r} under {root}{root_reason} "
             f"({'AST parse' if used_ast else 'regular expression'}, "
             f"not a language server):"]
    if not found:
        lines.append("")
        lines.append("  None found in the files that were searched.")
        lines.append("  This is a text-level search, not a language server, so a "
                     "definition behind a metaclass, a C extension, a code "
                     "generator or a dynamic import would not appear here. "
                     "'Not found' means not found by a parser, not proven absent.")
        return "\n".join(lines)
    for item in found[:MAX_DEFINITIONS]:
        lines.append(f"  {item['file']}:{item['line']}  {item['kind']}")
        if item.get("signature"):
            lines.append(f"      {item['signature']}")
        if item.get("docstring"):
            lines.append(f"      \"{item['docstring'].splitlines()[0][:160]}\"")
    if len(found) > MAX_DEFINITIONS:
        lines.append(f"  ... and {len(found) - MAX_DEFINITIONS} more, cut at the cap.")
    return "\n".join(lines)


def _render_references(symbol: str, found: Sequence[Dict[str, Any]], root: Path,
                       used_ast: bool) -> str:
    lines = [f"References to {symbol!r} under {root} "
             f"({'AST parse' if used_ast else 'regular expression'}, "
             f"not a language server):"]
    if not found:
        lines.append("  None found. As with definitions, this is a text-level "
                     "search: a call made through getattr, a string passed to "
                     "eval, or a template that builds the name at runtime would "
                     "not be seen.")
        return "\n".join(lines)
    for item in found[:MAX_REFERENCES]:
        lines.append(f"  {item['file']}:{item['line']}  {item.get('kind', 'use')}"
                     f"  {item.get('source', '')}")
    if len(found) > MAX_REFERENCES:
        lines.append(f"  ... and {len(found) - MAX_REFERENCES} more, cut at the cap.")
    return "\n".join(lines)


# --- the tools -------------------------------------------------------------


def lsp_status() -> Dict[str, Any]:
    """Report which language servers are installed, and which are not.

    A status that listed only what works would be the one report nobody can
    act on. Every known server is named, with the path it was found at or the
    command that would install it, and the tools that work without any server
    are named as such.
    """
    servers = _server_availability()
    configured = _configured_server()
    installed = [row for row in servers if row["installed"]]
    missing = [row for row in servers if not row["installed"]]

    lines = ["Language servers:"]
    for row in servers:
        mark = "installed" if row["installed"] else "NOT INSTALLED"
        lines.append(f"  [{mark}] {row['server']:<26} {row['languages']}"
                     + (f"  ->  {row['path']}" if row["path"]
                        else f"   (looked for: {row['tried']}; install: {row['install']})"))
    lines.append("")
    lines.append(f"{len(installed)} of {len(servers)} known language servers are "
                 f"installed on this machine.")
    if configured["configured"]:
        lines.append(f"Configured server: {configured['configured']} -- "
                     + (f"found at {configured['path']}." if configured["available"]
                        else configured["note"] + "."))
    else:
        lines.append(configured["note"])
    lines.append("")
    lines.append("What this plugin does without any language server:")
    js = _WHICH("node")
    lines.append("  lsp_diagnostics   syntax only: Python via the in-process ast "
                 "parser, JavaScript via `node --check`"
                 + (f" (node is at {js})." if js else ", and node is not on PATH, so "
                    "JavaScript is not checked at all."))
    lines.append("  lsp_definitions   an AST walk for Python, a regex for other "
                 "languages. Confirms the text, not the type.")
    lines.append("  lsp_references    the same, counting occurrences.")
    lines.append("  lsp_hover         the source around a line, and a docstring if "
                 "the file has one.")
    lines.append("")
    lines.append("None of those is cross-file type inference, completion, or "
                 "rename -- a regex hit is not a type check, and an AST parse of "
                 "one file is not a program-wide one. Install a server and it still "
                 "is not used unless you name one in ADAPTIVE_HARNESS_LSP_SERVER: "
                 "finding a binary on PATH is not permission to run it.")
    return {"success": True, "output": "\n".join(lines),
            "metadata": {"servers": servers, "installed": [r["server"] for r in installed],
                         "missing": [r["server"] for r in missing],
                         "configured_server": configured,
                         "any_server_installed": bool(installed)}}


def lsp_diagnostics(path: str) -> Dict[str, Any]:
    """Report syntax errors in a file, using whatever is on this machine.

    Python is parsed in process. JavaScript is checked with ``node --check``
    when node is installed, and when it is not, the result says the file was
    not checked rather than reporting it clean.
    """
    target = _resolve_path(path)
    if not target.is_file():
        return {"success": False, "error": f"No such file: {path}"}
    root = project_root(target)
    language = language_of(target)

    if language == "python":
        diagnostics, note = _python_diagnostics(target)
        checker = "python ast.parse (in process; no language server)"
        used_subprocess = False
        checked = True
    elif language == "javascript":
        node = _WHICH("node")
        if not node:
            return {"success": True,
                    "output": (f"{_relative(target, root)} -- javascript: node is not "
                               f"on PATH, so this file was NOT checked. No diagnostic is "
                               f"reported because none was produced, and not checked "
                               f"is not the same as clean. Install Node, or read the "
                               f"file."),
                    "metadata": {"language": language, "checker": "none (node absent)",
                                 "checked": False, "clean": False,
                                 "diagnostics": [], "used_subprocess": False,
                                 "file": str(target), "root": str(root)}}
        diagnostic, note = _node_check(target)
        diagnostics = [diagnostic] if diagnostic else []
        # A caveat that is not decoration: `node --check` classifies a `.js`
        # file itself, and on Node 22 a `.js` file whose first parse fails is
        # retried as an ES module -- so `export const x = ;` exits 0 there
        # while the same text in a `.mjs` file is reported. Reporting that
        # difference is more useful than silently inheriting it.
        note = note or (
            "node --check classified this file itself, and its verdict depends on "
            "how it did: on Node 22 a .js file that fails to parse as CommonJS is "
            "retried as an ES module, so some syntax errors in .js are not "
            "reported. Copy it to a .mjs file and check that to be sure."
            if target.suffix.lower() in (".js", ".cjs") else "")
        checker = "node --check"
        used_subprocess = True
        checked = True
    else:
        rows = [r for r in _server_availability()
                if language in r["languages"].split(",")]
        have = [r["server"] for r in rows if r["installed"]]
        detail = (f"a language server for {language} is installed ({', '.join(have)}) "
                  f"but this plugin does not speak the LSP protocol" if have else
                  f"no parser for {target.suffix or 'this file type'} is available "
                  f"here")
        return {"success": True,
                "output": (f"{_relative(target, root)} -- {language}: {detail}, so this "
                           f"file was NOT checked. No diagnostic is reported because "
                           f"none was produced; that is different from a clean file."),
                "metadata": {"language": language, "checker": "none",
                             "checked": False, "clean": False,
                             "diagnostics": [], "used_subprocess": False,
                             "file": str(target), "root": str(root)}}

    return {"success": True,
            "output": _render_diagnostics(target, language, checker, diagnostics,
                                          note, root),
            "metadata": {"language": language, "checker": checker, "checked": checked,
                         "clean": not diagnostics, "diagnostics": list(diagnostics),
                         "used_subprocess": used_subprocess, "file": str(target),
                         "root": str(root)}}


def lsp_definitions(path: str, symbol: str, directory: str = "") -> Dict[str, Any]:
    """Find where a symbol is defined, by parsing the workspace.

    Works with no language server: an AST walk for Python, a bounded regex for
    anything else. A regex match is reported as a *candidate*, because nothing
    here has parsed the grammar and the difference is stated in the result.
    """
    name = str(symbol or "").strip()
    if not name:
        return {"success": False, "error": "No symbol given."}
    target = _resolve_path(path)
    if not target.is_file():
        return {"success": False, "error": f"No such file: {path}"}

    if directory:
        root = _resolve_path(directory)
        if not root.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}
        reason = " (given explicitly)"
    else:
        root = project_root(target)
        reason = "" if any((root / marker).exists() for marker in ROOT_MARKERS) \
            else " (no project marker found, so the file's own directory was used)"

    language = language_of(target)
    files, used_ast = _search_files(root, target)

    definitions: List[Dict[str, Any]] = []
    unreadable = 0
    for candidate in files:
        if used_ast:
            tree = _parse_python(candidate)
            if tree is None:
                unreadable += 1
                continue
            index = _PythonIndex(candidate, root, name)
            index.visit(tree)
            definitions.extend(index.definitions)
        else:
            definitions.extend(_regex_definitions(candidate, root, name))
        if len(definitions) > MAX_DEFINITIONS * 2:
            break

    definitions.sort(key=lambda item: (item["file"], item["line"]))
    note = f"  ({unreadable} file(s) could not be parsed and were skipped)" if unreadable else ""
    output = _render_definitions(name, definitions, root, used_ast, reason) + note
    return {"success": True, "output": output,
            "metadata": {"symbol": name, "language": language, "used_ast": used_ast,
                         "root": str(root), "files_searched": len(files),
                         "unparseable": unreadable, "count": len(definitions),
                         "definitions": definitions[:MAX_DEFINITIONS]}}


def lsp_references(path: str, symbol: str, directory: str = "",
                   limit: int = 40) -> Dict[str, Any]:
    """Find where a symbol is used, by parsing the workspace.

    The same honest limit as lsp_definitions: a text-level search sees the
    occurrences it can see, and the result says that is all it saw.
    """
    name = str(symbol or "").strip()
    if not name:
        return {"success": False, "error": "No symbol given."}
    try:
        cap = max(1, min(int(limit), MAX_REFERENCES))
    except (TypeError, ValueError):
        cap = 40
    target = _resolve_path(path)
    if not target.is_file():
        return {"success": False, "error": f"No such file: {path}"}

    if directory:
        root = _resolve_path(directory)
        if not root.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}
    else:
        root = project_root(target)

    language = language_of(target)
    files, used_ast = _search_files(root, target)

    references: List[Dict[str, Any]] = []
    unreadable = 0
    for candidate in files:
        if used_ast:
            tree = _parse_python(candidate)
            if tree is None:
                unreadable += 1
                continue
            index = _PythonIndex(candidate, root, name)
            index.visit(tree)
            references.extend(index.references)
        else:
            references.extend(_regex_references(candidate, root, name))
        if len(references) > cap * 3:
            break

    references.sort(key=lambda item: (item["file"], item["line"]))
    truncated = len(references) > cap
    shown = references[:cap]
    note = (f"  ({unreadable} file(s) could not be parsed and were skipped)"
            if unreadable else "")
    if truncated:
        note += (f"  (showing the first {cap} of {len(references)}; raise `limit` "
                 f"to see more)")
    return {"success": True,
            "output": _render_references(name, shown, root, used_ast) + note,
            "metadata": {"symbol": name, "language": language, "used_ast": used_ast,
                         "root": str(root), "files_searched": len(files),
                         "unparseable": unreadable, "count": len(references),
                         "shown": len(shown), "references": shown}}


def lsp_hover(path: str, line: int, symbol: str = "") -> Dict[str, Any]:
    """Describe what is at a line, from the source around it.

    For Python that is the enclosing definition, its signature and its
    docstring. For anything else it is the line and the comment block above it.
    It is a reading of the file, not a type: nothing is inferred.
    """
    target = _resolve_path(path)
    if not target.is_file():
        return {"success": False, "error": f"No such file: {path}"}
    try:
        number = int(line)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Line must be a number, got {line!r}."}
    if number < 1:
        return {"success": False, "error": f"Line numbers start at 1, got {number}."}

    text = _read(target)
    lines = text.splitlines()
    if number > len(lines):
        return {"success": False,
                "error": (f"{_relative(target, Path.cwd())} has {len(lines)} lines; "
                          f"line {number} is past the end.")}

    name = str(symbol or "").strip()
    source_line = lines[number - 1].strip()[:200]
    block = [lines[number - 1].rstrip()]
    # Walk back over the comment block directly above: for most languages that
    # is the only documentation the file has.
    for previous in range(number - 2, max(-1, number - 12), -1):
        stripped = lines[previous].strip()
        if stripped.startswith(("#", "//", "*", "/*", '"""')):
            block.append(lines[previous].rstrip())
            continue
        break
    block.reverse()

    out: List[str] = [f"{_relative(target, Path.cwd())}:{number}"]
    out.append(f"  {source_line}" if source_line else "  (blank line)")
    if len(block) > 1:
        out.append("  context above:")
        out.extend(f"    {row}" for row in block[:-1])

    enclosing: Dict[str, Any] = {}
    if language_of(target) == "python":
        tree = _parse_python(target)
        if tree is None:
            out.append("")
            out.append("  This file does not parse, so no enclosing definition could "
                       "be found. The lines above are the text, nothing more.")
        else:
            enclosing = _enclosing(target, tree, number)
            if enclosing:
                out.append("")
                out.append(f"  Enclosing {enclosing['kind']}: {enclosing['signature']}")
                if enclosing["docstring"]:
                    body = enclosing["docstring"]
                    out.append(f"  Docstring: {body.splitlines()[0][:200]}")
                    for extra in body.splitlines()[1:6]:
                        out.append(f"             {extra.rstrip()[:200]}")
            elif name:
                found = _PythonIndex(target, target, name)
                found.visit(tree)
                if found.definitions:
                    hit = found.definitions[0]
                    out.append("")
                    out.append(f"  {name!r} is defined here: {hit['signature'] or hit['source']}")
                    enclosing = hit
            if name and name not in source_line:
                out.append("")
                out.append(f"  Note: {name!r} does not appear on line {number}. "
                           f"This is the context of the line, not of that symbol.")
    elif name:
        candidates = _regex_definitions(target, target, name)
        if candidates:
            out.append("")
            out.append(f"  {name!r} may be defined here: {candidates[0]['line']}: "
                       f"{candidates[0]['source']}")
            out.append("  (a regular-expression match, not a parsed definition)")
    out.append("")
    out.append("  This is the file's own text. No type was inferred and no symbol "
               "table was consulted; for that, install a language server and "
               "configure it.")
    return {"success": True, "output": "\n".join(out),
            "metadata": {"file": str(target), "line": number, "symbol": name,
                         "source": source_line, "enclosing": enclosing,
                         "language": language_of(target)}}


def _enclosing(path: Path, tree: ast.AST, line: int) -> Dict[str, Any]:
    """The innermost def or class whose body contains ``line``."""
    best: Optional[ast.AST] = None
    best_span = 0

    def walk(node: ast.AST) -> None:
        nonlocal best, best_span
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(child, "lineno", 0)
                end = getattr(child, "end_lineno", start) or start
                if start <= line <= end:
                    span = end - start
                    if best is None or span < best_span:
                        best, best_span = child, span
                walk(child)

    walk(tree)
    if best is None:
        return {}
    kind = ("class" if isinstance(best, ast.ClassDef)
            else "async function" if isinstance(best, ast.AsyncFunctionDef)
            else "function")
    return {"kind": kind, "name": getattr(best, "name", ""),
            "signature": _python_signature(best), "docstring": _docstring(best),
            "line": getattr(best, "lineno", 0)}


__all__ = [
    "SERVERS", "ServerSpec", "language_of", "project_root", "lsp_status",
    "lsp_diagnostics", "lsp_definitions", "lsp_references", "lsp_hover",
]
