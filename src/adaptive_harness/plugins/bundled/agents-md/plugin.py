"""agents-md — the project-memory loader, as a pure reader.

A coding agent pointed at a repository it did not write has no idea what that
repository's conventions are: which test command counts as passing, where the
generated files go, what is deliberately slow. Project instructions are where a
project says so, in ``AGENTS.md`` and in ``.harness/instructions/*.md``. This
plugin reads them and hands them to the model.

**It never writes.** That is the whole design, and it is not a missing feature.
A memory loader that scaffolds a file on first run has quietly modified a
repository the user cloned, and the user is the one who has to notice and undo
it. So ``memory_check`` prints the scaffold a user would need to create *and
says plainly that nothing was created* — a report is information, a file is a
side effect, and only one of those is what was asked for. The ``tools``
permission is the whole manifest for the same reason: the plugin reads.

**Precedence, and why it is stated.** Instructions are returned in a fixed
order, and the order is part of the contract rather than an accident of
directory listing:

1. ``AGENTS.md`` at the workspace root, then
2. ``.harness/instructions/*.md`` in sorted filename order.

Later entries are the *more specific* layer and win where two instructions
disagree, so the combined text is emitted in that order and says so in its
header. Numbering the files (``10-testing.md``, ``20-style.md``) is therefore
meaningful: the numbering is the precedence. A loader that returned them in
glob order would make a convention silently depend on the filesystem.

**The include chain.** An instruction file may pull in another with a line of
the form ``@relative/path.md``. The include is resolved against the *including*
file's directory, must stay inside the workspace, and is followed recursively
with a depth cap and a cycle check — an include loop must be a reported error,
not a hang. This is the mechanism ``docs/master-plan.md`` §1.5 asks for, and it
is read-only: an include never creates what it points at.

Everything is stdlib and everything is bounded. Files are read with a size cap
and the combined text is truncated with the truncation stated, because a
context contribution nobody bounded is a context contribution that can fill the
window on its own.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: The root instruction file, relative to the workspace.
ROOT_FILE = "AGENTS.md"

#: Where per-topic instruction files live, relative to the workspace.
INSTRUCTIONS_DIR = (".harness", "instructions")

#: An include directive: ``@path/to/file.md``, alone on its line. The path is
#: relative to the file that wrote it, and must stay inside the workspace.
INCLUDE = re.compile(r"^\s*@(?P<path>[^\s@]+)\s*$")

#: Depth cap for the include chain. Two is a structure; ten is a cycle that
#: happens to alternate between two files.
MAX_INCLUDE_DEPTH = 5

#: A single instruction file larger than this is truncated. 256 KB of markdown
#: is not an instruction, it is a document someone meant to link to.
MAX_FILE_BYTES = 256 * 1024

#: Ceiling on the combined text handed to the model.
MAX_TOTAL_BYTES = 128 * 1024

#: The scaffold ``memory_check`` prints when a project has no instructions. It
#: is a string in this file, not a file this file writes: the difference is the
#: entire point of the plugin.
SCAFFOLD = """\
# AGENTS.md — how to work in this repository

<!-- Written for an AI coding agent, and for the humans who maintain it.
     Delete any section that does not apply; this file is only useful if it
     is true. -->

## What this project is

<!-- Two or three sentences. What it does, and what it is for. -->

## Commands

- Install:      <the exact command>
- Run tests:    <the exact command, and what a pass looks like>
- Lint/format:  <the exact command>
- Build:        <the exact command>

## Conventions

<!-- The things a change to this repository can get wrong: where new code goes,
     how errors are handled, what must not be imported, what is generated and
     must not be hand-edited. -->

## Things that will surprise you

<!-- Slow tests. Fixtures that need a service running. A directory that looks
     like source but is a copy. Anything that cost an afternoon once. -->

## Scope

<!-- Where the agent may edit, and where it must ask first. -->
"""

#: The same scaffold for a per-topic file, for a project that prefers several
#: small files over one large one.
TOPIC_SCAFFOLD = """\
# <topic>.md — one topic, one file

<!-- Drop this in .harness/instructions/. Filenames are sorted, and the sort
     order is the precedence order, so number them:

       10-testing.md   20-architecture.md   30-style.md

     Later files are the more specific layer and win where two instructions
     disagree. -->

## <Topic>

<!-- The rule, in the imperative. "Run `pytest tests/ -q` before calling
     anything done", not "tests should generally be run". -->
"""


# --- reading ---------------------------------------------------------------


def _workspace(raw: Any = "") -> Path:
    """The workspace root, from the argument, the environment, or the cwd.

    A tool call arrives with its path arguments already confined to the
    workspace, so ``path`` is the normal way in. The environment override exists
    for a run whose workspace differs from its working directory; the cwd is
    the fallback, and it is the least trustworthy of the three.
    """
    text = str(raw or "").strip()
    if text:
        candidate = Path(text).expanduser()
        return candidate if candidate.is_absolute() else (Path.cwd() / candidate)
    override = (os.environ.get("ADAPTIVE_HARNESS_WORKSPACE", "") or "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_absolute() else (Path.cwd() / candidate)
    return Path.cwd()


def _contained(candidate: Path, root: Path) -> Optional[Path]:
    """Resolve ``candidate`` inside ``root``, or ``None`` if it escapes.

    The include chain follows a path written by a text file, which is exactly
    the shape of a traversal (``@../../../etc/passwd``). Resolution happens
    before the check, so a symlink is caught as well as a ``..``.
    """
    try:
        resolved = candidate.resolve()
        if resolved == root or resolved.is_relative_to(root):
            return resolved
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def _read(path: Path) -> Tuple[str, str]:
    """Read a file, returning ``(text, note)``.

    ``note`` is non-empty when the text was not the whole file: too large, or
    unreadable. The note travels with the content all the way to the model,
    because text that was silently cut reads as complete.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return "", f"could not be read ({exc})"
    if size > MAX_FILE_BYTES:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_BYTES], \
                (f"truncated: the file is {size} bytes and only the first "
                 f"{MAX_FILE_BYTES} were read")
        except OSError as exc:
            return "", f"could not be read ({exc})"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), ""
    except OSError as exc:
        return "", f"could not be read ({exc})"


def _resolve_includes(text: str, origin: Path, root: Path, depth: int,
                      seen: Tuple[str, ...]) -> Tuple[str, List[str]]:
    """Expand ``@path`` directives, returning the text and any problems.

    A cycle, an escape from the workspace, a missing target and a depth overrun
    are all *reported in the text* rather than raised. A memory file with one
    bad include is still worth loading, and the reader is told which line to fix.
    """
    problems: List[str] = []
    out: List[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = INCLUDE.match(line)
        if not match:
            out.append(line)
            continue
        relative = match.group("path")
        if depth >= MAX_INCLUDE_DEPTH:
            problems.append(f"{origin.name}:{number} include {relative!r} skipped: "
                            f"the include chain is deeper than {MAX_INCLUDE_DEPTH}")
            continue
        target = _contained(origin.parent / relative, root)
        if target is None:
            problems.append(f"{origin.name}:{number} include {relative!r} refused: "
                            f"it resolves outside the workspace")
            continue
        if str(target) in seen:
            problems.append(f"{origin.name}:{number} include {relative!r} skipped: "
                            f"it is already in this chain (cycle)")
            continue
        if not target.is_file():
            problems.append(f"{origin.name}:{number} include {relative!r} skipped: "
                            f"no such file in the workspace")
            continue
        nested, note = _read(target)
        if note:
            problems.append(f"{origin.name}:{number} include {relative!r}: {note}")
        expanded, inner = _resolve_includes(nested, target, root, depth + 1,
                                            seen + (str(target),))
        problems.extend(inner)
        out.append(f"\n<!-- included from {relative} -->\n")
        out.append(expanded)
        out.append(f"\n<!-- end of {relative} -->\n")
    return "\n".join(out), problems


def _collect(root: Path) -> List[Dict[str, Any]]:
    """Every instruction file in the workspace, in precedence order.

    ``AGENTS.md`` first, then ``.harness/instructions/*.md`` sorted by
    filename — which is why the sort is by name and not by mtime. Precedence
    has to be something a reader can see in the repository.

    The rank is the file's *position in this list*, so it always runs 1..N. A
    rank counting the layers a workspace happens to have would number a lone
    topic file "2", which reads as a file that lost to something absent.
    """
    entries: List[Dict[str, Any]] = []
    conventions = root / ROOT_FILE
    if conventions.is_file():
        entries.append({"path": conventions, "layer": "root",
                        "description": "AGENTS.md (root conventions)"})
    directory = root.joinpath(*INSTRUCTIONS_DIR)
    if directory.is_dir():
        for path in sorted(p for p in directory.glob("*.md") if p.is_file()):
            entries.append({"path": path, "layer": "instructions",
                            "description": path.relative_to(root).as_posix()})
    for rank, entry in enumerate(entries, start=1):
        entry["precedence"] = rank
    return entries


def _render(entries: List[Dict[str, Any]], problems: List[str], total: int) -> str:
    """The combined text, with the precedence rule in its header.

    The header is not decoration. Two instruction files that disagree are
    resolved by order, and a reader who cannot see the order cannot tell which
    one won.
    """
    out = [
        f"Project instructions, in precedence order ({len(entries)} file(s), "
        f"{total} bytes). Later entries are the more specific layer and win where "
        f"two instructions disagree.",
        "",
    ]
    for entry in entries:
        body = entry["text"].strip()
        out.append(f"--- [{entry['precedence']}] {entry['description']} "
                   f"({len(body)} chars) ---")
        if entry["note"]:
            out.append(f"[{entry['note']}]")
        out.append(body)
        out.append("")
    if problems:
        out.append("Include problems (nothing was written to fix them):")
        out.extend(f"  {problem}" for problem in problems)
    return "\n".join(out).rstrip()


# --- the tools -------------------------------------------------------------


def memory_load(path: str = "", max_chars: int = 60000) -> Dict[str, Any]:
    """Read the project's instructions and return them as one block of text.

    Reads ``AGENTS.md`` and then ``.harness/instructions/*.md`` in sorted
    filename order, expands any ``@relative/path.md`` includes, and returns the
    combination with the precedence rule stated in the header. Writes nothing,
    creates nothing, and says which files contributed. Returns an honest
    "there are none" when there are none, rather than an empty string that
    reads like instructions that happened to say nothing.
    """
    root = _workspace(path)
    if not root.is_dir():
        return {"success": False, "error": f"Not a directory: {root}"}
    try:
        ceiling = max(500, min(int(max_chars), 200000))
    except (TypeError, ValueError):
        ceiling = 60000

    entries = _collect(root)
    if not entries:
        return {"success": True,
                "output": (f"No project instructions in {root}. There is no "
                           f"{ROOT_FILE} and no {INSTRUCTIONS_DIR[0]}/{INSTRUCTIONS_DIR[1]}/"
                           f"*.md. Nothing was created: this tool only reads. Call "
                           f"memory_check to see the scaffold a user would need to "
                           f"write, and to have it printed."),
                "metadata": {"workspace": str(root), "files": [], "count": 0,
                             "precedence": [], "problems": [], "bytes": 0,
                             "wrote": False}}

    problems: List[str] = []
    loaded: List[Dict[str, Any]] = []
    for entry in entries:
        text, note = _read(entry["path"])
        expanded, inner = _resolve_includes(text, entry["path"], root, 0,
                                            (str(entry["path"].resolve()),))
        problems.extend(inner)
        if note:
            problems.append(f"{entry['path'].name}: {note}")
        loaded.append({**entry, "text": expanded, "note": note,
                       "bytes": len(expanded.encode("utf-8", "replace"))})

    total = sum(entry["bytes"] for entry in loaded)
    over_budget = total > MAX_TOTAL_BYTES
    combined = _render(loaded, problems, total)
    if over_budget:
        combined += (f"\n\n[The instructions total {total} bytes, over the "
                     f"{MAX_TOTAL_BYTES}-byte budget for one project context. "
                     f"What is shown may be cut; split the project up or trim the "
                     f"files rather than relying on this.]")
    truncated = len(combined) > ceiling
    if truncated:
        combined = combined[:ceiling] + "\n\n[cut at the max_chars you asked for]"

    return {"success": True, "output": combined,
            "metadata": {"workspace": str(root),
                         "files": [str(e["path"]) for e in loaded],
                         "relative": [e["path"].relative_to(root).as_posix()
                                      for e in loaded],
                         "count": len(loaded),
                         "precedence": [e["precedence"] for e in loaded],
                         "problems": problems,
                         "bytes": sum(e["bytes"] for e in loaded),
                         "truncated": truncated, "wrote": False}}


def memory_check(path: str = "") -> Dict[str, Any]:
    """Report whether this project has instructions, and say what is missing.

    When there are none, it prints the scaffold a user would need to create and
    states plainly that the harness did not create it. That sentence is the
    point: a memory tool that quietly writes a file into a repository the user
    only pointed the agent at is a tool that changed something nobody asked it
    to change, and the user is the one who has to notice.
    """
    root = _workspace(path)
    if not root.is_dir():
        return {"success": False, "error": f"Not a directory: {root}"}

    entries = _collect(root)
    convention = root / ROOT_FILE
    topic_dir = root.joinpath(*INSTRUCTIONS_DIR)

    if not entries:
        return {"success": True,
                "output": "\n".join([
                    f"No project instructions found in {root}.",
                    "",
                    f"Checked: {ROOT_FILE} (absent) and "
                    f"{INSTRUCTIONS_DIR[0]}/{INSTRUCTIONS_DIR[1]}/*.md "
                    f"({'absent' if not topic_dir.is_dir() else 'present but empty'}).",
                    "",
                    "NOTHING WAS CREATED. The harness did not write a file here, and "
                    "this tool cannot: it only reads. If you want these instructions "
                    "to exist, create them yourself with the scaffold below.",
                    "",
                    "--- scaffold for AGENTS.md (not written) ---",
                    SCAFFOLD,
                    "--- scaffold for .harness/instructions/10-<topic>.md (not written) ---",
                    TOPIC_SCAFFOLD,
                ]),
                "metadata": {"workspace": str(root), "has_agents_md": False,
                             "has_instructions_dir": topic_dir.is_dir(),
                             "count": 0, "files": [], "wrote": False,
                             "scaffold_shown": True}}

    rows: List[str] = []
    for entry in entries:
        text, note = _read(entry["path"])
        rows.append(f"  [{entry['precedence']}] {entry['description']} -- "
                    f"{len(text)} chars, {text.count(chr(10)) + 1} lines"
                    + (f"  ({note})" if note else ""))
    notes: List[str] = []
    topic_files = [e for e in entries if e["layer"] == "instructions"]
    if not convention.is_file():
        notes.append(f"There is no {ROOT_FILE} at the workspace root, so the "
                     f"project-wide conventions are unstated. Topic files under "
                     f"{INSTRUCTIONS_DIR[0]}/{INSTRUCTIONS_DIR[1]}/ are being used "
                     f"without them.")
    if not topic_files:
        where = (f"the {INSTRUCTIONS_DIR[0]}/{INSTRUCTIONS_DIR[1]}/ directory is "
                 f"empty" if topic_dir.is_dir()
                 else f"there is no {INSTRUCTIONS_DIR[0]}/{INSTRUCTIONS_DIR[1]}/ "
                      f"directory")
        notes.append(f"There are no topic files: {where}, so everything lives in "
                     f"{ROOT_FILE}. That is fine; splitting by topic with numbered "
                     f"filenames makes precedence visible.")
    if len(entries) == 1 and convention.is_file() and not _read(convention)[0].strip():
        notes.append(f"{ROOT_FILE} is empty, so it contributes nothing.")

    return {"success": True,
            "output": "\n".join([
                f"Project instructions in {root}, in precedence order "
                f"(later wins):",
                *rows,
                "",
                f"{len(entries)} file(s) contribute. memory_load returns them "
                f"combined, with any @include directives expanded.",
                *([f"Note: {note}" for note in notes] if notes else []),
                "Nothing was written: this tool only reads.",
            ]),
            "metadata": {"workspace": str(root), "has_agents_md": convention.is_file(),
                         "has_instructions_dir": topic_dir.is_dir(),
                         "count": len(entries),
                         "files": [str(e["path"]) for e in entries],
                         "wrote": False, "scaffold_shown": False}}


__all__ = ["memory_load", "memory_check", "SCAFFOLD", "ROOT_FILE",
           "INSTRUCTIONS_DIR"]
