"""A worked example plugin: find TODO/FIXME markers in the workspace.

It exists to demonstrate the plugin contract, not because TODO scanning is
something the harness needs. It is deliberately small, declares exactly the
permissions it uses, and every one of those is a `read` -- there is no
subprocess, no network, and no access outside the workspace.
"""

from __future__ import annotations

import re
from pathlib import Path

#: A marker line, with the file and the line number, kept to what a person
#: would want to see rather than the whole file's contents.
_MARKER = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]*(.*)")

#: Directories that are never worth scanning, matching the harness's own
#: workspace walk.
_SKIP = {".git", "__pycache__", "node_modules", ".venv", ".tox",
         ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build"}


def scan_markers(path=".", limit=50):
    """Report TODO-style markers under `path`, newest file first is not needed.

    Returns the harness's own tool result shape, so a plugin is indistinguishable
    from a built-in at the call site.
    """
    root = Path(path)
    if not root.is_dir():
        return {"success": False, "error": f"Not a directory: {path}"}
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 50

    found: list[str] = []
    for file_path in sorted(root.rglob("*")):
        if len(found) >= limit:
            break
        if any(part in _SKIP for part in file_path.parts):
            continue
        if not file_path.is_file() or file_path.suffix not in {".py", ".js", ".ts", ".md", ".toml"}:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            match = _MARKER.search(line)
            if match:
                relative = file_path.relative_to(root)
                note = match.group(2).strip()[:80]
                found.append(f"{relative}:{number}: {match.group(1)}{' ' + note if note else ''}")
            if len(found) >= limit:
                break

    if not found:
        return {"success": True, "output": "No TODO-style markers found."}
    return {"success": True,
            "output": f"{len(found)} marker(s):\n" + "\n".join(found)}
