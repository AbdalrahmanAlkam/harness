"""Workspace inspection tools: directory listing and text grep."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from adaptive_harness.tools.base import Tool, ToolResult, workspace_path


class ListDirectoryTool(Tool):
    """Lists files and folders within a workspace directory."""

    name = "list_directory"
    description = "Lists files and subdirectories in a directory path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path (default '.').", "default": "."},
        },
    }

    def __init__(self, workspace_root: Optional[Path | str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def execute(self, path: str = ".", **kwargs: Any) -> ToolResult:
        try:
            dir_path = workspace_path(self.workspace_root, path)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        if not dir_path.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {path}")
        if not dir_path.is_dir():
            return ToolResult(success=False, output="", error=f"Path is not a directory: {path}")

        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            lines = []
            for e in entries:
                if e.name.startswith(".") and e.name not in [".venv"]:
                    continue  # skip hidden files except .venv
                prefix = "[DIR] " if e.is_dir() else "      "
                size = f"({e.stat().st_size:,} bytes)" if e.is_file() else ""
                lines.append(f"{prefix} {e.name} {size}")

            return ToolResult(
                success=True,
                output="\n".join(lines) if lines else "(empty directory)",
                metadata={"count": len(lines)},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to list directory: {e}")


class SearchFilesTool(Tool):
    """Searches workspace files for matching regex or literal text patterns."""

    name = "search_files"
    description = "Searches for text pattern or regular expression across files in the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text pattern or regex to search for."},
            "file_extension": {"type": "string", "description": "Optional extension filter, e.g. 'py', 'md'."},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace_root: Optional[Path | str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def execute(self, pattern: str, file_extension: Optional[str] = None, **kwargs: Any) -> ToolResult:
        matches = []
        regex = re.compile(pattern, re.IGNORECASE)

        for root, dirs, files in os.walk(self.workspace_root):
            # Prune hidden or heavy directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "output")]
            for f in files:
                if file_extension and not f.endswith(f".{file_extension.lstrip('.')}"):
                    continue
                p = Path(root) / f
                try:
                    workspace_path(self.workspace_root, str(p.relative_to(self.workspace_root)))
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(text.splitlines(), start=1):
                        if regex.search(line):
                            rel_path = p.relative_to(self.workspace_root)
                            matches.append(f"{rel_path}:{i}: {line.strip()[:120]}")
                            if len(matches) >= 50:
                                break
                except Exception:
                    continue
                if len(matches) >= 50:
                    break

        out_text = "\n".join(matches) if matches else f"No matches found for pattern '{pattern}'."
        return ToolResult(
            success=True,
            output=out_text,
            metadata={"match_count": len(matches)},
        )
