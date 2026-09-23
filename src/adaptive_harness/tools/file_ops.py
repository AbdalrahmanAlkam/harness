"""File operation tools: reading, writing, and surgical editing with unified diffs."""

from __future__ import annotations

import ast
import difflib
import os
from pathlib import Path
from typing import Any, Dict, Optional

from adaptive_harness.tools.base import Tool, ToolResult, workspace_path


class ReadFileTool(Tool):
    """Reads text from a local workspace file with optional line slice bounds."""

    name = "read_file"
    description = "Reads content from a file in the workspace with optional line slice bounds."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path to the file."},
            "start_line": {"type": "integer", "description": "1-based starting line number (optional)."},
            "end_line": {"type": "integer", "description": "1-based ending line number (optional)."},
            "symbol": {"type": "string", "description": "Python function, class, or Class.method to extract."},
        },
        "required": ["path"],
    }

    def __init__(self, workspace_root: Optional[Path | str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            file_path = workspace_path(self.workspace_root, path)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        if not file_path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        if not file_path.is_file():
            return ToolResult(success=False, output="", error=f"Path is not a file: {path}")

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)

            if symbol:
                if file_path.suffix != ".py":
                    return ToolResult(success=False, output="", error="Symbol extraction requires a Python file")
                try:
                    tree = ast.parse(content, filename=str(file_path))
                except SyntaxError as exc:
                    return ToolResult(success=False, output="", error=f"SyntaxError: {exc.msg} at line {exc.lineno}")
                node = tree
                for part in symbol.split("."):
                    node = next((child for child in getattr(node, "body", [])
                                 if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                                 and child.name == part), None)
                    if node is None:
                        return ToolResult(success=False, output="", error=f"Symbol not found: {symbol}")
                start_line, end_line = node.lineno, node.end_lineno
                imports = [f"{item.lineno:4d} | {''.join(lines[item.lineno-1:item.end_lineno]).rstrip()}"
                           for item in tree.body if isinstance(item, (ast.Import, ast.ImportFrom))]
            else:
                imports = []

            if start_line is None and end_line is None and len(lines) > 200:
                end_line = 120

            start = max(1, start_line) if start_line is not None else 1
            end = min(len(lines), end_line) if end_line is not None else len(lines)

            selected_lines = lines[start - 1 : end]
            numbered = [f"{i + start:4d} | {line}" for i, line in enumerate(selected_lines)]
            output_str = ("Imports:\n" + "\n".join(imports) + "\n\n" if imports else "") + "".join(numbered)
            if not symbol and start == 1 and end < len(lines) and start_line is None:
                output_str += f"\n… {len(lines)-end} lines omitted. Use symbol or start_line/end_line for a focused read."

            return ToolResult(
                success=True,
                output=output_str,
                metadata={"total_lines": len(lines), "start_line": start, "end_line": end, "symbol": symbol},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to read file: {e}")


class WriteFileTool(Tool):
    """Creates or overwrites a file in the workspace."""

    name = "write_file"
    description = "Creates a new file or overwrites an existing file with the provided text."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Target relative file path."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace_root: Optional[Path | str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def execute(self, path: str, content: str, **kwargs: Any) -> ToolResult:
        try:
            file_path = workspace_path(self.workspace_root, path)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        syntax_error = _python_syntax_error(file_path, content)
        if syntax_error:
            return ToolResult(success=False, output="", error=syntax_error)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to `{path}`.",
                metadata={"bytes_written": len(content.encode("utf-8"))},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to write file: {e}")


class EditFileTool(Tool):
    """Replaces a targeted substring in an existing file and produces a diff."""

    name = "edit_file"
    description = "Surgically replaces an exact text block in an existing file with new text."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to modify."},
            "target_text": {"type": "string", "description": "Exact text substring to replace."},
            "replacement_text": {"type": "string", "description": "Replacement text content."},
        },
        "required": ["path", "target_text", "replacement_text"],
    }

    def __init__(self, workspace_root: Optional[Path | str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def execute(
        self,
        path: str,
        target_text: str,
        replacement_text: str,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            file_path = workspace_path(self.workspace_root, path)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        if not file_path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path}")

        try:
            original = file_path.read_text(encoding="utf-8")
            if target_text not in original:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Target text was not found in `{path}`. Check exact whitespace and lines.",
                )

            # Check uniqueness
            count = original.count(target_text)
            if count > 1:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Target text matched {count} times in `{path}`. Provide a larger unique context block.",
                )

            updated = original.replace(target_text, replacement_text, 1)
            syntax_error = _python_syntax_error(file_path, updated)
            if syntax_error:
                return ToolResult(success=False, output="", error=syntax_error)
            file_path.write_text(updated, encoding="utf-8")

            # Generate unified diff
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )

            return ToolResult(
                success=True,
                output=f"Successfully edited `{path}`.\n\nDiff:\n{diff}",
                metadata={"diff": diff},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to edit file: {e}")


def _python_syntax_error(path: Path, content: str) -> str | None:
    if path.suffix != ".py":
        return None
    try:
        ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} at line {exc.lineno}, column {exc.offset}"
    return None
