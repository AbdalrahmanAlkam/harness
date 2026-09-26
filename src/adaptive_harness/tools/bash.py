"""Safe Bash execution tool with timeouts, safety guards, and output capture."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
import re
import shlex

from adaptive_harness.tools.base import Tool, ToolResult, workspace_path
from adaptive_harness.tools.process import run_process


FORBIDDEN_PATTERNS: List[str] = [
    "rm -rf /",
    "rm -rf /*",
    ":(){ :|:& };:",
    "> /dev/sda",
    "mkfs",
]


class RunBashTool(Tool):
    """Executes non-interactive shell commands in the project workspace."""

    name = "run_bash"
    description = "Executes shell commands (e.g. pytest, git status, pip, python) and captures stdout/stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command line to execute.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Timeout in seconds before terminating process (default 30).",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace_root: Optional[Path | str] = None, *, read_only: bool = False):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.read_only = read_only

    def execute(self, command: str, timeout_seconds: int = 30, **kwargs: Any) -> ToolResult:
        cmd_strip = command.strip()
        if not cmd_strip:
            return ToolResult(success=False, output="", error="Command cannot be empty")
        try:
            words = shlex.split(cmd_strip)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=f"Invalid shell quoting: {exc}")
        if self.read_only:
            # Reviewers can check JavaScript syntax without executing project
            # scripts or giving a shell command permission to edit the workspace.
            if len(words) != 3 or words[:2] != ["node", "--check"]:
                return ToolResult(success=False, output="",
                                  error="Reviewer shell access only permits node --check <workspace JavaScript file>")
            try:
                checked_path = workspace_path(self.workspace_root, words[2])
            except ValueError as exc:
                return ToolResult(success=False, output="", error=str(exc))
            if checked_path.suffix not in {".js", ".mjs", ".cjs"} or not checked_path.is_file():
                return ToolResult(success=False, output="",
                                  error="Reviewer syntax check requires an existing JavaScript file")
            command_to_run: str | list[str] = ["node", "--check", str(checked_path)]
        else:
            command_to_run = cmd_strip
        if ("rm" in words and any(word in {"/", "/*", "--no-preserve-root"} for word in words)
                and any(word.startswith("-") and ("r" in word or word == "--recursive") for word in words)):
            return ToolResult(success=False, output="", error="Command blocked by safety filter: recursive root deletion")
        if re.search(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&", cmd_strip):
            return ToolResult(success=False, output="", error="Command blocked by safety filter: fork bomb")

        # Guard against dangerous patterns
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in cmd_strip:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command blocked by safety filter: contains dangerous pattern '{pattern}'",
                )

        import sys

        env = os.environ.copy()
        venv_bin = Path(sys.prefix) / "bin"
        if venv_bin.exists():
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

        try:
            res = run_process(
                command_to_run,
                shell=not self.read_only,
                cwd=str(self.workspace_root),
                timeout=timeout_seconds,
                env=env,
            )

            stdout = res.stdout.strip()
            stderr = res.stderr.strip()
            combined = stdout
            if stderr:
                combined = f"{stdout}\n[STDERR]\n{stderr}" if stdout else stderr

            return ToolResult(
                success=(res.returncode == 0),
                output=combined or "(command produced no output)",
                error=stderr if res.returncode != 0 else None,
                metadata={"exit_code": res.returncode, "command": cmd_strip},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout_seconds} seconds",
                metadata={"timeout": True},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Subprocess execution error: {type(e).__name__}: {str(e)}",
            )
