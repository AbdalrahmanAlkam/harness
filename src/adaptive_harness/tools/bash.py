"""Safe Bash execution tool with timeouts, safety guards, and output capture."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List

from adaptive_harness.tools.base import Tool, ToolResult


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

    def __init__(self, workspace_root: Optional[Path | str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def execute(self, command: str, timeout_seconds: int = 30, **kwargs: Any) -> ToolResult:
        cmd_strip = command.strip()

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
            res = subprocess.run(
                cmd_strip,
                shell=True,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
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
