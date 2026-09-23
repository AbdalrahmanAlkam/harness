"""Automated test execution tool running pytest with structured outcome parsing."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Optional

from adaptive_harness.tools.base import Tool, ToolResult


class RunPytestTool(Tool):
    """Executes pytest suites and extracts structured test statistics."""

    name = "run_pytest"
    description = "Runs pytest on specified test paths or the entire test suite and reports results."
    parameters = {
        "type": "object",
        "properties": {
            "test_path": {
                "type": "string",
                "description": "Optional specific test file or directory (default 'tests/').",
                "default": "tests/",
            },
            "extra_args": {
                "type": "string",
                "description": "Additional pytest CLI flags (e.g. '-k test_name', '-x').",
            },
        },
    }

    def __init__(self, workspace_root: Optional[Path | str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def execute(self, test_path: str = "tests/", extra_args: Optional[str] = None, **kwargs: Any) -> ToolResult:
        import sys

        cmd = f"{sys.executable} -m pytest {test_path} -v"
        if extra_args:
            cmd = f"{cmd} {extra_args}"

        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=60,
            )

            output = res.stdout.strip()
            stderr = res.stderr.strip()
            if stderr:
                output = f"{output}\n[STDERR]\n{stderr}"

            # Parse test summary line (e.g. "38 passed in 12.35s" or "2 failed, 36 passed")
            passed = 0
            failed = 0
            match_pass = re.search(r"(\d+)\s+passed", output)
            if match_pass:
                passed = int(match_pass.group(1))
            match_fail = re.search(r"(\d+)\s+failed", output)
            if match_fail:
                failed = int(match_fail.group(1))

            return ToolResult(
                success=(res.returncode == 0),
                output=output or "(pytest produced no output)",
                error=stderr if res.returncode != 0 else None,
                metadata={
                    "exit_code": res.returncode,
                    "passed": passed,
                    "failed": failed,
                    "total": passed + failed,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Pytest run timed out after 60 seconds")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to run pytest: {e}")
