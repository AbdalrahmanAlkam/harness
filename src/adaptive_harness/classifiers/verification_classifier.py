"""Verification Classifier analyzing tool outputs and recommending recovery actions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from adaptive_harness.tools.base import ToolResult


@dataclass
class VerificationAssessment:
    status: str  # 'SUCCESS', 'SYNTAX_ERROR', 'TEST_FAILURE', 'FILE_ERROR', 'RUNTIME_ERROR'
    needs_retry: bool
    recommended_action: str
    diagnostic_summary: str


class VerificationClassifier:
    """Classifies tool execution results to drive autonomous agent self-healing and recovery."""

    def evaluate(self, tool_name: str, tool_result: ToolResult) -> VerificationAssessment:
        if tool_result.success:
            return VerificationAssessment(
                status="SUCCESS",
                needs_retry=False,
                recommended_action="PROCEED",
                diagnostic_summary=f"Tool `{tool_name}` completed successfully.",
            )

        err = (tool_result.error or "") + " " + (tool_result.output or "")
        err_lower = err.lower()

        # Syntax error
        if "syntaxerror" in err_lower or "indentationerror" in err_lower:
            return VerificationAssessment(
                status="SYNTAX_ERROR",
                needs_retry=True,
                recommended_action="AUTO_RETRY_SYNTAX_FIX",
                diagnostic_summary="Python syntax or indentation error detected. Prompt agent with compiler error to fix syntax.",
            )

        # Test failure
        if "failed" in err_lower and ("test" in tool_name or "pytest" in err_lower or "assert" in err_lower):
            return VerificationAssessment(
                status="TEST_FAILURE",
                needs_retry=True,
                recommended_action="AUTO_RETRY_TEST_FIX",
                diagnostic_summary="Unit tests failed assertion checks. Feed failing assertions back to agent.",
            )

        # File not found
        if "file not found" in err_lower or "filenotfounderror" in err_lower or "no such file" in err_lower:
            return VerificationAssessment(
                status="FILE_ERROR",
                needs_retry=True,
                recommended_action="VERIFY_PATH_OR_SEARCH",
                diagnostic_summary="File or directory path does not exist. Use search_files or list_directory first.",
            )

        # General runtime exception
        return VerificationAssessment(
            status="RUNTIME_ERROR",
            needs_retry=True,
            recommended_action="AUTO_RETRY",
            diagnostic_summary=f"Runtime error in `{tool_name}`: {tool_result.error or 'Non-zero exit code'}",
        )
