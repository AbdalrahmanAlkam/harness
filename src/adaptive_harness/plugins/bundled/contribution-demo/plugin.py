"""A plugin that exercises every contribution type at once.

This exists for the acceptance criterion "a test plugin uses every one of the
seven types". It is deliberately small and does nothing useful in production --
its job is to prove the contract is complete and to give the type-checking tests
something real to load.

Types used: tools, skills, settings, prompts, commands (with a model override),
mcp, subagents, context, and hooks.
"""

from __future__ import annotations

from adaptive_harness.plugins.types import (
    ALLOW,
    ContextFragment,
    HookVerdict,
    PRIORITY_PROJECT,
    Trigger,
)
from adaptive_harness.skills.registry import BaseSkill

#: Tools the demo blocks. Nothing in the harness's own vocabulary, so this
#: cannot interfere with a real run.
BLOCKED_TOOLS = {"run_dangerous_demo"}


def note_size(path):
    """Report the size of a file, to prove a plain tool still works."""
    from pathlib import Path

    target = Path(path)
    if not target.is_file():
        return {"success": False, "error": f"No such file: {path}"}
    return {"success": True, "output": f"{target.stat().st_size} bytes"}


def build_skill(source="plugin"):
    return BaseSkill(
        name="demo_auditor", title="Demo Auditor", category="Demo",
        trigger="demo audit, check the demo",
        instructions="Run the demo auditor's checks and report what they found.",
        tools=("note_size",), invariants=("evidence_checked",), source=source)


def project_conventions(source="", provenance=""):
    """A context fragment built in code, to exercise the factory path."""
    return ContextFragment(
        source="demo:conventions",
        content="This project prefers explicit errors over silent fallbacks.",
        tokens=16,
        priority=PRIORITY_PROJECT,
        trigger=Trigger.parse("keyword:fallback"),
        provenance=provenance or source,
    )


def guard_pre_tool(tool_name, arguments):
    """A PRE_TOOL hook that can deny.

    Demonstrates the two-phase protocol: it returns a real verdict, and a denial
    carries a reason so the model is told why rather than seeing a silent skip.
    """
    if tool_name in BLOCKED_TOOLS:
        return HookVerdict("deny", reason="run_dangerous_demo is disabled by the demo plugin")
    return ALLOW


def redact_post_tool(tool_name, arguments, output, success):
    """A POST_TOOL hook that attaches metadata to a successful call.

    It deliberately cannot touch a failure -- the host ignores a redaction of a
    failed call, so error detail is never lost.
    """
    if not success:
        return None
    return {"metadata": {"demo_plugin_saw": tool_name}}


def observe_final(summary, requirement_set, evidence):
    """An ON_FINAL hook: the classifier's mount point, used by the final gate."""
    return {"summary_chars": len(summary or ""),
            "requirements": len(requirement_set or []),
            "evidence": len(evidence or [])}
