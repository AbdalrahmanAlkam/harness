"""Bounded single-role delegation exposed to a swarm-enabled agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from adaptive_harness.tools.base import Tool, ToolResult, workspace_path


class DelegateSubagentTool(Tool):
    name = "delegate_subagent"
    description = "Run an architect, coder, or reviewer subagent on a bounded workspace task."
    parameters = {"type": "object", "properties": {
        "role": {"type": "string", "enum": ["architect", "coder", "reviewer"]},
        "task": {"type": "string"},
        "target_dir": {"type": "string", "description": "Optional directory inside the workspace"},
    }, "required": ["role", "task"]}

    def __init__(self, workspace_root: str | Path, *, llm_client_factory: Callable[[], Any] | None = None):
        self.workspace_root = Path(workspace_root).resolve()
        self.llm_client_factory = llm_client_factory

    def execute(self, role: str, task: str, target_dir: str | None = None, **kwargs: Any) -> ToolResult:
        from adaptive_harness.agent.swarm import (DeveloperAgentWorker, SwarmAssignment,
                                                   SwarmPhase, SwarmRole)

        roles = {"architect": (SwarmRole.ARCHITECT, SwarmPhase.PLAN),
                 "coder": (SwarmRole.CODER, SwarmPhase.IMPLEMENT),
                 "reviewer": (SwarmRole.QA, SwarmPhase.VERIFY)}
        if role not in roles or not isinstance(task, str) or not task.strip():
            return ToolResult(success=False, output="", error="Choose architect, coder, or reviewer and provide a task")
        try:
            root = workspace_path(self.workspace_root, target_dir or ".")
            root.mkdir(parents=True, exist_ok=True)
            selected_role, phase = roles[role]
            assignment = SwarmAssignment(selected_role, phase, task, root)
            result = DeveloperAgentWorker(llm_client_factory=self.llm_client_factory)(assignment)
            return ToolResult(success=result.success, output=json.dumps(result.to_dict(), ensure_ascii=False),
                              error=result.error, metadata={"role": role, "verified": result.verified})
        except (OSError, ValueError) as exc:
            return ToolResult(success=False, output="", error=f"Delegation failed: {exc}")
