"""Bounded, artifact-driven collaboration for a developer task.

Only one worker may edit the worktree. Independent planning and final review
jobs run concurrently; each receives compact JSON artifacts from earlier jobs
instead of another agent's conversation history.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol


class SwarmRole(str, Enum):
    ARCHITECT = "architect"
    CODER = "coder"
    QA = "qa"
    SECURITY = "security"


class SwarmPhase(str, Enum):
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"


_ROLE_INSTRUCTIONS: dict[tuple[SwarmRole, SwarmPhase], str] = {
    (SwarmRole.ARCHITECT, SwarmPhase.PLAN):
        "Inspect relevant interfaces and produce a concise implementation plan with risks. Do not edit files.",
    (SwarmRole.QA, SwarmPhase.PLAN):
        "Inspect relevant tests and propose concrete regression cases for the task. Do not edit files.",
    (SwarmRole.CODER, SwarmPhase.IMPLEMENT):
        "Implement the task in the assigned workspace. Create the requested directories and files with write_file or edit_file. Run relevant checks.",
    (SwarmRole.QA, SwarmPhase.VERIFY):
        "Independently inspect the implementation and run relevant tests. Report exact test evidence; do not edit files.",
    (SwarmRole.SECURITY, SwarmPhase.VERIFY):
        "Review the changed files for injection, path traversal, unsafe shell use, and credential leaks. Do not edit files.",
}


@dataclass(frozen=True)
class SwarmAssignment:
    role: SwarmRole
    phase: SwarmPhase
    task: str
    workspace_root: Path
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.role.value}:{self.phase.value}"

    @property
    def may_edit(self) -> bool:
        return self.role is SwarmRole.CODER and self.phase is SwarmPhase.IMPLEMENT

    @property
    def instruction(self) -> str:
        return _ROLE_INSTRUCTIONS[(self.role, self.phase)]


@dataclass(frozen=True)
class SwarmResult:
    role: SwarmRole
    phase: SwarmPhase
    success: bool
    summary: str
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    verified: bool = False
    error: str | None = None

    @property
    def key(self) -> str:
        return f"{self.role.value}:{self.phase.value}"

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "phase": self.phase.value,
                "success": self.success, "summary": self.summary,
                "artifacts": dict(self.artifacts), "verified": self.verified,
                "error": self.error}


@dataclass(frozen=True)
class SwarmReport:
    results: tuple[SwarmResult, ...]
    success: bool
    status: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "status": dict(self.status),
                "results": [result.to_dict() for result in self.results]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class SwarmWorker(Protocol):
    def __call__(self, assignment: SwarmAssignment) -> SwarmResult: ...


StatusCallback = Callable[[Mapping[str, str]], None]


class SwarmCoordinator:
    """Run read-only roles in parallel and serialize the sole writer.

    The caller supplies a workspace. Only the coder can mutate files, so
    direct workspaces remain safe from concurrent agent writes.
    """

    def __init__(self, worker: SwarmWorker, *, max_workers: int = 2,
                 on_status: StatusCallback | None = None) -> None:
        if max_workers < 1 or max_workers > 4:
            raise ValueError("max_workers must be between 1 and 4")
        self.worker = worker
        self.max_workers = max_workers
        self.on_status = on_status

    def run(self, task: str, workspace_root: str | Path, *,
            isolated: bool = False) -> SwarmReport:
        root = Path(workspace_root).resolve()
        if not root.is_dir():
            raise ValueError(f"Workspace does not exist: {root}")
        if not task.strip():
            raise ValueError("Swarm task cannot be empty")

        statuses = {"architect:plan": "queued", "coder:implement": "queued",
                    "qa:verify": "queued"}
        results: list[SwarmResult] = []
        self._notify(statuses)

        def wave(assignments: tuple[SwarmAssignment, ...]) -> list[SwarmResult]:
            for assignment in assignments:
                statuses[assignment.key] = "running"
            self._notify(statuses)
            collected: dict[str, SwarmResult] = {}
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(assignments))) as pool:
                futures = {pool.submit(self.worker, assignment): assignment for assignment in assignments}
                for future in as_completed(futures):
                    assignment = futures[future]
                    try:
                        result = future.result()
                        if result.role is not assignment.role or result.phase is not assignment.phase:
                            raise ValueError("Worker returned a result for a different assignment")
                        # Reject opaque, unserializable payloads at the boundary.
                        json.dumps(result.to_dict(), ensure_ascii=False)
                    except Exception as exc:
                        result = SwarmResult(assignment.role, assignment.phase, False,
                                             "Subagent failed", error=f"{type(exc).__name__}: {exc}")
                    # Retry a transient worker failure once. The writer is
                    # still serialized because each wave finishes before the next.
                    if not result.success and (result.error or assignment.may_edit):
                        try:
                            retry = self.worker(assignment)
                            if retry.role is not assignment.role or retry.phase is not assignment.phase:
                                raise ValueError("Worker returned a result for a different assignment")
                            json.dumps(retry.to_dict(), ensure_ascii=False)
                            result = retry
                        except Exception as exc:
                            result = SwarmResult(assignment.role, assignment.phase, False,
                                                 "Subagent failed after retry",
                                                 error=f"{type(exc).__name__}: {exc}")
                    statuses[assignment.key] = "done" if result.success else "failed"
                    collected[assignment.key] = result
                    self._notify(statuses)
            return [collected[assignment.key] for assignment in assignments]

        planning = wave((SwarmAssignment(SwarmRole.ARCHITECT, SwarmPhase.PLAN, task, root),))
        results.extend(planning)
        implementation = wave((SwarmAssignment(SwarmRole.CODER, SwarmPhase.IMPLEMENT,
            task, root, {item.key: item.to_dict() for item in planning}),))
        results.extend(implementation)
        if not implementation[0].success:
            statuses["qa:verify"] = "skipped"
            self._notify(statuses)
            return SwarmReport(tuple(results), False, dict(statuses))

        review_context = {item.key: item.to_dict() for item in results}
        reviews = wave((SwarmAssignment(SwarmRole.QA, SwarmPhase.VERIFY, task, root, review_context),))
        results.extend(reviews)
        success = implementation[0].success and all(item.success for item in reviews) and reviews[0].verified
        return SwarmReport(tuple(results), success, dict(statuses))

    def _notify(self, status: Mapping[str, str]) -> None:
        if self.on_status is not None:
            self.on_status(dict(status))


class DeveloperAgentWorker:
    """Opt-in role runner using separate agent state for each assignment.

    Only the coder receives file mutation and shell tools, and
    every file tool is rooted in the assigned workspace.
    """

    def __init__(self, *, llm_client_factory: Callable[[], Any] | None = None,
                 max_steps: int = 8) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.llm_client_factory = llm_client_factory
        self.max_steps = max_steps

    def __call__(self, assignment: SwarmAssignment) -> SwarmResult:
        from adaptive_harness.agent.agent import DeveloperAgent, DEFAULT_SYSTEM_PROMPT
        from adaptive_harness.tools.bash import RunBashTool
        from adaptive_harness.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
        from adaptive_harness.tools.testing import RunPytestTool
        from adaptive_harness.tools.workspace import ListDirectoryTool, SearchFilesTool

        root = str(assignment.workspace_root)
        tools = [ReadFileTool(workspace_root=root), ListDirectoryTool(workspace_root=root),
                 SearchFilesTool(workspace_root=root)]
        if assignment.may_edit:
            tools += [WriteFileTool(workspace_root=root), EditFileTool(workspace_root=root),
                      RunBashTool(workspace_root=root), RunPytestTool(workspace_root=root)]
        elif assignment.role is SwarmRole.QA and assignment.phase is SwarmPhase.VERIFY:
            tools.append(RunBashTool(workspace_root=root))
            if (assignment.workspace_root / "tests").is_dir():
                tools.append(RunPytestTool(workspace_root=root))
        role_prompt = ("You are a read-only planning or review subagent. Do not attempt write_file or edit_file. "
                       "Inspect available evidence and return a useful plan or review with the tools you have."
                       if not assignment.may_edit else DEFAULT_SYSTEM_PROMPT)
        agent = DeveloperAgent(llm_client=self.llm_client_factory() if self.llm_client_factory else None,
            tools=tools, workspace_root=root,
            forced_mode="security" if assignment.role is SwarmRole.SECURITY else "coding",
            require_file_changes=assignment.may_edit,
            enable_skill_routing=False,
            system_prompt=f"{role_prompt}\n{assignment.instruction}\n"
                          "Only use available tools within the assigned workspace.")
        prompt = assignment.task
        if assignment.context:
            prompt += "\nPrior verified artifacts (JSON):\n" + json.dumps(assignment.context, ensure_ascii=False)
        response: dict[str, Any] = {}
        evidence: list[dict[str, Any]] = []
        for event in agent.run_stream(prompt, max_steps=self.max_steps):
            if event.event_type == "response":
                response = event.payload
            elif event.event_type == "tool_result":
                evidence.append({"tool": event.payload.get("name"),
                                 "success": bool(event.payload.get("success")),
                                 "error": event.payload.get("error")})
        test_results = [item for item in evidence if item["tool"] == "run_pytest"]
        successful_tests = bool(test_results and test_results[-1]["success"])
        summary = str(response.get("content", ""))[:4000]
        refused = bool(re.search(r"\b(?:i cannot|i can't|unable to|cannot coordinate|no write_file|no edit_file)\b",
                                 summary, re.I))
        success = bool(response.get("success")) and not refused
        inspected_files = any(item["tool"] == "read_file" and item["success"] for item in evidence)
        verified = ((successful_tests or inspected_files) if assignment.role is SwarmRole.QA and
                    assignment.phase is SwarmPhase.VERIFY else success)
        return SwarmResult(assignment.role, assignment.phase, success,
            summary,
            {"tool_evidence": evidence, "stop_reason": response.get("stop_reason")}, verified,
            error="Model refused the assigned role" if refused else None)
