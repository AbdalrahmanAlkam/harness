"""Bounded, artifact-driven collaboration for a developer task.

Only one worker may edit the worktree. Independent planning and final review
jobs run concurrently; each receives compact JSON artifacts from earlier jobs
instead of another agent's conversation history.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

# A refusal is structural: explicit decline phrasing AND no completed tool work.
_REFUSAL_PHRASES = re.compile(
    r"\b(?:i refuse|i must decline"
    r"|i (?:will not|won't|cannot|can't) (?:do|perform|comply|carry out|assist|help with)"
    r"|unable to (?:assist|comply|perform|do)|cannot (?:assist|comply)"
    r"|(?:no|not) (?:write_file|edit_file) (?:tool )?is available)\b", re.I)

_INSPECTION_TOOLS = {"read_file", "search_files", "list_directory", "run_bash", "run_pytest"}


def is_role_refusal(summary: str, had_successful_tools: bool) -> bool:
    """Detect an actual role refusal, not benign role-limitation language."""
    return not had_successful_tools and bool(_REFUSAL_PHRASES.search(summary or ""))


class SwarmRole(str, Enum):
    ARCHITECT = "architect"
    CODER = "coder"
    QA = "qa"
    SECURITY = "security"


class SwarmPhase(str, Enum):
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"


def _role_instruction(role: SwarmRole, phase: SwarmPhase, workspace_root: Path) -> str:
    """Editable role instructions live in the prompt registry (swarm.instruction.*)."""
    from adaptive_harness.prompts import PromptRegistry
    return PromptRegistry.for_workspace(workspace_root).get(
        f"swarm.instruction.{role.value}.{phase.value}")


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
        return _role_instruction(self.role, self.phase, self.workspace_root)


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


def _invoke(worker: SwarmWorker, assignment: SwarmAssignment) -> SwarmResult:
    """Run one assignment with strict boundary validation."""
    try:
        result = worker(assignment)
        if result.role is not assignment.role or result.phase is not assignment.phase:
            raise ValueError("Worker returned a result for a different assignment")
        # Reject opaque, unserializable payloads at the boundary.
        json.dumps(result.to_dict(), ensure_ascii=False)
        return result
    except Exception as exc:
        return SwarmResult(assignment.role, assignment.phase, False, "Subagent failed",
                           error=f"{type(exc).__name__}: {exc}")


def _is_deterministic_failure(result: SwarmResult) -> bool:
    """Refusals and payload-validation errors do not benefit from a retry."""
    error = result.error or ""
    return (error == "Model refused the assigned role"
            or "different assignment" in error
            or "not JSON serializable" in error)


def run_assignment(worker: SwarmWorker, assignment: SwarmAssignment, *,
                   retry: bool = True) -> SwarmResult:
    """Execute one assignment; retry transient failures once with failure feedback.

    The retry prompt carries the previous failure so the subagent can correct
    instead of repeating the same mistake. Deterministic failures (refusals,
    boundary violations) are returned immediately.
    """
    result = _invoke(worker, assignment)
    if retry and not result.success and not _is_deterministic_failure(result):
        feedback = {"prior_failure": {"error": result.error, "summary": result.summary[:500],
                                      "stop_reason": result.artifacts.get("stop_reason")}}
        retried = _invoke(worker, replace(assignment,
                                          context={**dict(assignment.context), **feedback}))
        if retried.success or not retried.error:
            return retried
        return SwarmResult(retried.role, retried.phase, False,
                           retried.summary or "Subagent failed after retry",
                           retried.artifacts, retried.verified, error=retried.error)
    return result


class SwarmCoordinator:
    """Run read-only roles in parallel and serialize the sole writer.

    The caller supplies a workspace. Only the coder can mutate files, so
    direct workspaces remain safe from concurrent agent writes.
    """

    def __init__(self, worker: SwarmWorker, *, max_workers: int = 2,
                 on_status: StatusCallback | None = None,
                 max_repair_cycles: int = 2) -> None:
        if max_workers < 1 or max_workers > 4:
            raise ValueError("max_workers must be between 1 and 4")
        if max_repair_cycles < 0:
            raise ValueError("max_repair_cycles must be non-negative")
        self.worker = worker
        self.max_workers = max_workers
        self.on_status = on_status
        self.max_repair_cycles = max_repair_cycles

    def run(self, task: str, workspace_root: str | Path, *,
            isolated: bool = False) -> SwarmReport:
        root = Path(workspace_root).resolve()
        if not root.is_dir():
            raise ValueError(f"Workspace does not exist: {root}")
        if not task.strip():
            raise ValueError("Swarm task cannot be empty")

        statuses = {"architect:plan": "queued", "coder:implement": "queued",
                    "qa:verify": "queued", "security:verify": "queued"}
        results: list[SwarmResult] = []
        self._notify(statuses)

        def wave(assignments: tuple[SwarmAssignment, ...]) -> list[SwarmResult]:
            for assignment in assignments:
                statuses[assignment.key] = "running"
            self._notify(statuses)
            collected: dict[str, SwarmResult] = {}
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(assignments))) as pool:
                futures = {pool.submit(run_assignment, self.worker, assignment): assignment
                           for assignment in assignments}
                for future in as_completed(futures):
                    assignment = futures[future]
                    result = future.result()
                    statuses[assignment.key] = "done" if result.success else "failed"
                    collected[assignment.key] = result
                    self._notify(statuses)
            return [collected[assignment.key] for assignment in assignments]

        def artifacts(items: list[SwarmResult]) -> dict[str, Any]:
            return {item.key: item.to_dict() for item in items}

        planning = wave((SwarmAssignment(SwarmRole.ARCHITECT, SwarmPhase.PLAN, task, root),))
        results.extend(planning)
        implementation = wave((SwarmAssignment(SwarmRole.CODER, SwarmPhase.IMPLEMENT,
            task, root, artifacts(planning)),))
        results.extend(implementation)
        if not implementation[0].success:
            statuses["qa:verify"] = statuses["security:verify"] = "skipped"
            self._notify(statuses)
            return SwarmReport(tuple(results), False, dict(statuses))

        implemented = implementation[0]
        # Read-only reviews run in parallel; only the coder mutates the workspace.
        def review_wave() -> tuple[SwarmResult, SwarmResult]:
            context = artifacts(results)
            security, qa = wave((SwarmAssignment(SwarmRole.SECURITY, SwarmPhase.VERIFY, task, root, context),
                                 SwarmAssignment(SwarmRole.QA, SwarmPhase.VERIFY, task, root, context)))
            results.extend((security, qa))
            return security, qa

        security, qa = review_wave()
        # Repair loop: failed or unverified reviews route back to the coder with
        # the review findings as feedback, bounded by max_repair_cycles.
        repairs = 0
        while (not (qa.success and qa.verified and security.success)) and repairs < self.max_repair_cycles:
            repairs += 1
            feedback = {"review_findings": {item.key: item.to_dict() for item in (security, qa)
                                            if not (item.success and item.verified)}}
            repair = wave((SwarmAssignment(SwarmRole.CODER, SwarmPhase.IMPLEMENT, task, root,
                {**artifacts(results), **feedback}),))
            results.extend(repair)
            implemented = repair[0]
            if not implemented.success:
                break
            security, qa = review_wave()
        success = bool(implemented.success and qa.success and qa.verified and security.success)
        return SwarmReport(tuple(results), success, dict(statuses))

    def _notify(self, status: Mapping[str, str]) -> None:
        if self.on_status is not None:
            self.on_status(dict(status))


class DeveloperAgentWorker:
    """Opt-in role runner using separate agent state for each assignment.

    Only the coder receives file mutation and shell tools, and
    every file tool is rooted in the assigned workspace.

    ``tool_names`` overrides the default set with an explicit list, which is how
    the research swarm equips a division with the instruments its methodology
    needs — a Lean formaliser needs ``run_lean_proof``, a simulator needs the
    Python REPL — rather than the coder/reviewer split used for software work.
    """

    def __init__(self, *, llm_client_factory: Callable[[], Any] | None = None,
                 max_steps: int | None = None,
                 step_policy: str = "classifier",
                 repository: Any = None,
                 safety_profile: str = "turbo",
                 tool_names: Sequence[str] | None = None,
                 forced_mode: str | None = None,
                 forced_thinking: str | None = None,
                 enable_skill_routing: bool = False,
                 write_target: Path | None = None,
                 system_prompt: str | None = None,
                 on_event: Callable[[Any], None] | None = None) -> None:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.llm_client_factory = llm_client_factory
        self.max_steps = max_steps
        self.step_policy = step_policy
        self.repository = repository
        self.safety_profile = safety_profile
        self.tool_names = tuple(tool_names) if tool_names else None
        self.forced_mode = forced_mode
        self.forced_thinking = forced_thinking
        self.enable_skill_routing = enable_skill_routing
        self.write_target = write_target
        self.system_prompt = system_prompt
        self.on_event = on_event

    def _build_tools(self, root: str, assignment: SwarmAssignment) -> list[Any]:
        """Construct the tool list for one assignment."""
        from adaptive_harness.tools.bash import RunBashTool
        from adaptive_harness.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
        from adaptive_harness.tools.lean import RunLeanProofTool
        from adaptive_harness.tools.research import WebSearchTool
        from adaptive_harness.tools.research_swarm import CompileTypstTool
        from adaptive_harness.tools.python_repl import RunPythonReplTool
        from adaptive_harness.tools.testing import RunPytestTool
        from adaptive_harness.tools.workspace import ListDirectoryTool, SearchFilesTool

        if self.tool_names is not None:
            available = {
                "read_file": lambda: ReadFileTool(workspace_root=root),
                "write_file": lambda: WriteFileTool(
                    workspace_root=root,
                    allowed_paths=(self.write_target,) if self.write_target else None),
                "edit_file": lambda: EditFileTool(workspace_root=root),
                "list_directory": lambda: ListDirectoryTool(workspace_root=root),
                "search_files": lambda: SearchFilesTool(workspace_root=root),
                "run_bash": lambda: RunBashTool(workspace_root=root),
                "run_pytest": lambda: RunPytestTool(workspace_root=root),
                "run_python_repl": lambda: RunPythonReplTool(),
                "run_lean_proof": lambda: RunLeanProofTool(workspace_root=root,
                                                            lean_dir="proofs/lean"),
                "web_search": lambda: WebSearchTool(allow_public_metadata=True),
                "compile_typst": lambda: CompileTypstTool(workspace_root=root),
            }
            unknown = [name for name in self.tool_names if name not in available]
            if unknown:
                raise ValueError(f"Unsupported worker tools requested: {unknown}")
            return [available[name]() for name in self.tool_names]

        tools = [ReadFileTool(workspace_root=root), ListDirectoryTool(workspace_root=root),
                 SearchFilesTool(workspace_root=root)]
        if assignment.may_edit:
            tools += [WriteFileTool(workspace_root=root), EditFileTool(workspace_root=root),
                      RunBashTool(workspace_root=root), RunPytestTool(workspace_root=root)]
        elif assignment.role is SwarmRole.QA and assignment.phase is SwarmPhase.VERIFY:
            tools.append(RunBashTool(workspace_root=root, read_only=True))
            if (assignment.workspace_root / "tests").is_dir():
                tools.append(RunPytestTool(workspace_root=root))
        return tools

    def __call__(self, assignment: SwarmAssignment) -> SwarmResult:
        from adaptive_harness.agent.agent import DeveloperAgent
        from adaptive_harness.prompts import PromptRegistry

        root = str(assignment.workspace_root)
        tools = self._build_tools(root, assignment)
        prompts = PromptRegistry.for_workspace(assignment.workspace_root)
        role_prompt = self.system_prompt or (
            prompts.get("system.default") + "\n" + prompts.get("swarm.role.coder") if assignment.may_edit else
            prompts.get("swarm.role.security") if assignment.role is SwarmRole.SECURITY else
            prompts.get("swarm.role.read_only"))
        agent = DeveloperAgent(llm_client=self.llm_client_factory() if self.llm_client_factory else None,
            tools=tools, workspace_root=root, repository=self.repository,
            forced_mode=self.forced_mode or ("security" if assignment.role is SwarmRole.SECURITY else "coding"),
            forced_thinking=self.forced_thinking,
            require_file_changes=assignment.may_edit,
            enable_skill_routing=self.enable_skill_routing,
            step_policy=self.step_policy,
            safety_profile=self.safety_profile,
            system_prompt=f"{role_prompt}\n{assignment.instruction}\n"
                          + prompts.get("swarm.workspace_limit"),
            prompts=prompts)
        prompt = assignment.task
        if assignment.context:
            prompt += prompts.get("swarm.context_header") + json.dumps(assignment.context, ensure_ascii=False)
        response: dict[str, Any] = {}
        evidence: list[dict[str, Any]] = []
        for event in agent.run_stream(prompt, max_steps=self.max_steps):
            if self.on_event is not None:
                self.on_event(event)
            if event.event_type == "response":
                response = event.payload
            elif event.event_type == "tool_result":
                evidence.append({"tool": event.payload.get("name"),
                                 "success": bool(event.payload.get("success")),
                                 "error": event.payload.get("error")})
        stop_reason = response.get("stop_reason")
        test_results = [item for item in evidence if item["tool"] == "run_pytest"]
        successful_tests = bool(test_results and test_results[-1]["success"])
        summary = str(response.get("content", ""))[:4000]
        if not summary.strip():
            # Failure paths can clear the answer; never report an empty summary.
            summary = f"(no final answer; stop reason: {stop_reason or 'unknown'})"
        had_successful_tools = any(item["success"] for item in evidence)
        refused = is_role_refusal(summary, had_successful_tools)
        success = bool(response.get("success")) and not refused
        inspected = any(item["tool"] in _INSPECTION_TOOLS and item["success"] for item in evidence)
        if assignment.role is SwarmRole.QA and assignment.phase is SwarmPhase.VERIFY:
            # Verification requires evidence: passing tests or any successful
            # inspection tool call (file reads, searches, shell checks). This
            # keeps test-less repositories verifiable without letting a QA run
            # pass on claims alone.
            verified = bool(successful_tests or inspected)
        else:
            verified = success
        error = ("Model refused the assigned role" if refused else
                 None if success else
                 f"Subagent run did not complete (stop reason: {stop_reason or 'unknown'})")
        return SwarmResult(assignment.role, assignment.phase, success,
            summary,
            {"tool_evidence": evidence, "stop_reason": stop_reason}, verified, error=error)
