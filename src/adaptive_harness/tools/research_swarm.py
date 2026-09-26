"""Agent-facing tools for the autonomous research swarm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from adaptive_harness.tools.base import Tool, ToolResult


class SpawnSubagentTool(Tool):
    """Dynamically scale a research division's worker pool."""

    name = "spawn_subagent"
    description = (
        "Spawn an autonomous research worker under a parent leader. Returns the new agent_id. "
        "Use it to scale a division's pool up for a hard gap, or scale down by retiring idle workers.")
    parameters = {"type": "object", "properties": {
        "parent_id": {"type": "string", "description": "Parent leader or director agent id."},
        "role_name": {"type": "string", "description": "e.g. SymPy Prover, Falsifier, Scout."},
        "directive": {"type": "string", "description": "The single gap this worker must close."},
        "allowed_tools": {"type": "array", "items": {"type": "string"},
                          "description": "Tools this worker may use."},
        "budget_tokens": {"type": "integer", "minimum": 1, "default": 16000},
    }, "required": ["parent_id", "role_name", "directive", "allowed_tools"]}

    def __init__(self, swarm: Any, *, on_spawn: Callable[[str], None] | None = None):
        self.swarm = swarm
        self.on_spawn = on_spawn

    def execute(self, parent_id: str, role_name: str, directive: str,
                allowed_tools: list[str], budget_tokens: int = 16000, **kwargs: Any) -> ToolResult:
        try:
            agent_id = self.swarm.spawn_subagent(parent_id, role_name, directive,
                                                 allowed_tools, int(budget_tokens))
        except (KeyError, ValueError, RuntimeError) as exc:
            return ToolResult(success=False, output="", error=f"Spawn refused: {exc}")
        if self.on_spawn is not None:
            self.on_spawn(agent_id)
        agent = self.swarm.agents[agent_id]
        return ToolResult(success=True, output=json.dumps(agent.to_dict(), ensure_ascii=False),
                          metadata={"agent_id": agent_id, "division": agent.division.value
                                    if agent.division else None})


class ScaleDivisionTool(Tool):
    """Grow or shrink a division's live worker pool."""

    name = "scale_division"
    description = "Scale one research division's worker pool to a target size; returns the changed agent ids."
    parameters = {"type": "object", "properties": {
        "division": {"type": "string", "enum": ["literature", "theory", "empirical", "adversarial"]},
        "size": {"type": "integer", "minimum": 0},
    }, "required": ["division", "size"]}

    def __init__(self, swarm: Any):
        self.swarm = swarm

    def execute(self, division: str, size: int, **kwargs: Any) -> ToolResult:
        from adaptive_harness.research.roles import Division
        try:
            target = Division(division)
        except ValueError:
            return ToolResult(success=False, output="", error=f"Unknown division: {division}")
        try:
            changed = self.swarm.scale_division(target, int(size))
        except (ValueError, RuntimeError) as exc:
            return ToolResult(success=False, output="", error=f"Rescale refused: {exc}")
        live = sorted(agent.agent_id for agent in self.swarm.agents.values()
                      if agent.division is target and not agent.is_leader)
        return ToolResult(success=True,
                          output=json.dumps({"changed": list(changed), "live_workers": live},
                                            ensure_ascii=False),
                          metadata={"division": target.value, "live": len(live)})


class VerifyProofTool(Tool):
    """Run the exact-derivation gate over ``proofs/`` and return receipts."""

    name = "verify_proofs"
    description = ("Execute every proof script and return its receipt. A script must be free of "
                   "floating-point approximation and exit 0 to be admitted as evidence.")
    parameters = {"type": "object", "properties": {}}

    def __init__(self, swarm: Any):
        self.swarm = swarm

    def execute(self, **kwargs: Any) -> ToolResult:
        receipts = self.swarm.proofs.run_all()
        failed = [receipt.theorem_id for receipt in receipts if not receipt.verified]
        payload = {"verified": not failed, "receipts": [receipt.to_dict() for receipt in receipts]}
        return ToolResult(success=not failed and bool(receipts), output=json.dumps(payload, ensure_ascii=False),
                          error=None if (not failed and receipts) else
                          f"{len(failed)} proof(s) not discharged",
                          metadata={"failed": failed})


class RunExperimentTool(Tool):
    """Run the seeded replication gate over ``experiments/``."""

    name = "run_experiments"
    description = ("Execute every experiment script under a pinned seed, hash its data artifacts, "
                   "and report the replication verdict.")
    parameters = {"type": "object", "properties": {"seed": {"type": "integer", "default": 20260926}}}

    def __init__(self, swarm: Any):
        self.swarm = swarm

    def execute(self, seed: int = 20260926, **kwargs: Any) -> ToolResult:
        scripts = self.swarm.experiments.scripts()
        if not scripts:
            return ToolResult(success=False, output="", error="No experiment script exists yet")
        receipts = self.swarm.experiments.run_all({path.name: int(seed) for path in scripts})
        failed = [receipt.experiment_id for receipt in receipts if not receipt.replicated]
        payload = {"replicated": not failed, "seed": int(seed),
                   "receipts": [receipt.to_dict() for receipt in receipts]}
        return ToolResult(success=not failed, output=json.dumps(payload, ensure_ascii=False),
                          error=None if not failed else f"{len(failed)} experiment(s) not replicated",
                          metadata={"failed": failed})


class CompileTypstTool(Tool):
    """Compile a Typst source to PDF, rejecting any build warning."""

    name = "compile_typst"
    description = ("Compile a Typst file to a publication PDF. Typst is resolved from PATH or the "
                   "typst Python wrapper. Any Typst warning fails the build.")
    parameters = {"type": "object", "properties": {
        "typst_path": {"type": "string"},
        "output_pdf_path": {"type": "string"},
    }, "required": ["typst_path"]}

    def __init__(self, workspace_root: str | Path | None = None, *, allow_install: bool = True):
        from adaptive_harness.research.typst import TypstCompiler
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.compiler = TypstCompiler(root=self.workspace_root, allow_install=allow_install)

    def execute(self, typst_path: str, output_pdf_path: str | None = None, **kwargs: Any) -> ToolResult:
        from adaptive_harness.research.typst import TypstError
        try:
            result = self.compiler.compile(typst_path, output_pdf_path)
        except TypstError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        return ToolResult(success=result.success, output=json.dumps(result.to_dict(), ensure_ascii=False),
                          error=result.error, metadata={"pdf_path": str(result.pdf_path),
                                                        "pages": result.pages,
                                                        "typst_version": result.typst_version})


class StopWorkerTool(Tool):
    """Cancel, pause, or resume one research worker, attributing the decision."""

    name = "stop_worker"
    description = (
        "Stop, pause, or resume a named research worker. The actor and reason are required and are "
        "recorded in the audit ledger, so a stop is always attributable. Cancelling also kills the "
        "child processes that worker started and prevents any further tool call by it.")
    parameters = {"type": "object", "properties": {
        "agent_id": {"type": "string", "description": "The worker to act on."},
        "action": {"type": "string", "enum": ["cancel", "pause", "resume"],
                   "description": "cancel stops it for good; pause is resumable."},
        "actor": {"type": "string",
                  "description": "Who is issuing this, e.g. a lead's agent_id. Required."},
        "reason": {"type": "string", "description": "Why. Recorded verbatim. Required."},
    }, "required": ["agent_id", "action", "actor", "reason"]}

    def __init__(self, swarm: Any):
        self.swarm = swarm

    def execute(self, agent_id: str, action: str, actor: str, reason: str,
                **kwargs: Any) -> ToolResult:
        from adaptive_harness.research.coordination import StopKind

        verb = str(action).lower()
        try:
            if verb == "resume":
                if not self.swarm.resume_worker(agent_id):
                    return ToolResult(success=False, output="",
                                      error=f"{agent_id} is not paused, so there is nothing to resume")
                record = self.swarm.control.record(agent_id)
                return ToolResult(success=True, output=json.dumps(record.to_dict(),
                                                                   ensure_ascii=False))
            if verb not in {"cancel", "pause"}:
                return ToolResult(success=False, output="",
                                  error="action must be cancel, pause, or resume")
            applied = (self.swarm.stop_worker(agent_id, actor=actor, reason=reason)
                       if verb == "cancel" else
                       self.swarm.pause_worker(agent_id, actor=actor, reason=reason))
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        if not applied:
            record = self.swarm.control.record(agent_id)
            state = record.state.value if record else "unregistered"
            return ToolResult(success=False, output="",
                              error=f"{agent_id} is {state}; only an active worker can be {verb}ed")
        record = self.swarm.control.record(agent_id)
        assert record is not None
        return ToolResult(success=True, output=json.dumps(record.to_dict(), ensure_ascii=False),
                          metadata={"kind": StopKind.CANCEL.value if verb == "cancel"
                                    else StopKind.PAUSE.value})


class SwarmStatusTool(Tool):
    """Report the live state of the swarm: workers, tasks, gates, and open requests."""

    name = "swarm_status"
    description = (
        "Report the current state of the research swarm: every worker's lifecycle state and "
        "assignment, open board tasks with their owners and artifacts, unanswered help requests, "
        "and how many ledger entries have been recorded. Read-only.")
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, swarm: Any):
        self.swarm = swarm

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True,
                          output=json.dumps(self.swarm.swarm_status(), ensure_ascii=False, indent=2))
