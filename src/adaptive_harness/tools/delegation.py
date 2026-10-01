"""Bounded single-role delegation exposed to a swarm-enabled agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from adaptive_harness.tools.base import Tool, ToolResult, workspace_path


class DelegateSubagentTool(Tool):
    name = "delegate_subagent"
    description = "Run an architect, coder, reviewer, or security subagent on a bounded workspace task."
    parameters = {"type": "object", "properties": {
        "role": {"type": "string", "enum": ["architect", "coder", "reviewer", "security"]},
        "task": {"type": "string"},
        "target_dir": {"type": "string", "description": "Optional directory inside the workspace"},
    }, "required": ["role", "task"]}

    def __init__(self, workspace_root: str | Path, *, llm_client_factory: Callable[[], Any] | None = None,
                 repository: Any = None, safety_profile: str = "turbo",
                 step_policy: str = "classifier", max_steps: int | None = None,
                 on_event: Callable[[Any], None] | None = None,
                 registry: Any = None, parent_id: str = "main",
                 agents: Any = None):
        self.workspace_root = Path(workspace_root).resolve()
        self.llm_client_factory = llm_client_factory
        self.repository = repository
        self.safety_profile = safety_profile
        self.step_policy = step_policy
        self.max_steps = max_steps
        self.on_event = on_event
        # A subagent that cannot be named cannot be watched or stopped. The
        # registry gives each one an id, a lifecycle, and a place in the
        # monitoring view.
        self.registry = registry
        self.agents = agents
        self.parent_id = parent_id

    def execute(self, role: str, task: str, target_dir: str | None = None, **kwargs: Any) -> ToolResult:
        from adaptive_harness.agent.swarm import (DeveloperAgentWorker, SwarmAssignment,
                                                   SwarmPhase, SwarmRole, run_assignment)

        roles = {"architect": (SwarmRole.ARCHITECT, SwarmPhase.PLAN),
                 "coder": (SwarmRole.CODER, SwarmPhase.IMPLEMENT),
                 "reviewer": (SwarmRole.QA, SwarmPhase.VERIFY),
                 "security": (SwarmRole.SECURITY, SwarmPhase.VERIFY)}

        # A subagent is now a *file*, not one of four hardcoded roles. A name
        # that matches a definition runs that definition; a name that matches a
        # built-in role still runs the role, so nothing that worked before
        # stops working.
        from adaptive_harness.agents import AgentRegistry, builtin_definitions, resolve

        # `registry` tracks *running* agents; `agents` holds the *definitions*.
        # They are different things, and conflating them sends a lookup here to a
        # method only the former has.
        if self.agents is None:
            self.agents = AgentRegistry(self.workspace_root)
            self.agents.discover()
        definition = resolve(role, self.agents)
        if definition is None:
            known = sorted(set(self.agents.names()) | set(builtin_definitions()))
            return ToolResult(success=False, output="",
                              error=f"No subagent named {role!r}. Define one at "
                                    f".harness/agents/{role}.md, or use one of: "
                                    f"{', '.join(known)}")
        if role not in roles:
            # A user-defined agent has no built-in phase. It is planned unless
            # it asked to write, which is the honest default for something
            # whose whole job is to produce an answer.
            roles[role] = ((SwarmRole.CODER if definition.is_write_capable
                            else SwarmRole.ARCHITECT),
                           SwarmPhase.IMPLEMENT if definition.is_write_capable
                           else SwarmPhase.PLAN)

        if not isinstance(task, str) or not task.strip():
            return ToolResult(success=False, output="",
                              error=f"Give the {role} agent something to do.")
        try:
            root = workspace_path(self.workspace_root, target_dir or ".")
            selected_role, phase = roles[role]
            if selected_role is SwarmRole.CODER:
                root.mkdir(parents=True, exist_ok=True)
            elif not root.is_dir():
                return ToolResult(success=False, output="",
                                  error=f"Read-only subagent target directory does not exist: {target_dir}")
            assignment = SwarmAssignment(selected_role, phase, task, root)

            record = None
            reporter = None
            if self.registry is not None:
                # Imported here: `agent.agent` imports this module, so a
                # module-scope import would be a cycle.
                from adaptive_harness.agent.agent import AgentEvent
                from adaptive_harness.agent.subagents import (
                    SubagentReporter, complete_event, spawn_event)

                record = self.registry.spawn(parent_id=self.parent_id, role=role,
                                             description=task.strip()[:120])
                reporter = SubagentReporter(self.registry)
                if self.on_event is not None:
                    self.on_event(AgentEvent("agent_spawned", spawn_event(record)))

            def forward(event: Any) -> None:
                if reporter is not None and record is not None:
                    reporter.on_event(record.id, event)
                if self.on_event is not None:
                    self.on_event(event)

            # The definition decides the tool surface, not just the name. An
            # agent handed write_file when its file said `permissionMode: plan`
            # would be a promise the file did not keep.
            tools = list(definition.tools) or None
            if tools is not None and definition.disallowed_tools:
                tools = [name for name in tools if name not in definition.disallowed_tools]
            worker = DeveloperAgentWorker(llm_client_factory=self.llm_client_factory,
                repository=self.repository, safety_profile=self.safety_profile,
                step_policy=self.step_policy,
                max_steps=definition.max_turns or self.max_steps,
                tool_names=tools,
                system_prompt=definition.system_prompt or None,
                on_event=forward)
            result = run_assignment(worker, assignment)
            if record is not None and self.registry is not None:
                self.registry.complete(record.id, result=result.summary,
                                       error=result.error,
                                       stop_reason=str((result.artifacts or {}).get("stop_reason") or ""))
                if self.on_event is not None:
                    self.on_event(AgentEvent("agent_completed", complete_event(record)))

            # The parent gets the ledger, not the transcript. Replaying a
            # subagent's chatter into every later request is one of the largest
            # avoidable costs in a tool loop, and the full stream is still
            # available in the monitoring view for anyone who asks.
            payload = result.to_dict()
            if record is not None:
                payload["agent_id"] = record.id
                payload["ledger"] = record.ledger()
            return ToolResult(success=result.success, output=json.dumps(payload, ensure_ascii=False),
                              error=result.error,
                              metadata={"role": role, "verified": result.verified,
                                        "agent_id": record.id if record else None})
        except (OSError, ValueError) as exc:
            return ToolResult(success=False, output="", error=f"Delegation failed: {exc}")
