"""Bounded single-role delegation exposed to a swarm-enabled agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from adaptive_harness.tools.base import Tool, ToolResult, workspace_path


#: How much of a subagent's own words come back to the parent. The rest stays
#: in the registry. A subagent that writes a 4,000-character answer would
#: otherwise pay for those 4,000 characters on every later request.
RESULT_TO_PARENT_CHARS = 2000


class BackgroundResultTool(Tool):
    """Collect a subagent that was started with ``wait: false``.

    The tool schema tells the model to collect a background agent by id, so
    this has to be a real tool. An id the model is told to use, with nothing to
    use it on, is an instruction to guess.
    """

    name = "background_result"
    description = ("Collect a backgrounded subagent by the agent_id returned when it "
                   "was started. Reports whether it is still running.")
    parameters = {"type": "object", "properties": {
        "agent_id": {"type": "string", "description": "The id the background call returned."},
    }, "required": ["agent_id"]}

    def __init__(self, delegate: "DelegateSubagentTool") -> None:
        self.delegate = delegate

    def execute(self, agent_id: str, **kwargs: Any) -> ToolResult:
        return self.delegate.background_result(agent_id)


class DelegateSubagentTool(Tool):
    name = "delegate_subagent"
    description = "Run an architect, coder, reviewer, or security subagent on a bounded workspace task."
    parameters = {"type": "object", "properties": {
        "role": {"type": "string", "enum": ["architect", "coder", "reviewer", "security"]},
        "task": {"type": "string"},
        "target_dir": {"type": "string", "description": "Optional directory inside the workspace"},
        "wait": {"type": "boolean",
                 "description": "Wait for the subagent to finish (default). Pass false to "
                                "run it in the background and get an id back immediately; "
                                "collect it later with background_result."},
    }, "required": ["role", "task"]}

    def __init__(self, workspace_root: str | Path, *, llm_client_factory: Callable[[], Any] | None = None,
                 repository: Any = None, safety_profile: str = "turbo",
                 step_policy: str = "classifier", max_steps: int | None = None,
                 on_event: Callable[[Any], None] | None = None,
                 registry: Any = None, parent_id: str = "main",
                 agents: Any = None, plugins: Any = None):
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
        self.plugins = plugins
        # `background: true` runs the subagent without blocking the caller's
        # turn. The id comes back immediately and the run is picked up from
        # `background_result` when asked, which is the whole point: the parent
        # can keep working instead of sitting idle.
        self._background: dict[str, Any] = {}
        self.parent_id = parent_id

    @staticmethod
    def _finish_isolated(manager, worktree, result, definition):
        """Say what an isolated agent did, and leave its work for review.

        The patch is printed rather than applied: the whole point of isolation is
        that a human decides whether the change is wanted. An agent that changed
        nothing leaves no worktree behind.
        """
        from adaptive_harness.agent.swarm import SwarmResult

        try:
            patch = manager.patch(worktree)
        except (OSError, RuntimeError):
            patch = ""
        if not patch.strip():
            manager.abort(worktree)
            return result
        note = (f"\n\n[{definition.name} worked in an isolated worktree at "
                f"{worktree.path}. Its changes are NOT applied. Review them with "
                f"`git -C {worktree.path} diff`, and apply with "
                f"`git apply` or `harness review`.\n\n```diff\n{patch[:6000]}\n```")
        return SwarmResult(
            role=result.role, phase=result.phase, success=result.success,
            summary=(result.summary or "") + note, artifacts=dict(result.artifacts),
            verified=result.verified, error=result.error, agent_id=result.agent_id)

    def background_result(self, agent_id: str) -> ToolResult:
        """Collect a backgrounded subagent, or say it is still going.

        A background agent that cannot be collected is worse than one that
        blocks: the caller holds an id and no way to turn it into an answer.
        """
        entry = self._background.get(agent_id)
        if entry is None:
            known = ", ".join(sorted(self._background)) or "none"
            return ToolResult(success=False, output="",
                              error=f"No background subagent with id {agent_id!r}. "
                                    f"Running: {known}")
        if not entry.get("done"):
            return ToolResult(success=True, output=json.dumps({
                "agent_id": agent_id, "status": "still running"}),
                metadata={"agent_id": agent_id, "done": False})
        if entry.get("error"):
            return ToolResult(success=False, output="", error=entry["error"],
                              metadata={"agent_id": agent_id, "done": True})
        result = entry["result"]
        return ToolResult(success=result.success,
                          output=json.dumps(result.to_dict(), ensure_ascii=False),
                          error=result.error, metadata={"agent_id": agent_id, "done": True})

    def execute(self, role: str, task: str, target_dir: str | None = None,
                wait: bool = True, **kwargs: Any) -> ToolResult:
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
        # A user file may shadow a built-in name, and then it is a user file
        # that runs -- with its own permissions. Reading the role table first
        # gave a `permissionMode: plan` agent the implement phase and
        # `require_file_changes`, so it held no write tool and could never
        # succeed, burning its whole retry budget every time.
        shadowed = self.agents is not None and self.agents.get(role) is not None
        if role not in roles or shadowed:
            # Planned unless it asked to write, which is the honest default for
            # something whose whole job is to produce an answer.
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
            # `isolation: worktree` gives the subagent its own checkout, so its
            # edits are reviewable rather than already applied. Parsed but
            # ignored, the flag was a promise the file made and nothing kept.
            worktree = None
            manager = None
            if definition.isolation == "worktree":
                from adaptive_harness.workspace.worktree import WorktreeError, WorktreeManager

                # The constructor itself raises outside a repository, so the
                # check has to come first or the reason never reaches the user.
                if not WorktreeManager.is_git_repo(root):
                    return ToolResult(
                        success=False, output="",
                        error=(f"{definition.name} asks for isolation: worktree, but "
                               f"{root} is not a git repository. An isolated agent "
                               f"needs somewhere to put its changes."))
                try:
                    manager = WorktreeManager(root)
                    worktree = manager.create()
                except (WorktreeError, OSError, RuntimeError) as exc:
                    return ToolResult(success=False, output="",
                                      error=f"Could not create a worktree: {exc}")
                root = worktree.workspace

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
                record.color = definition.color
                reporter = SubagentReporter(self.registry)
                if self.plugins is not None:
                    self.plugins.run_subagent_start(record)
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
            # `effective_tools` applies the definition's own permissions, so a
            # `permissionMode: plan` agent cannot be handed write_file.
            # Two cases that look alike and need opposite defaults.
            # Declaring no tools at all means "use the default set"; declaring
            # tools and having every one filtered out means "this agent may do
            # nothing" -- and handing it the full default set there is the
            # worst possible answer, since a definition that disallowed both
            # writing and reading would get both.
            if not definition.tools:
                tools = None
            else:
                tools = list(definition.effective_tools())
                if not tools:
                    return ToolResult(
                        success=False, output="",
                        error=(f"Every tool {definition.name} declares was removed by its "
                               f"own permissions, so it would fall back to the full "
                               f"default set. Give it a tool it is allowed, or drop "
                               f"the `tools:` line to use the defaults."))
            worker = DeveloperAgentWorker(llm_client_factory=self.llm_client_factory,
                repository=self.repository, safety_profile=self.safety_profile,
                step_policy=self.step_policy,
                max_steps=definition.max_turns or self.max_steps,
                tool_names=tools,
                system_prompt=definition.system_prompt or None,
                # `model:` and `effort:` are the reason a subagent can cost
                # less than the parent. Parsed but ignored, they were a promise
                # the file made and the harness did not keep.
                explicit_model=definition.model or None,
                forced_thinking=definition.effort or None,
                memory_tier=definition.memory or "",
                on_event=forward)
            # A definition marked `background: true` runs without holding the
            # caller's turn. The id comes back immediately and the result is
            # collected later, which is the point: the parent keeps working
            # instead of sitting idle while a reviewer reads a diff.
            run_it = lambda: run_assignment(worker, assignment)  # noqa: E731
            if definition.background and not wait:
                if record is None:
                    # A background agent is addressed by id, so it cannot exist
                    # without one. Failing here is better than returning an id
                    # that will never resolve.
                    return ToolResult(success=False, output="",
                                      error=f"{definition.name} is a background agent, "
                                            f"which needs a subagent registry to be "
                                            f"addressable. Run through the TUI or the "
                                            f"swarm, which supply one.")
                import threading

                def background_run() -> None:
                    # The registry has to learn the agent finished, exactly as
                    # it does for a blocking run. Otherwise `/tasks` shows an
                    # agent still running for work that ended minutes ago --
                    # the precise state a monitoring view exists to prevent.
                    try:
                        result = run_it()
                        if worktree is not None:
                            result = self._finish_isolated(manager, worktree,
                                                           result, definition)
                        self._background[record.id] = {"result": result, "done": True}
                    except Exception as exc:  # noqa: BLE001 - reported, not lost
                        result = None
                        self._background[record.id] = {
                            "result": None, "done": True,
                            "error": f"{type(exc).__name__}: {exc}"}
                    finally:
                        if record is not None and self.registry is not None:
                            self.registry.complete(
                                record.id,
                                result=(result.summary if result is not None else ""),
                                error=(result.error or "") if result is not None else "",
                                stop_reason=((result.artifacts or {}).get("stop_reason") or ""))

                thread = threading.Thread(target=background_run, daemon=True,
                                         name=f"subagent-{record.id}")
                thread.start()
                return ToolResult(success=True, output=json.dumps({
                    "agent_id": record.id, "status": "running in the background",
                    "how_to_collect": "call background_result with this id",
                }), metadata={"agent_id": record.id, "background": True})
            result = run_it()
            if worktree is not None:
                result = self._finish_isolated(manager, worktree, result, definition)
            # A stop hook may send a correction back to the subagent. This is
            # what makes the hook useful rather than decorative: "you did not
            # run the tests" has to reach the model that can act on it.
            correction = ""
            if record is not None and self.plugins is not None:
                # Guarded at the call site as well as inside the host. A hook is
                # third-party code on the critical path of finishing a subagent,
                # and it must not be able to fail the run it was only observing.
                try:
                    correction = self.plugins.run_subagent_stop(record) or ""
                except Exception:  # noqa: BLE001 - a hook must not break the run
                    correction = ""
                if correction and self.on_event is not None:
                    from adaptive_harness.agent.agent import AgentEvent

                    self.on_event(AgentEvent("agent_correction",
                                            {"agent_id": record.id,
                                             "feedback": correction}))
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
            # What the parent receives is what gets replayed on every later
            # request, so it is bounded here rather than in the display. The
            # ledger is the capped account; the raw summary and the full tool
            # evidence stay in the registry for anyone who asks to see them.
            payload = {
                "role": result.role.value,
                "phase": result.phase.value,
                "success": result.success,
                "verified": result.verified,
                "error": result.error,
                "agent_id": record.id if record is not None else None,
                "summary": (result.summary or "")[:RESULT_TO_PARENT_CHARS],
                "ledger": record.ledger() if record is not None else "",
            }
            if record is not None:
                payload["tools"] = len(record.tools)
                payload["input_tokens"] = record.input_tokens
                payload["output_tokens"] = record.output_tokens
                payload["elapsed_ms"] = record.elapsed_ms
                payload["truncated"] = len(result.summary or "") > RESULT_TO_PARENT_CHARS
            # A tool the definition asked for and this build does not have. The
            # run continues without it, so the fact has to reach the user: an
            # agent quietly missing the tool it was declared with is a promise the
            # file did not keep.
            dropped = list(getattr(worker, "unknown_tools", ()) or ())
            if dropped:
                payload["unavailable_tools"] = dropped
                if self.on_event is not None:
                    from adaptive_harness.agent.agent import AgentEvent

                    self.on_event(AgentEvent("agent_tool_unavailable",
                                            {"agent": definition.name, "tools": dropped}))
            return ToolResult(success=result.success, output=json.dumps(payload, ensure_ascii=False),
                              error=result.error,
                              metadata={"role": role, "verified": result.verified,
                                        "agent_id": record.id if record else None,
                                        "unavailable_tools": dropped})
        except (OSError, ValueError) as exc:
            return ToolResult(success=False, output="", error=f"Delegation failed: {exc}")
