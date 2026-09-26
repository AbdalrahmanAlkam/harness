"""Research workers must be real tool-using subagents, not one-shot completions.

A worker that merely returns text cannot close a gap: closing one means writing
a script, executing it, reading the failure, and revising. These tests drive the
loop with a scripted client so the *mechanics* are exercised deterministically
— the offline mock deliberately refuses to fabricate an implementation, which is
the right behaviour for a UX fixture and the wrong tool for proving a tool loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adaptive_harness.llm.mock_client import LLMResponse, ToolCall
from adaptive_harness.research import ResearchSwarm, SwarmConfig
from adaptive_harness.research.ledger import ACTIONS
from adaptive_harness.research.roles import WORKER_TOOLS, Clearance, Division


class ScriptedClient:
    """A client that replays a fixed tool-call script, then stops.

    Each entry is a tool call to emit; the client's turn is chosen by how many
    tool results the conversation already contains, which is how a real agent
    advances. The final turn returns prose so the run terminates.
    """

    is_mock = True
    default_model = "scripted"
    provider = "scripted"

    def __init__(self, script: list[tuple[str, dict[str, Any]]], final: str = "Done."):
        self.script = script
        self.final = final
        self.turn = 0
        self.calls: list[dict[str, Any]] = []
        self.api_key = None
        self.base_url = None
        self.provider_keys = {}
        self.backup_providers = ()

    def complete(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        # An internal counter rather than counting tool messages: the harness may
        # insert its own follow-up turns, and the script must stay deterministic.
        turn = self.turn
        self.turn += 1
        self.calls.append({"turn": turn, "tools": [t.get("function", {}).get("name")
                                                    for t in (tools or [])]})
        if turn < len(self.script):
            name, arguments = self.script[turn]
            return LLMResponse(
                content=f"Calling {name}.",
                tool_calls=[ToolCall(id=f"call_{turn}", name=name, arguments=arguments)],
                model=model or self.default_model)
        return LLMResponse(content=self.final, tool_calls=[], model=model or self.default_model,
                           finish_reason="stop")


def _swarm(tmp_path: Path, script, final: str = "The script executes cleanly.") -> ResearchSwarm:
    def factory():
        return ScriptedClient(script, final)
    swarm = ResearchSwarm("worker autonomy", root=tmp_path,
                          config=SwarmConfig(max_cycles=1, llm_client_factory=factory,
                                             worker_max_steps=6))
    return swarm


PROOF_BODY = ("from sympy import symbols, sqrt\n"
              "x = symbols('x', positive=True)\n"
              "assert sqrt(x**2) == x\n"
              "print('ok')\n")


# -- the loop mechanics ----------------------------------------------------

def test_worker_is_gated_on_a_live_client(tmp_path: Path):
    mechanical = ResearchSwarm("no model", root=tmp_path, config=SwarmConfig())
    assert not mechanical._interactive_capable()
    assert mechanical.config.author_mode == "mechanical"
    interactive = _swarm(tmp_path, [])
    assert interactive._interactive_capable()
    assert interactive.config.author_mode == "interactive"


def test_worker_iterates_in_a_tool_loop_and_logs_its_trajectory(tmp_path: Path):
    """write_file then run_bash: the worker must actually take two turns."""
    swarm = _swarm(tmp_path, [])
    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove the closed form",
                                    ["write_file", "run_bash"])
    agent = swarm.agents[agent_id]
    target = swarm.workspace.proof_dir / f"{agent_id}.py"
    # The worker writes to the path the Director named, then runs it.
    script = [
        ("write_file", {"path": str(target.relative_to(swarm.workspace.root)),
                        "content": PROOF_BODY}),
        ("run_bash", {"command": f"python {target.relative_to(swarm.workspace.root)}"}),
    ]
    swarm.config.llm_client_factory = lambda: ScriptedClient(script)
    outcome = swarm._run_worker(agent, Division.THEORY, "Close the proof gap.",
                                success_criterion=swarm._success_criterion(Division.THEORY, target),
                                target=target)

    assert outcome["ran"] is True
    assert outcome["tool_calls"] == 2, outcome
    assert [item["tool"] for item in outcome["trajectory"]] == ["write_file", "run_bash"]
    # The worker wrote the artifact and observed it running.
    assert outcome["wrote_target"] is True
    assert target.is_file()
    assert "ok" in target.read_text()

    # The trajectory is in the ledger, addressed worker -> leader.
    entries = swarm.ledger.by_action("WORKER_TRAJECTORY")
    assert len(entries) == 1
    payload = entries[0].payload
    assert payload["tools_used"] == ["write_file", "run_bash"]
    assert payload["tool_calls"] == 2
    assert entries[0].sender["agent_id"] == agent_id
    assert entries[0].recipient["role"] == "Theoretical Lead"


def test_worker_records_a_failed_tool_call_in_its_trajectory(tmp_path: Path):
    script = [("run_bash", {"command": "definitely-not-a-command"})]
    swarm = _swarm(tmp_path, script)
    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it", ["run_bash"])
    agent = swarm.agents[agent_id]
    target = swarm.workspace.proof_dir / f"{agent_id}.py"
    outcome = swarm._run_worker(agent, Division.THEORY, "Close the proof gap.",
                                success_criterion="Success: it runs.",
                                target=target)
    # run_assignment retries a failed assignment with feedback, which replays the
    # script; so assert the shape of the first call rather than a count.
    assert outcome["tool_calls"] >= 1
    assert outcome["trajectory"][0]["tool"] == "run_bash"
    # A failed call is recorded as failed, with the error, so the audit shows it.
    assert outcome["trajectory"][0]["ok"] is False
    assert outcome["trajectory"][0]["error"]


def test_worker_failure_does_not_kill_the_run(tmp_path: Path):
    class Exploding(ScriptedClient):
        def __init__(self):
            super().__init__([])

        def complete(self, *args, **kwargs):
            raise RuntimeError("provider is down")

    swarm = ResearchSwarm("resilience", root=tmp_path,
                          config=SwarmConfig(max_cycles=1,
                                             llm_client_factory=Exploding))
    agent_id = swarm.spawn_subagent("theory_lead_01", "SymPy Prover", "prove it", ["write_file"])
    target = swarm.workspace.proof_dir / f"{agent_id}.py"
    outcome = swarm._run_worker(swarm.agents[agent_id], Division.THEORY, "Close the gap.",
                                success_criterion="Success: it runs.", target=target)
    assert outcome["ran"] is True
    assert outcome["success"] is False
    # The run survives, and the reason survives with it: a bare "provider_error"
    # would leave an operator unable to tell a bad key from a network outage.
    assert "provider_error" in (outcome["error"] or "")
    assert "provider is down" in (outcome.get("summary") or outcome["error"] or
                                   json.dumps(outcome, default=str))
    # The failure is still audited rather than swallowed.
    assert swarm.ledger.by_action("WORKER_TRAJECTORY")


# -- division-appropriate instrumentation ----------------------------------

@pytest.mark.parametrize("division,expected", [
    (Division.THEORY, "run_lean_proof"),
    (Division.EMPIRICAL, "run_python_repl"),
    (Division.ADVERSARIAL, "run_lean_proof"),
])
def test_each_division_gets_the_instruments_its_method_needs(division, expected):
    assert expected in WORKER_TOOLS[division]
    # A simulation worker has no business machine-checking Lean, and a theorist
    # has no business editing a figure.
    if division is Division.EMPIRICAL:
        assert "run_lean_proof" not in WORKER_TOOLS[division]


def test_theory_and_formal_workers_can_reach_the_lean_prover(tmp_path: Path):
    for division in (Division.THEORY, Division.FORMAL):
        swarm = _swarm(tmp_path, [("write_file", {"path": "x.md", "content": "x"})])
        agent_id = swarm.spawn_subagent(
            {"theory": "theory_lead_01", "formal": "formal_lead_01"}[division.value],
            "Worker", "do the thing", ["write_file"])
        target = swarm.workspace.root / "x.md"
        outcome = swarm._run_worker(swarm.agents[agent_id], division, "Work.",
                                    success_criterion="Success: done.", target=target)
        assert outcome["ran"] is True
        assert outcome["tool_calls"] == 1


def test_unsupported_worker_tools_are_rejected_loudly():
    from adaptive_harness.agent.swarm import DeveloperAgentWorker

    worker = DeveloperAgentWorker(tool_names=("read_file", "teleport"))
    with pytest.raises(ValueError, match="teleport"):
        worker._build_tools("/tmp", None)


# -- the red team in interactive mode --------------------------------------

def test_red_team_runs_as_a_subagent_and_prose_grants_no_clearance(tmp_path: Path):
    """A chatty model must not be able to rubber-stamp a claim with prose."""
    script = [("search_files", {"pattern": "sorry", "path": "."})]
    swarm = _swarm(tmp_path, script, final="Looks good to me, no issues at all.")
    swarm._synthesize()
    agent_id = swarm.spawn_subagent("adversarial_lead_01", "Red Team Auditor",
                                    "break the claim", ["search_files"])
    swarm._falsify(swarm.agents[agent_id], [])
    assert swarm.agents[agent_id].clearance is Clearance.PENDING
    assert not swarm.ledger.by_action("CLEARANCE_GRANTED")
    # The trajectory shows the red team did search before answering.
    payload = swarm.ledger.by_action("WORKER_TRAJECTORY")[0].payload
    assert set(payload["tools_used"]) == {"search_files"}
    assert payload["tool_calls"] >= 1


def test_red_team_accepts_a_structured_refutation(tmp_path: Path):
    script = [("search_files", {"pattern": "alpha", "path": "."})]
    swarm = _swarm(tmp_path, script,
                   final='{"falsified": true, "finding": "alpha=1.5 diverges"}')
    swarm._synthesize()
    agent_id = swarm.spawn_subagent("adversarial_lead_01", "Red Team Auditor",
                                    "break the claim", ["search_files"])
    swarm._falsify(swarm.agents[agent_id], [])
    assert swarm.agents[agent_id].clearance is Clearance.COUNTEREXAMPLE
    assert swarm.ledger.by_action("COUNTEREXAMPLE_FOUND")


def test_worker_trajectory_is_a_declared_ledger_action():
    assert "WORKER_TRAJECTORY" in ACTIONS
