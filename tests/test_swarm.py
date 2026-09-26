"""Swarm orchestration must serialize writes and pass bounded artifacts."""

from __future__ import annotations

import json
from threading import Lock
import time

import pytest

from adaptive_harness.agent.swarm import (
    DeveloperAgentWorker, SwarmAssignment, SwarmCoordinator, SwarmPhase, SwarmResult,
    SwarmRole, is_role_refusal, run_assignment,
)
from adaptive_harness.llm.mock_client import LLMResponse, ToolCall


def test_swarm_three_stage_handoff_has_single_writer(tmp_path):
    lock = Lock()
    active = 0
    peak = 0
    active_edit = 0
    peak_edit = 0
    calls = []
    statuses = []

    def worker(assignment):
        nonlocal active, peak, active_edit, peak_edit
        with lock:
            active += 1
            peak = max(peak, active)
            if assignment.may_edit:
                active_edit += 1
                peak_edit = max(peak_edit, active_edit)
            calls.append((assignment.key, assignment.may_edit, dict(assignment.context)))
        time.sleep(0.03)
        with lock:
            active -= 1
            if assignment.may_edit:
                active_edit -= 1
        return SwarmResult(assignment.role, assignment.phase, True,
                           f"{assignment.key} complete", {"file": "module.py"},
                           verified=assignment.phase is SwarmPhase.VERIFY)

    coordinator = SwarmCoordinator(worker, on_status=lambda status: statuses.append(status))
    report = coordinator.run("Fix module", tmp_path)

    assert report.success
    assert peak == 2  # read-only reviews run in parallel
    assert peak_edit == 1  # only the coder ever writes, serialized
    assert [key for key, _, _ in calls if key == "coder:implement"] == ["coder:implement"]
    assert all(may_edit == (key == "coder:implement") for key, may_edit, _ in calls)
    by_key = {key: context for key, _, context in calls}
    assert set(by_key["coder:implement"]) == {"architect:plan"}
    assert set(by_key["qa:verify"]) == {"architect:plan", "coder:implement"}
    assert set(by_key["security:verify"]) == {"architect:plan", "coder:implement"}
    assert set(report.status) == {"architect:plan", "coder:implement", "qa:verify", "security:verify"}
    assert all(value == "done" for value in report.status.values())
    assert any(state["architect:plan"] == "running" and
               state["coder:implement"] == "queued" for state in statuses)
    assert json.loads(report.to_json())["success"] is True


def test_swarm_runs_coder_and_review_after_plan_failure(tmp_path):
    calls = []

    def worker(assignment):
        calls.append(assignment.key)
        success = assignment.role is not SwarmRole.ARCHITECT
        return SwarmResult(assignment.role, assignment.phase, success, "done",
                           verified=assignment.phase is SwarmPhase.VERIFY)

    coordinator = SwarmCoordinator(worker)
    report = coordinator.run("Fix", tmp_path)
    assert report.success
    assert set(calls) == {"architect:plan", "coder:implement", "qa:verify", "security:verify"}
    assert report.status["coder:implement"] == "done"


def test_swarm_catches_worker_errors_and_requires_qa_evidence(tmp_path):
    def worker(assignment):
        if assignment.phase is SwarmPhase.VERIFY:
            raise RuntimeError("reviewer unavailable")
        return SwarmResult(assignment.role, assignment.phase, True, "done")

    report = SwarmCoordinator(worker).run("Fix", tmp_path, isolated=True)
    assert not report.success
    assert report.status["qa:verify"] == "failed"
    assert "reviewer unavailable" in report.results[-1].error

    def unverified_worker(assignment):
        return SwarmResult(assignment.role, assignment.phase, True, "done")

    report = SwarmCoordinator(unverified_worker).run("Fix", tmp_path, isolated=True)
    assert not report.success
    assert report.status["qa:verify"] == "done"


def test_swarm_retries_transient_subagent_exception(tmp_path):
    attempts = {}

    def worker(assignment):
        attempts[assignment.key] = attempts.get(assignment.key, 0) + 1
        if assignment.key == "architect:plan" and attempts[assignment.key] == 1:
            raise RuntimeError("temporary provider error")
        return SwarmResult(assignment.role, assignment.phase, True, "done",
                           verified=assignment.phase is SwarmPhase.VERIFY)

    report = SwarmCoordinator(worker).run("Fix module", tmp_path)
    assert report.success
    assert attempts["architect:plan"] == 2


def test_swarm_rejects_mismatched_and_unserializable_worker_results(tmp_path):
    def wrong_worker(assignment):
        return SwarmResult(SwarmRole.CODER, SwarmPhase.IMPLEMENT, True, "wrong")

    report = SwarmCoordinator(wrong_worker).run("Fix", tmp_path, isolated=True)
    assert not report.success
    assert "different assignment" in report.results[0].error

    def unserializable_worker(assignment):
        return SwarmResult(assignment.role, assignment.phase, True, "bad", {"object": object()})

    report = SwarmCoordinator(unserializable_worker).run("Fix", tmp_path, isolated=True)
    assert not report.success
    assert "not JSON serializable" in report.results[0].error


def test_swarm_worker_coder_gets_all_core_tools_and_writes_real_file(tmp_path):
    schemas = []
    coder_prompts = []

    class Client:
        default_model = "z-ai/glm-5.3-flash"

        def complete(self, **kwargs):
            names = {item["function"]["name"] for item in kwargs["tools"]}
            schemas.append(names)
            if kwargs["messages"][-1]["role"] == "tool":
                return LLMResponse(content="Done and verified.")
            if "security reviewer" in str(kwargs["messages"][0].get("content", "")).lower():
                return LLMResponse(tool_calls=[ToolCall(id="sec", name="read_file",
                                                        arguments={"path": "gps_app/index.html"})])
            if "write_file" in names:
                coder_prompts.append(kwargs["messages"][0]["content"])
                return LLMResponse(tool_calls=[
                    ToolCall(id="html", name="write_file", arguments={
                        "path": "gps_app/index.html", "content": "<!doctype html><title>GPS</title><link rel='stylesheet' href='style.css'><script src='main.js'></script>"}),
                    ToolCall(id="css", name="write_file", arguments={
                        "path": "gps_app/style.css", "content": "body { background: #111; color: white; }"}),
                    ToolCall(id="js", name="write_file", arguments={
                        "path": "gps_app/main.js", "content": "console.log('GPS simulation');"}),
                ])
            if "run_bash" in names and "write_file" not in names:
                return LLMResponse(tool_calls=[ToolCall(id="read", name="read_file", arguments={
                    "path": "gps_app/index.html"})])
            return LLMResponse(content="I cannot do what you're asking; no write_file is available in this planning role.")

    report = SwarmCoordinator(DeveloperAgentWorker(llm_client_factory=Client)).run(
        "create a GPS app using multiple agents", tmp_path)
    assert report.success
    assert not report.results[0].success
    assert report.results[1].success
    assert (tmp_path / "gps_app/index.html").read_text().startswith("<!doctype html>")
    assert (tmp_path / "gps_app/style.css").is_file()
    assert (tmp_path / "gps_app/main.js").is_file()
    assert any({"write_file", "edit_file", "read_file", "list_directory", "search_files",
                "run_bash", "run_pytest"}.issubset(names) for names in schemas)
    assert any("write-enabled Coder subagent" in prompt for prompt in coder_prompts)


# --- Remediation regressions -----------------------------------------------

def test_benign_role_language_is_not_a_refusal():
    # Role limitations are not refusals: a read-only architect must be able to
    # say it cannot edit without being marked failed.
    assert not is_role_refusal(
        "Plan: 1) create parser.py. I cannot edit files in this role, so the coder must apply the changes.", False)
    assert not is_role_refusal("I was unable to run the integration suite locally.", False)
    assert not is_role_refusal("I can't verify the migration without a live database; run the tests in CI.", False)
    # Actual refusals: explicit decline phrasing with no completed tool work.
    assert is_role_refusal("I cannot do what you're asking; no write_file is available in this planning role.", False)
    assert is_role_refusal("I refuse to perform this task.", False)
    # ...but not when the subagent actually did the work first.
    assert not is_role_refusal("I cannot do what you're asking further; the files are done.", True)


def test_retry_carries_failure_feedback(tmp_path):
    attempts = {}

    def worker(assignment):
        attempts[assignment.key] = attempts.get(assignment.key, 0) + 1
        if assignment.key == "architect:plan" and attempts[assignment.key] == 1:
            return SwarmResult(assignment.role, assignment.phase, False, "boom",
                               {"stop_reason": "step_limit"}, error="transient provider error")
        if assignment.key == "architect:plan":
            assert "prior_failure" in assignment.context  # failure feedback reached the retry
        return SwarmResult(assignment.role, assignment.phase, True, "done",
                           verified=assignment.phase is SwarmPhase.VERIFY)

    report = SwarmCoordinator(worker).run("Fix", tmp_path)
    assert report.success
    assert attempts["architect:plan"] == 2


def test_review_feedback_repairs_implementation(tmp_path):
    coder_runs = []

    def worker(assignment):
        if assignment.role is SwarmRole.CODER:
            coder_runs.append(dict(assignment.context))
            return SwarmResult(assignment.role, assignment.phase, True, "implemented")
        if assignment.role is SwarmRole.QA and len(coder_runs) == 1:
            return SwarmResult(assignment.role, assignment.phase, False,
                               "QA: missing regression test for edge case", error="missing test")
        return SwarmResult(assignment.role, assignment.phase, True, "clean",
                           verified=assignment.phase is SwarmPhase.VERIFY)

    report = SwarmCoordinator(worker, max_repair_cycles=1).run("Fix", tmp_path)
    assert report.success
    assert len(coder_runs) == 2
    assert "review_findings" in coder_runs[1]
    assert "qa:verify" in coder_runs[1]["review_findings"]


def test_deterministic_failures_are_not_retried(tmp_path):
    attempts = []

    def refusing_worker(assignment):
        attempts.append(assignment.key)
        return SwarmResult(assignment.role, assignment.phase, False, "no",
                           error="Model refused the assigned role")

    result = run_assignment(refusing_worker,
                            SwarmAssignment(SwarmRole.ARCHITECT, SwarmPhase.PLAN, "t", tmp_path))
    assert not result.success and len(attempts) == 1


def test_worker_never_reports_empty_summaries_and_forwards_events(tmp_path):
    events = []

    class SilentClient:
        default_model = "test/model"

        def complete(self, **kwargs):
            return LLMResponse(model=self.default_model, content="")

    worker = DeveloperAgentWorker(llm_client_factory=SilentClient, on_event=events.append)
    result = worker(SwarmAssignment(SwarmRole.ARCHITECT, SwarmPhase.PLAN, "plan the work", tmp_path))
    assert result.summary.startswith("(no final answer")
    assert "stop reason" in result.summary
    assert result.error and "did not complete" in result.error
    assert any(event.event_type == "response" for event in events)


def test_qa_verification_accepts_inspection_evidence_without_tests(tmp_path):
    class InspectorClient:
        default_model = "test/model"

        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(model=self.default_model, tool_calls=[ToolCall(
                    id="1", name="list_directory", arguments={"path": "."})])
            return LLMResponse(model=self.default_model,
                               content="Reviewed the layout; implementation matches the task.")

    worker = DeveloperAgentWorker(llm_client_factory=InspectorClient)
    result = worker(SwarmAssignment(SwarmRole.QA, SwarmPhase.VERIFY, "review the work", tmp_path))
    assert result.success and result.verified  # no tests/ dir, but inspection evidence exists


def test_coder_shell_created_files_count_as_mutation(tmp_path):
    class ShellWriter:
        default_model = "test/model"

        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(model=self.default_model, tool_calls=[ToolCall(
                    id="1", name="run_bash",
                    arguments={"command": "mkdir -p out && echo hi > out/new.txt"})])
            return LLMResponse(model=self.default_model, content="Created out/new.txt via shell.")

    worker = DeveloperAgentWorker(llm_client_factory=ShellWriter)
    result = worker(SwarmAssignment(SwarmRole.CODER, SwarmPhase.IMPLEMENT, "create out/new.txt", tmp_path))
    assert result.success
    assert (tmp_path / "out" / "new.txt").is_file()
    assert result.artifacts.get("stop_reason") != "missing_file_changes"


def test_delegation_tool_supports_security_role(tmp_path):
    class Client:
        default_model = "test/model"

        def complete(self, **kwargs):
            return LLMResponse(model=self.default_model, content="No security findings.")

    from adaptive_harness.tools.delegation import DelegateSubagentTool
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=Client)
    result = tool.execute(role="security", task="review the changes for security issues")
    assert result.success and result.metadata["role"] == "security"
    assert not tool.execute(role="ninja", task="x").success
    assert not tool.execute(role="coder", task="  ").success
