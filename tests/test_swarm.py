"""Swarm orchestration must serialize writes and pass bounded artifacts."""

from __future__ import annotations

import json
from threading import Lock
import time

import pytest

from adaptive_harness.agent.swarm import (
    DeveloperAgentWorker, SwarmCoordinator, SwarmPhase, SwarmResult, SwarmRole,
)
from adaptive_harness.llm.mock_client import LLMResponse, ToolCall


def test_swarm_three_stage_handoff_has_single_writer(tmp_path):
    lock = Lock()
    active = 0
    peak = 0
    calls = []
    statuses = []

    def worker(assignment):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((assignment.key, assignment.may_edit, dict(assignment.context)))
        time.sleep(0.03)
        with lock:
            active -= 1
        return SwarmResult(assignment.role, assignment.phase, True,
                           f"{assignment.key} complete", {"file": "module.py"},
                           verified=assignment.phase is SwarmPhase.VERIFY)

    coordinator = SwarmCoordinator(worker, on_status=lambda status: statuses.append(status))
    report = coordinator.run("Fix module", tmp_path)

    assert report.success
    assert peak == 1
    assert [key for key, _, _ in calls if key == "coder:implement"] == ["coder:implement"]
    assert all(may_edit == (key == "coder:implement") for key, may_edit, _ in calls)
    by_key = {key: context for key, _, context in calls}
    assert set(by_key["coder:implement"]) == {"architect:plan"}
    assert set(by_key["qa:verify"]) == {"architect:plan", "coder:implement"}
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
    assert set(calls) == {"architect:plan", "coder:implement", "qa:verify"}
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

    class Client:
        default_model = "z-ai/glm-5.3-flash"

        def complete(self, **kwargs):
            names = {item["function"]["name"] for item in kwargs["tools"]}
            schemas.append(names)
            if kwargs["messages"][-1]["role"] == "tool":
                return LLMResponse(content="Done and verified.")
            if "write_file" in names:
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
