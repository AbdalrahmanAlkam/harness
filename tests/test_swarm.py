"""Swarm orchestration must serialize writes and pass bounded artifacts."""

from __future__ import annotations

import json
from threading import Lock
import time

import pytest

from adaptive_harness.agent.swarm import (
    SwarmCoordinator, SwarmPhase, SwarmResult, SwarmRole,
)


def test_swarm_parallel_readers_single_writer_and_structured_handoff(tmp_path):
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
    assert peak == 2
    assert [key for key, _, _ in calls if key == "coder:implement"] == ["coder:implement"]
    assert all(may_edit == (key == "coder:implement") for key, may_edit, _ in calls)
    by_key = {key: context for key, _, context in calls}
    assert set(by_key["coder:implement"]) == {"architect:plan", "qa:plan"}
    assert set(by_key["security:verify"]) == {"architect:plan", "qa:plan", "coder:implement"}
    assert all(value == "done" for value in report.status.values())
    assert any(state["architect:plan"] == "running" and
               state["qa:plan"] == "running" for state in statuses)
    assert json.loads(report.to_json())["success"] is True


def test_swarm_runs_in_direct_workspace_and_stops_on_plan_failure(tmp_path):
    calls = []

    def worker(assignment):
        calls.append(assignment.key)
        success = assignment.role is not SwarmRole.ARCHITECT
        return SwarmResult(assignment.role, assignment.phase, success, "done")

    coordinator = SwarmCoordinator(worker)
    report = coordinator.run("Fix", tmp_path)
    assert not report.success
    assert set(calls) == {"architect:plan", "qa:plan"}
    assert report.status["coder:implement"] == "queued"


def test_swarm_catches_worker_errors_and_requires_qa_evidence(tmp_path):
    def worker(assignment):
        if assignment.role is SwarmRole.SECURITY:
            raise RuntimeError("auditor unavailable")
        return SwarmResult(assignment.role, assignment.phase, True, "done")

    report = SwarmCoordinator(worker).run("Fix", tmp_path, isolated=True)
    assert not report.success
    assert report.status["security:verify"] == "failed"
    assert "auditor unavailable" in report.results[-1].error

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
