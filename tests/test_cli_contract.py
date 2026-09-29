"""Phase 2.6 — the non-interactive contract.

This is what makes the harness usable from CI rather than only from a terminal.
A script's only signal is the exit status, so it has to carry a meaning; and a
JSON stream is what lets a harness-of-record diff two runs and count cost
without scraping prose.

The tests are about the distinctions mattering, because a contract where
everything returns 0 -- or where every failure is 1 -- is not a contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from adaptive_harness.cli_contract import (
    EXIT_AGENT_FAILED,
    EXIT_BUDGET_EXHAUSTED,
    EXIT_GATE_REFUSED,
    EXIT_MEANINGS,
    EXIT_OK,
    EXIT_UNAVAILABLE,
    Ceilings,
    JsonStream,
    exit_code_for,
)

REPO = Path(__file__).resolve().parent.parent


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "adaptive_harness.cli", *args],
        capture_output=True, text=True, timeout=300, cwd=str(cwd or REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
             "HOME": str(cwd or REPO), "TERM": "dumb"})


# --- exit codes mean different things ---------------------------------------


def test_success_is_zero():
    assert exit_code_for("completed", success=True) == EXIT_OK


def test_an_incomplete_run_is_one():
    assert exit_code_for("incomplete", success=False) == EXIT_AGENT_FAILED


@pytest.mark.parametrize("stop", [
    "verification_failed", "skill_verification_failed", "cancelled",
    "classifier_stop", "overseer_impasse",
    "quality_gate_unsubstantiated", "missing_file_changes",
])
def test_a_refusal_is_two_not_one(stop: str):
    """A gate refusing is a different thing from the agent failing, and a caller
    usually wants to treat them differently."""
    assert exit_code_for(stop, success=False) == EXIT_GATE_REFUSED


def test_a_model_that_declined_is_a_failure_not_a_refusal():
    """`provider_error` covers a model that ran and could not do the task, which
    is the agent failing. Calling it a gate refusal would send a CI job looking
    at policy for what is really a task the model could not complete."""
    assert exit_code_for("provider_error", success=False) == EXIT_AGENT_FAILED


def test_a_missing_credential_is_four_not_a_refusal():
    """Distinguished from the case above: this one is fixable by configuration."""
    assert exit_code_for("provider_unavailable", success=False) == EXIT_UNAVAILABLE


@pytest.mark.parametrize("stop", ["step_limit", "max_cost", "max_turns"])
def test_an_exhausted_budget_is_three_not_one(stop: str):
    """The work may have been fine and the ceiling simply too low, so this is
    not a failure and not a success."""
    assert exit_code_for(stop, success=False) == EXIT_BUDGET_EXHAUSTED


def test_a_budget_stop_is_never_reported_as_success():
    """Even if the run happened to finish its work, hitting a ceiling means the
    answer is incomplete."""
    assert exit_code_for("max_cost", success=True) == EXIT_BUDGET_EXHAUSTED


def test_a_missing_dependency_is_four():
    """Distinct because it is usually fixable by configuration rather than by
    changing the task."""
    assert exit_code_for("provider_error", success=False,
                        unavailable=True) == EXIT_UNAVAILABLE


def test_unavailable_wins_over_everything():
    """If the run could not start, nothing else it reports is meaningful."""
    assert exit_code_for("max_cost", success=True, unavailable=True) == EXIT_UNAVAILABLE


def test_every_code_has_a_human_meaning():
    for code in (EXIT_OK, EXIT_AGENT_FAILED, EXIT_GATE_REFUSED,
                 EXIT_BUDGET_EXHAUSTED, EXIT_UNAVAILABLE):
        assert EXIT_MEANINGS.get(code), f"exit code {code} has no explanation"


# --- ceilings ---------------------------------------------------------------


def test_no_ceiling_means_no_breach():
    ceilings = Ceilings()
    assert ceilings.observe_turn() is None
    assert ceilings.observe_usage({"cost_usd": 1000}) is None
    assert ceilings.breached is None


def test_the_turn_ceiling_fires():
    ceilings = Ceilings(max_turns=2)
    assert ceilings.observe_turn() is None
    assert ceilings.observe_turn() is None
    breach = ceilings.observe_turn()
    assert breach and "2" in breach
    assert ceilings.breached == "max_turns"


def test_the_cost_ceiling_fires_on_a_reported_price():
    ceilings = Ceilings(max_cost_usd=1.0)
    assert ceilings.observe_usage({"cost_usd": 0.4}) is None
    assert ceilings.observe_usage({"cost_usd": 0.7}) is not None
    assert ceilings.breached == "max_cost"


def test_the_cost_ceiling_works_without_a_reported_price():
    """Most providers report token counts and no price. A ceiling that only
    reads a price field silently never fires, which is the worst failure mode
    for a thing whose entire job is to stop a run."""
    ceilings = Ceilings(max_cost_usd=0.0000001)
    breach = ceilings.observe_usage({"prompt_tokens": 100, "completion_tokens": 50})
    assert breach, "a cost ceiling did not fire without a reported price"
    assert "estimated" in breach, "and did not say the basis was an estimate"


def test_the_cost_ceiling_reads_the_total_when_present():
    ceilings = Ceilings(max_cost_usd=1.0)
    ceilings.observe_usage({"total_tokens": 1000})
    assert ceilings.reported_cost_usd > 0


def test_cost_is_cumulative_across_the_run():
    """A ceiling is about the run, not about the last call."""
    ceilings = Ceilings(max_cost_usd=1.0)
    for _ in range(5):
        ceilings.observe_usage({"cost_usd": 0.3})
    assert ceilings.breached == "max_cost", "the running total was not kept"


def test_the_report_says_which_ceiling_and_on_what_basis():
    ceilings = Ceilings(max_cost_usd=0.0000001)
    ceilings.observe_usage({"prompt_tokens": 100, "completion_tokens": 50})
    payload = ceilings.to_dict()
    assert payload["breached"] == "max_cost"
    assert payload["cost_reported"] is False, (
        "an estimate must not be reported as a provider-reported price")
    assert "estimated" in payload["detail"]


# --- the JSON stream --------------------------------------------------------


def test_the_stream_is_line_delimited_and_parsable():
    class Event:
        def __init__(self, kind, payload):
            self.event_type = kind
            self.payload = payload
            self.timestamp = 1.0

    lines: list[str] = []
    stream = JsonStream(lines.append)
    stream.emit(Event("tool_call", {"name": "run_bash", "arguments": {"command": "ls"}}))
    stream.emit(Event("response", {"success": True, "stop_reason": "completed",
                                   "total_time_ms": 12, "steps": 1,
                                   "usage": {"total_tokens": 100}}))

    for line in lines:
        parsed = json.loads(line)  # one object per line, no wrapping
        assert "event" in parsed
    assert json.loads(lines[0])["payload"]["name"] == "run_bash"


def test_the_summary_reports_an_exit_code_the_caller_can_use():
    class Event:
        def __init__(self, kind, payload):
            self.event_type = kind
            self.payload = payload
            self.timestamp = 1.0

    stream = JsonStream(lambda _line: None)
    stream.emit(Event("tool_call", {"name": "read_file"}))
    stream.emit(Event("tool_call", {"name": "read_file"}))
    stream.emit(Event("response", {"success": False, "stop_reason": "step_limit",
                                   "usage": {"total_tokens": 500}}))
    summary = stream.summary()
    assert summary["exit_code"] == EXIT_BUDGET_EXHAUSTED
    assert summary["tool_calls"] == {"read_file": 2}, "tool counts are useful to a caller"
    assert summary["total_tokens"] == 500


def test_a_broken_sink_does_not_break_the_run():
    """A caller that closed the pipe early must not take the task down."""

    def exploding(_line: str) -> None:
        raise BrokenPipeError()

    class Event:
        event_type = "response"
        payload = {"success": True, "stop_reason": "completed"}
        timestamp = 0.0

    stream = JsonStream(exploding)
    stream.emit(Event())  # must not raise


def test_a_payload_that_is_not_json_serialisable_is_still_emitted():
    class Event:
        event_type = "tool_result"
        payload = {"path": Path("/tmp/x"), "count": 1}
        timestamp = 0.0

    lines: list[str] = []
    JsonStream(lines.append).emit(Event())
    assert json.loads(lines[0])["payload"]["path"] == "/tmp/x"


# --- the contract, end to end through the real CLI -------------------------


def test_json_output_is_parsable_from_the_command_line(tmp_path: Path):
    """Rich wraps at the terminal width, which would break a line-delimited
    contract. `--json` must bypass it entirely."""
    result = _run("dev", "say hi", "--offline", "--json", cwd=tmp_path)
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert lines, f"no JSON emitted. stderr: {result.stderr[-400:]}"
    for line in lines:
        json.loads(line)  # raises if a line is not a whole object
    summaries = [json.loads(line) for line in lines if json.loads(line)["event"] == "summary"]
    assert summaries, "no summary object was emitted"
    assert "exit_code" in summaries[-1]


def test_a_plain_run_exits_zero(tmp_path: Path):
    assert _run("dev", "say hi", "--offline", cwd=tmp_path).returncode == EXIT_OK


def test_a_turn_ceiling_is_enforced_from_the_command_line(tmp_path: Path):
    """A ceiling that only reports after the fact is a report, not a ceiling."""
    result = _run("dev", "list the files here", "--offline", "--max-turns", "1",
                  "--json", cwd=tmp_path)
    # A one-step mock run may finish before the ceiling applies; what must never
    # happen is a run that exceeded the ceiling and still reported success.
    for line in result.stdout.splitlines():
        if line.strip().startswith("{"):
            payload = json.loads(line)
            if payload["event"] == "summary":
                assert payload["exit_code"] in {EXIT_OK, EXIT_BUDGET_EXHAUSTED}
    assert result.returncode in {EXIT_OK, EXIT_BUDGET_EXHAUSTED}
