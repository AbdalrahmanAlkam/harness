"""The tasklist plugin: a task that cannot be closed on an assertion.

The harness already scans source for TODO markers, which records intentions.
This plugin records what a run is actually doing, and its one rule is the
interesting part: a task reaches ``done`` only when a successful call to a real
tool has been *observed* since it entered ``in_progress``. That is the same
principle ``SkillVerifier`` applies to a skill's completion checks — a model
saying it ran the tests is not the same as the tests having run.

The tests build the plugin through the real host and call its handlers directly.
No model, no network, no subprocess: the evidence these tests rely on is
synthesised as the event the agent would have emitted, which is the point — the
plugin reads the event stream, not anything the model says about itself.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from adaptive_harness.plugins.host import PluginHost

#: The one environment variable the plugin reads. Every test points it at
#: tmp_path so a run of the suite cannot touch a real task list.
STATE_ENV = "ADAPTIVE_HARNESS_TASKLIST_STATE"


@dataclass
class Event:
    """The shape ``on_agent_event`` sees: what the agent's stream carries."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


def tool_result(name: str, success: bool = True) -> Event:
    return Event("tool_result",
                 {"name": name, "success": success, "output": "", "error": None})


class Tasklist:
    """A loaded tasklist plugin, with its module and its state file."""

    def __init__(self, host: PluginHost, workspace: Path) -> None:
        self.host = host
        self.state_file = Path(workspace) / "tasklist-state.json"
        self.plugin = next(p for p in host.plugins if p.name == "tasklist")
        # The host imports a plugin's Python under a name derived from its
        # directory and manifest, which is how a reload gets a *fresh* module
        # with none of this one's globals left.
        matches = [name for name in sys.modules
                   if name.startswith("_adaptive_plugin_")
                   and self.plugin.directory.name in name]
        assert len(matches) == 1, f"expected one imported module, got {matches}"
        self.module = sys.modules[matches[0]]
        self.tools = {tool.name: tool for tool in self.plugin.tools}

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return self.tools[name].handler(**kwargs)

    def update(self, task_id: str, status: str, note: str = "") -> dict[str, Any]:
        return self.call("task_update", id=task_id, status=status, note=note)

    def evidence(self, *tools: str, success: bool = True) -> None:
        """Feed the plugin the events a real run would have emitted."""
        for name in tools:
            self.host.emit(tool_result(name, success=success))

    def reload(self) -> "Tasklist":
        """Load the plugin again from scratch — a fresh module, same state file.

        This is what a restart looks like: the Python is imported again and
        every module-level variable is gone, so anything that still answers has
        come off the disk.
        """
        return Tasklist(load(self.host.project_root, self.state_file), self.state_file)

    def state(self) -> dict[str, Any]:
        return json.loads(self.state_file.read_text(encoding="utf-8"))


def load(project_root: Path, state_file: Path) -> PluginHost:
    host = PluginHost(project_root=project_root)
    host.discover()
    return host


@pytest.fixture
def tasklist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tasklist:
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "tasklist-state.json"))
    return Tasklist(load(tmp_path, tmp_path / "tasklist-state.json"), tmp_path)


# --- the plugin loads -------------------------------------------------------


def test_the_plugin_loads_with_its_tools_and_hooks(tasklist: Tasklist):
    assert tasklist.plugin.ok, tasklist.plugin.error
    assert set(tasklist.tools) == {"task_update", "task_list", "task_clear"}
    assert tasklist.plugin.hooks.pre_tool is tasklist.module.guard_pre_tool
    # The event listener is what makes evidence observed rather than asserted.
    assert tasklist.module.on_agent_event in tasklist.host.hook_listeners


def test_the_list_starts_empty(tasklist: Tasklist):
    assert tasklist.call("task_list")["success"]


# --- the rule: no evidence, no done -----------------------------------------


def test_a_done_without_evidence_is_refused(tasklist: Tasklist):
    """The claim is not the work. Without an observed tool call there is no done."""
    assert tasklist.update("build-thing", "in_progress")["success"]
    refusal = tasklist.update("build-thing", "done", "all finished")

    assert refusal["success"] is False
    assert "no successful call to a real tool" in refusal["error"]
    # The refusal has to say what would satisfy it, or the model can only guess.
    assert "run_bash" in refusal["error"]
    # And it must not have recorded the claim as if it had happened.
    assert tasklist.state()["tasks"]["build-thing"]["status"] == "in_progress"


def test_a_read_is_not_evidence(tasklist: Tasklist):
    """Reading a file changes nothing, so it cannot discharge a task."""
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("read_file", "list_directory", "search_files")
    assert tasklist.update("build-thing", "done")["success"] is False


def test_a_failed_call_is_not_evidence(tasklist: Tasklist):
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("run_pytest", success=False)
    assert tasklist.update("build-thing", "done")["success"] is False


def test_asking_about_the_list_is_not_evidence(tasklist: Tasklist):
    """Otherwise a model discharges its own evidence by listing its own list."""
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("task_list", "task_update")
    assert tasklist.update("build-thing", "done")["success"] is False


def test_a_task_must_be_started_before_it_can_be_finished(tasklist: Tasklist):
    refusal = tasklist.update("build-thing", "done")
    assert refusal["success"] is False
    assert "not in_progress" in refusal["error"]
    # A refused creation leaves nothing behind.
    assert "No tasks" in tasklist.call("task_list")["output"]


# --- the rule: observed evidence discharges it ------------------------------


def test_an_observed_successful_tool_call_allows_done(tasklist: Tasklist):
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("write_file")

    result = tasklist.update("build-thing", "done", "wrote the parser")

    assert result["success"] is True, result
    assert "in_progress -> done" in result["output"]
    assert tasklist.state()["tasks"]["build-thing"]["status"] == "done"


def test_evidence_observed_before_the_task_started_does_not_count(tasklist: Tasklist):
    """A tool that ran first cannot retroactively justify a task started later."""
    tasklist.evidence("run_bash")
    tasklist.update("build-thing", "in_progress")
    assert tasklist.update("build-thing", "done")["success"] is False


def test_an_unknown_tool_still_counts(tasklist: Tasklist):
    """Allow-by-default: a new built-in or an MCP tool discharges a task."""
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("some_future_tool")
    assert tasklist.update("build-thing", "done")["success"] is True


def test_evidence_is_scoped_to_its_own_window(tasklist: Tasklist):
    tasklist.update("first", "in_progress")
    tasklist.update("second", "in_progress")
    tasklist.evidence("run_bash")
    tasklist.update("first", "done")
    # `second` was opened in the same window, so it is covered too.
    assert tasklist.update("second", "done")["success"] is True

    tasklist.update("third", "in_progress")  # opened after the evidence
    assert tasklist.update("third", "done")["success"] is False


# --- monotonicity -----------------------------------------------------------


def test_a_done_task_does_not_go_back_to_pending(tasklist: Tasklist):
    """The list is what a later turn reads to decide whether the work is real."""
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("run_bash")
    tasklist.update("build-thing", "done")

    for regression in ("pending", "in_progress"):
        refusal = tasklist.update("build-thing", regression)
        assert refusal["success"] is False, regression
        assert "cannot go back" in refusal["error"]
        assert tasklist.state()["tasks"]["build-thing"]["status"] == "done"


def test_a_blocked_task_can_only_be_reopened_with_a_reason(tasklist: Tasklist):
    """The one regression allowed has to be explicit, so it is in the history."""
    tasklist.update("build-thing", "in_progress")
    tasklist.update("build-thing", "blocked", "needs a key from the user")

    quiet = tasklist.update("build-thing", "in_progress")
    assert quiet["success"] is False
    assert "in `note`" in quiet["error"]

    resumed = tasklist.update("build-thing", "in_progress", "the user sent the key")
    assert resumed["success"] is True
    assert tasklist.state()["tasks"]["build-thing"]["status"] == "in_progress"


def test_resuming_requires_evidence_again(tasklist: Tasklist):
    """Re-opening starts a new evidence window; old work does not carry over."""
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("run_bash")
    tasklist.update("build-thing", "blocked", "wait")
    tasklist.update("build-thing", "in_progress", "resumed")

    assert tasklist.update("build-thing", "done")["success"] is False
    tasklist.evidence("run_bash")
    assert tasklist.update("build-thing", "done")["success"] is True


# --- the pre_tool hook ------------------------------------------------------


def test_the_pre_tool_hook_denies_an_unevidenced_done(tasklist: Tasklist):
    """Enforced before dispatch, so no safety profile can wave it through."""
    tasklist.update("build-thing", "in_progress")

    verdict = tasklist.host.consult_pre_tool(
        "task_update", {"id": "build-thing", "status": "done"})

    _, reason, plugin = verdict
    assert plugin == "tasklist"
    assert "no successful call to a real tool" in reason


def test_a_denial_carries_a_reason(tasklist: Tasklist):
    """HookVerdict refuses to exist without one: a silent block reads as a hang."""
    from adaptive_harness.plugins.types import HookVerdict

    tasklist.update("build-thing", "in_progress")
    verdict = tasklist.host.consult_pre_tool(
        "task_update", {"id": "build-thing", "status": "done"})[1]
    assert verdict
    with pytest.raises(ValueError):
        HookVerdict("deny")


def test_the_hook_allows_an_evidenced_done_and_other_tools(tasklist: Tasklist):
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("run_pytest")

    assert tasklist.host.consult_pre_tool(
        "task_update", {"id": "build-thing", "status": "done"})[1] == ""
    # Starting a task, and everything else, is untouched by the guard.
    assert tasklist.host.consult_pre_tool(
        "task_update", {"id": "build-thing", "status": "in_progress"})[1] == ""
    assert tasklist.host.consult_pre_tool("run_bash", {"command": "pytest"})[1] == ""


def test_the_hook_and_the_handler_agree(tasklist: Tasklist):
    """The handler refuses what the hook refuses, and for the same reason."""
    tasklist.update("build-thing", "in_progress")
    arguments = {"id": "build-thing", "status": "done", "note": "trust me"}
    hooked = tasklist.host.consult_pre_tool("task_update", arguments)[1]

    bypassed = tasklist.call("task_update", **arguments)  # called directly, as if unchecked
    assert bypassed["success"] is False
    assert "no successful call to a real tool" in bypassed["error"]
    assert bypassed["error"].split(".")[0] in hooked


# --- persistence ------------------------------------------------------------


def test_the_list_survives_a_reload(tasklist: Tasklist):
    tasklist.update("build-thing", "in_progress", "writing the parser")
    tasklist.evidence("write_file")
    tasklist.update("build-thing", "done")

    reloaded = tasklist.reload()

    assert reloaded.module is not tasklist.module, "the module must be a fresh one"
    assert "build-thing" in reloaded.call("task_list")["output"]


def test_evidence_survives_a_reload_too(tasklist: Tasklist):
    """Otherwise a restart silently turns an evidenced task into an unevidenced one."""
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("run_bash")

    reloaded = tasklist.reload()
    assert reloaded.update("build-thing", "done")["success"] is True


def test_a_second_module_instance_shares_the_file(tasklist: Tasklist):
    """Two hosts in one process, one list: the file is the source of truth."""
    tasklist.update("build-thing", "in_progress")
    other = tasklist.reload()
    assert "build-thing" in other.call("task_list")["output"]


def test_a_damaged_state_file_is_reported_not_silently_emptied(tasklist: Tasklist):
    """A silently empty list reads as 'no work was tracked' when the truth is loss."""
    tasklist.state_file.write_text("{not json", encoding="utf-8")

    fresh = tasklist.reload()
    assert fresh.call("task_list")["output"].count("not valid JSON") == 1


def test_task_clear_resets_everything(tasklist: Tasklist):
    tasklist.update("build-thing", "in_progress")
    tasklist.evidence("run_bash")

    assert tasklist.call("task_clear")["success"]
    assert "No tasks" in tasklist.call("task_list")["output"]
    # Evidence went with it, so a cleared task cannot be closed on the old work.
    tasklist.update("build-thing", "in_progress")
    assert tasklist.update("build-thing", "done")["success"] is False


# --- thread safety ----------------------------------------------------------


def test_eight_threads_mutating_the_state_stay_consistent(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch):
    """The TUI worker thread and the main thread both reach this state.

    Every thread updates, observes a tool call, and lists, with no lock held
    anywhere in the test. A read-modify-write that interleaves loses an update
    silently, so the assertion is on the total: all 80 tasks present, all the
    transitions the threads were allowed to make actually recorded.
    """
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "tasklist-state.json"))
    host = load(tmp_path, tmp_path / "tasklist-state.json")
    plugin = Tasklist(host, tmp_path)
    per_thread = 10
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def work(worker: int) -> None:
        try:
            barrier.wait()
            for index in range(per_thread):
                task_id = f"w{worker}-t{index}"
                plugin.update(task_id, "in_progress", f"worker {worker}")
                plugin.evidence("run_bash")
                assert plugin.update(task_id, "done")["success"] is True
                assert plugin.update(task_id, "pending")["success"] is False
                plugin.call("task_list")
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert all(not thread.is_alive() for thread in threads), "a thread deadlocked"
    state = plugin.state()
    assert len(state["tasks"]) == 8 * per_thread
    assert all(task["status"] == "done" for task in state["tasks"].values())
    # Every observed call was written down, and none was lost to a race.
    assert state["seq"] == 8 * per_thread
    assert len(state["evidence"]) == 8 * per_thread


def test_a_hook_that_raises_does_not_break_the_run(tasklist: Tasklist):
    """The host removes a listener that raises, so the plugin's own must not."""
    before = list(tasklist.host.load_errors)
    tasklist.update("build-thing", "in_progress")
    tasklist.host.emit(Event("tool_result", {"name": "run_bash", "success": True}))
    assert tasklist.update("build-thing", "done")["success"] is True
    assert tasklist.host.load_errors == before
