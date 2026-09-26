"""Regressions for autonomous edits, explicit delegation, and verified learning."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.agent.swarm import DeveloperAgentWorker, SwarmResult
from adaptive_harness.classifiers.risk_classifier import ToolRiskClassifier
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.learning.distill import export_distillation
from adaptive_harness.learning.harvester import harvest_verified_traces
from adaptive_harness.llm.mock_client import LLMResponse
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.tools.delegation import DelegateSubagentTool
from adaptive_harness.tools.bash import RunBashTool
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.widgets import ClassifierTelemetryWidget, QuickSelectModal, ThemePickerModal
import pytest


class TextOnlyClient:
    default_model = "mock"

    def __init__(self, first: str):
        self.first = first
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return LLMResponse(content=self.first if self.calls == 1 else "Created the file.")


def test_explicit_file_block_is_written_to_nested_directory(tmp_path: Path):
    client = TextOnlyClient("```python\n# path: gps_app/src/main.py\nprint('ready')\n```")
    agent = DeveloperAgent(llm_client=client, workspace_root=str(tmp_path),
                           preferences_dir=tmp_path / "prefs")
    events = list(agent.run_stream("create gps_app/src/main.py", max_steps=3))
    assert (tmp_path / "gps_app/src/main.py").read_text().strip() == "print('ready')"
    assert any(event.event_type == "tool_call" and event.payload["name"] == "write_file" for event in events)
    assert next(event.payload for event in events if event.event_type == "response")["success"]


def test_plain_code_does_not_count_as_completed_edit(tmp_path: Path):
    agent = DeveloperAgent(llm_client=TextOnlyClient("```python\nprint('not saved')\n```"),
                           workspace_root=str(tmp_path), preferences_dir=tmp_path / "prefs")
    response = next(event.payload for event in agent.run_stream("create app.py", max_steps=2)
                    if event.event_type == "response")
    assert not response["success"]
    assert response["stop_reason"] == "missing_file_changes"
    assert not (tmp_path / "app.py").exists()


def test_offline_mock_does_not_fabricate_requested_implementation(tmp_path: Path):
    agent = DeveloperAgent(llm_client=LLMClient(force_mock=True), workspace_root=str(tmp_path),
                           preferences_dir=tmp_path / "prefs")
    events = list(agent.run_stream("create hello.py with a greet function"))
    domain = next(event.payload for event in events if event.event_type == "domain_mode")
    result = next(event.payload for event in events if event.event_type == "response")
    assert domain["mode"] == "coding"
    assert not result["success"] and result["stop_reason"] == "provider_error"
    assert not (tmp_path / "hello.py").exists()
    assert not (tmp_path / "scratch.py").exists()


def test_turbo_risk_boundary_allows_local_cleanup_but_gates_catastrophe():
    risk = ToolRiskClassifier()
    assert risk.evaluate("run_bash", {"command": "rm -rf build"}, catastrophic_only=True) is None
    assert risk.evaluate("run_bash", {"command": "sudo apt install thing"}, catastrophic_only=True)
    assert risk.evaluate("run_bash", {"command": "mkfs.ext4 /dev/sda"}, catastrophic_only=True)
    assert risk.catastrophic_intent("format disk")


def test_delegate_subagent_restricts_workspace_and_returns_structured_result(tmp_path: Path, monkeypatch):
    seen = []

    def fake_worker(self, assignment):
        seen.append(assignment)
        return SwarmResult(assignment.role, assignment.phase, True, "implemented", verified=True)

    monkeypatch.setattr(DeveloperAgentWorker, "__call__", fake_worker)
    tool = DelegateSubagentTool(tmp_path)
    result = tool.execute("coder", "create the app", "new_app")
    assert result.success and seen[0].workspace_root == tmp_path / "new_app"
    assert (tmp_path / "new_app").is_dir()
    assert json.loads(result.output)["role"] == "coder"
    outside = tool.execute("coder", "escape", "../outside")
    assert not outside.success
    assert not (tmp_path.parent / "outside").exists()


def test_read_only_delegation_does_not_create_a_missing_target(tmp_path: Path, monkeypatch):
    seen = []

    def fake_worker(self, assignment):
        seen.append(assignment)
        return SwarmResult(assignment.role, assignment.phase, True, "planned")

    monkeypatch.setattr(DeveloperAgentWorker, "__call__", fake_worker)
    tool = DelegateSubagentTool(tmp_path)
    missing = tool.execute("architect", "plan the app", "not_created")
    assert not missing.success
    assert not (tmp_path / "not_created").exists()
    assert not seen
    existing = tmp_path / "existing"
    existing.mkdir()
    assert tool.execute("reviewer", "inspect the app", "existing").success
    assert seen[0].workspace_root == existing


def test_reviewer_shell_only_checks_workspace_javascript(tmp_path: Path):
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    tool = RunBashTool(tmp_path, read_only=True)
    script = tmp_path / "main.js"
    script.write_text("const answer = 42;\n")
    assert tool.execute("node --check main.js").success
    blocked = tool.execute("touch unexpected.txt")
    assert not blocked.success
    assert not (tmp_path / "unexpected.txt").exists()
    assert not tool.execute("node --check ../outside.js").success
    script.write_text("const = ;\n")
    assert not tool.execute("node --check main.js").success


def test_verified_trace_export_and_workspace_retrieval(tmp_path: Path):
    repository = ExperienceRepository(tmp_path / "experience.db")
    repository.save_agent_trace(str(tmp_path), "create alpha.py module",
        [{"tool": "write_file", "target": "alpha.py", "success": True}],
        "Created alpha.py", verified_success=True)
    repository.save_agent_trace(str(tmp_path), "create broken.py module",
        [{"tool": "write_file", "target": "broken.py", "success": False}],
        "Failed", verified_success=False)
    examples = harvest_verified_traces(repository.db_path)
    assert len(examples) == 1 and examples[0]["user_prompt"] == "create alpha.py module"
    exemplar = repository.lookup_verified_exemplar(str(tmp_path), "create alpha.py module with tests")
    assert exemplar and exemplar["trajectory_steps"][0]["tool"] == "write_file"
    assert repository.lookup_verified_exemplar(str(tmp_path / "other"), "create alpha.py module with tests") is None
    output = tmp_path / "distillation"
    assert export_distillation(repository.db_path, output, "qwen2.5-0.5b") == 1
    lines = (output / "instruction_dataset.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(json.loads(lines[0])["output"])["final_solution"] == "Created alpha.py"
    assert (output / "lora_config.json").is_file()
    assert (output / "train_lora.py").is_file()


def test_verified_exemplar_is_injected_before_model_request(tmp_path: Path):
    repository = ExperienceRepository(tmp_path / "experience.db")
    repository.save_agent_trace(str(tmp_path), "inspect alpha.py module",
        [{"tool": "read_file", "arguments": {"path": "alpha.py"}, "success": True}],
        "Alpha module is valid", verified_success=True)

    class CaptureClient(TextOnlyClient):
        def complete(self, **kwargs):
            self.messages = kwargs["messages"]
            return super().complete(**kwargs)

    client = CaptureClient("Inspected the module.")
    agent = DeveloperAgent(llm_client=client, repository=repository,
                           workspace_root=str(tmp_path), preferences_dir=tmp_path / "prefs")
    list(agent.run_stream("inspect alpha.py module carefully", max_steps=1))
    assert "Verified prior example" in client.messages[0]["content"]


def test_explicit_multi_agent_request_selects_swarm(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "ui.db", config_dir=tmp_path / "prefs",
                             workspace_root=str(tmp_path))
    assert app._should_swarm("using multiple agents build a 3D app")
    assert not app._should_isolate("using multiple agents build a 3D app")
    app.session_store.close()


def test_launch_workspace_and_manual_model_survive_session_resume(tmp_path: Path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    database = tmp_path / "ui.db"
    monkeypatch.chdir(first)
    app = AdaptiveHarnessApp(db_path=database, config_dir=tmp_path / "prefs")
    session_id = app.session.id
    assert app.workspace_root == str(first)
    assert app.agent.explicit_model == "stealth/space-bunny-alpha"
    assert app.agent.llm_client.base_url == "https://openrouter.ai/api/v1"
    assert app.swarm_mode == "auto" and app.isolation_mode == "off"
    app.session_store.close()
    monkeypatch.chdir(second)
    resumed = AdaptiveHarnessApp(db_path=database, config_dir=tmp_path / "prefs",
                                 session_id=session_id)
    assert resumed.workspace_root == str(second)
    assert resumed.agent.workspace_root == second
    resumed.session_store.close()


@pytest.mark.anyio
async def test_model_and_theme_popup_labels_have_visible_foreground(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "ui.db", config_dir=tmp_path / "prefs",
                             workspace_root=str(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(QuickSelectModal("Models", [("z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash")]))
        await pilot.pause()
        choice = app.screen.query(".quick-choice").first()
        assert choice.styles.color.a > 0.7
        assert "z-ai/glm-5.3-flash" in choice.label.plain
        await pilot.press("escape")
        app.push_screen(ThemePickerModal("textual-dark"))
        await pilot.pause()
        theme_choice = app.screen.query(".theme-option").first()
        assert theme_choice.styles.color.a > 0.7
        await pilot.press("escape")


@pytest.mark.anyio
async def test_swarm_prompt_updates_live_skill_probabilities(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "ui.db", config_dir=tmp_path / "prefs",
                             workspace_root=str(tmp_path))
    async with app.run_test(size=(120, 30)):
        app._prepare_swarm_telemetry("using multiple agents build a GPS app")
        telemetry = app.query_one("#telemetry", ClassifierTelemetryWidget)
        assert telemetry.primary_skill != "none"
        assert sum(telemetry.probabilities.values()) == pytest.approx(1.0, abs=0.001)
        assert telemetry.entropy > 0
