"""Regressions for autonomous edits, explicit delegation, and verified learning."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.agent.swarm import DeveloperAgentWorker, SwarmResult
from adaptive_harness.classifiers.risk_classifier import ToolRiskClassifier
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.learning.distill import export_distillation
from adaptive_harness.learning.harvester import harvest_verified_traces
from adaptive_harness.llm.mock_client import LLMResponse
from adaptive_harness.tools.delegation import DelegateSubagentTool
from adaptive_harness.tui.app import AdaptiveHarnessApp


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
    assert app._should_isolate("using multiple agents build a 3D app")
    app.session_store.close()
