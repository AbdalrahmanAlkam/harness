"""The prompt registry centralizes, exposes, and customizes every model-facing prompt."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.classifiers.runtime_overseer import RuntimeOverseer
from adaptive_harness.llm.mock_client import LLMResponse
from adaptive_harness.prompts import PromptRegistry


def test_registry_exposes_all_prompt_families():
    registry = PromptRegistry()
    names = registry.names()
    for family in ("system.", "domain.guidance.", "intervention.", "harness.", "swarm.", "classifier."):
        assert any(name.startswith(family) for name in names), family
    assert registry.get("intervention.looping", target="x.py").startswith("OVERSEER INTERVENTION")
    assert "STOP-CIRCLING" in registry.get("intervention.stop_circling")
    assert "TERMINATION VERDICT" in registry.get("intervention.termination_verdict",
                                                state="LOOPING_DETECTED")
    assert registry.get("intervention.hallucination", evidence="boom").startswith("OVERSEER INTERVENTION")


def test_workspace_override_file_customizes_prompts(tmp_path: Path):
    overrides = {"harness.missing_file_changes": "CUSTOM NUDGE: write the file now."}
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness" / "prompts.json").write_text(json.dumps(overrides), encoding="utf-8")
    registry = PromptRegistry.for_workspace(tmp_path)
    assert registry.get("harness.missing_file_changes") == "CUSTOM NUDGE: write the file now."
    assert registry.is_overridden("harness.missing_file_changes")
    assert not PromptRegistry().is_overridden("harness.missing_file_changes")
    assert registry.get("system.default") == PromptRegistry().get("system.default")


def test_agent_injections_use_registry_overrides(tmp_path: Path):
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness" / "prompts.json").write_text(json.dumps(
        {"harness.missing_file_changes": "CUSTOM NUDGE: write the file now."}), encoding="utf-8")

    class TextOnlyClient:
        default_model = "mock"

        def complete(self, **kwargs):
            return LLMResponse(content="```python\nprint('not saved')\n```")

    agent = DeveloperAgent(llm_client=TextOnlyClient(), workspace_root=str(tmp_path),
                           preferences_dir=tmp_path / "prefs")
    events = list(agent.run_stream("create app.py", max_steps=2))
    injections = [event.payload for event in events if event.event_type == "prompt_injection"]
    assert any(item["content"] == "CUSTOM NUDGE: write the file now." for item in injections)
    assert any(item["source"] == "harness" for item in injections)


def test_system_prompt_assembly_uses_registry(tmp_path: Path):
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness" / "prompts.json").write_text(json.dumps(
        {"system.tool_guidance": "CUSTOM TOOL GUIDANCE"}), encoding="utf-8")

    class TextClient:
        default_model = "mock"

        def complete(self, **kwargs):
            return LLMResponse(content="All good.")

    agent = DeveloperAgent(llm_client=TextClient(), workspace_root=str(tmp_path),
                           preferences_dir=tmp_path / "prefs", forced_mode="coding")
    events = list(agent.run_stream("say hello"))
    prompt = next(event.payload for event in events if event.event_type == "system_prompt")
    assert "CUSTOM TOOL GUIDANCE" in prompt["content"]
    assert "You are Adaptive Agent" in prompt["content"]


def test_overseer_directives_render_from_templates():
    overseer = RuntimeOverseer("fix parser", prompts=PromptRegistry())
    for _ in range(2):
        assert overseer.observe("edit_file", {"path": "parser.py"}, success=True)
    decision = overseer.observe("edit_file", {"path": "parser.py"}, success=True)
    assert decision.directive.startswith("OVERSEER INTERVENTION")
    assert "parser.py" in decision.directive


def test_export_contains_editable_text():
    data = PromptRegistry().export()
    assert data["system.default"].startswith("You are Adaptive Agent")
    assert "CLASSIFIER STOP-CIRCLING DIRECTIVE" in data["intervention.stop_circling"]
