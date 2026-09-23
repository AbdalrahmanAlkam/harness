"""Failure-injection and end-to-end regressions from the Phase 5 audit."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
from textual.widgets import Footer, Input

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.data.config import ConfigManager, PromptHistoryStore
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.llm.mock_client import LLMResponse, ToolCall
from adaptive_harness.skills.registry import BUILTIN_BY_NAME
from adaptive_harness.skills.router import SkillRouter
from adaptive_harness.skills.verifier import SkillVerifier
from adaptive_harness.tools.bash import RunBashTool
from adaptive_harness.tools.testing import RunPytestTool
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.widgets import ThemePickerModal


def test_timeout_kills_descendants_and_blocks_root_variants(tmp_path):
    tool = RunBashTool(tmp_path)
    result = tool.execute("(sleep 0.5; touch escaped) & wait", timeout_seconds=0.05)
    assert not result.success and result.metadata["timeout"]
    time.sleep(0.7)
    assert not (tmp_path / "escaped").exists()
    for command in ("rm -fr /", "rm --recursive /", ": () { : | : & }; :"):
        assert "blocked" in tool.execute(command).error
    assert not tool.execute("echo okay", timeout_seconds=0).success


def test_pytest_node_id_is_supported(tmp_path):
    (tmp_path / "test_example.py").write_text("def test_ok(): assert True\ndef test_bad(): assert False\n")
    result = RunPytestTool(tmp_path).execute("test_example.py::test_ok")
    assert result.success and result.metadata["passed"] == 1


def test_live_failure_is_redacted_and_switches_offline():
    client = LLMClient(api_key="private-test-key", force_mock=True)
    def fail(**kwargs):
        raise ConnectionError("failed with private-test-key")
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    response = client.complete([{"role": "user", "content": "hello"}])
    assert response.finish_reason == "error"
    assert "private-test-key" not in response.content
    assert response.usage == {"prompt_tokens": 0, "completion_tokens": 0}
    assert client.is_mock
    assert not response.tool_calls


def test_semantic_routing_does_not_require_keyword_overlap_and_can_chain():
    def decide(context, options):
        return SimpleNamespace(probabilities={name: (0.48 if name == "refactor_clean_code" else
            0.42 if name == "pytest_tdd_loop" else 0.1 / 20) for name in options})
    backend = SimpleNamespace(name="semif", engine=SimpleNamespace(decide=decide))
    selected = SkillRouter().route("Untangle this mess and demonstrate equivalence", BUILTIN_BY_NAME, backend)
    assert [skill.name for skill in selected.skills] == ["refactor_clean_code", "pytest_tdd_loop"]


def test_latest_failed_test_or_edit_invalidates_old_green_result():
    verifier = SkillVerifier()
    skill = BUILTIN_BY_NAME["pytest_tdd_loop"]
    passing = {"name": "run_pytest", "success": True}
    for subsequent in ({"name": "run_pytest", "success": False}, {"name": "edit_file", "success": True}):
        assert not verifier.verify((skill,), (passing, subsequent))[0].verified


def test_cache_preserves_case_sensitive_paths(tmp_path):
    repo = ExperienceRepository(tmp_path / "trace.db")
    (tmp_path / "A.py").write_text("UPPER")
    (tmp_path / "a.py").write_text("lower")
    repo.save_solution("read A.py", str(tmp_path), "settings", "UPPER",
                       {"A.py": hashlib.sha256(b"UPPER").hexdigest()}, 100)
    assert repo.lookup_solution("READ A.py", str(tmp_path), "settings")
    assert repo.lookup_solution("read a.py", str(tmp_path), "settings") is None


def test_corrupt_preferences_and_secret_history_are_safe(tmp_path):
    (tmp_path / "config.json").write_bytes(b"\xff")
    (tmp_path / "prompt_history.txt").write_bytes(b"\xff")
    assert ConfigManager(tmp_path).load() == {}
    history = PromptHistoryStore(tmp_path)
    assert history.entries == []
    assert not history.record("Use sk-or-v1-abcdefghijklmnop for this request")


def test_agent_recovers_from_missing_file_then_hits_memory(tmp_path):
    (tmp_path / "sample.py").write_text("answer = 42\n")
    class Client:
        calls = 0
        def complete(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                return LLMResponse(tool_calls=[ToolCall(id=str(self.calls), name="read_file",
                    arguments={"path": "missing.py" if self.calls == 1 else "sample.py"})])
            return LLMResponse(content="answer = 42")
    client = Client()
    agent = DeveloperAgent(llm_client=client, workspace_root=str(tmp_path),
        repository=ExperienceRepository(tmp_path / "trace.db"), preferences_dir=tmp_path / "prefs")
    events = list(agent.run_stream("read sample.py", max_steps=4))
    results = [event.payload for event in events if event.event_type == "tool_result"]
    assert [result["success"] for result in results] == [False, True]
    assert next(event.payload for event in events if event.event_type == "response")["success"]
    # A run containing a failed read must not be cached; a clean subsequent read is.
    assert not any(event.event_type == "memory_hit" for event in events)
    client.calls = 1
    list(agent.run_stream("read sample.py", max_steps=4))
    calls = client.calls
    cached = list(agent.run_stream("read sample.py", max_steps=4))
    assert any(event.event_type == "memory_hit" for event in cached)
    assert client.calls == calls


@pytest.mark.anyio
async def test_tiny_layout_and_mouse_theme_preview(tmp_path):
    app = AdaptiveHarnessApp(workspace_root=str(tmp_path), db_path=tmp_path / "ui.db", config_dir=tmp_path / "prefs")
    async with app.run_test(size=(120, 30)) as pilot:
        for size in ((40, 10), (30, 8), (80, 24), (120, 30)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            prompt = app.query_one("#prompt-input", Input)
            footer = app.query_one(Footer)
            assert prompt.region.bottom < footer.region.y
        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, ThemePickerModal)
        await pilot.hover("#theme-1")
        await pilot.pause()
        assert app.theme == "nord"
        await pilot.press("escape")
        assert app.theme == "textual-dark"
