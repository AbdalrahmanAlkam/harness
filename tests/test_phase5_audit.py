"""Failure-injection and end-to-end regressions from the Phase 5 audit."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import time
import json

import pytest
import httpx
from openai import OpenAI
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


def test_live_failure_keeps_saved_key_and_retries_live_next_call(tmp_path):
    config = ConfigManager(tmp_path / "config")
    config.save_key("private-test-key")
    client = LLMClient(api_key="private-test-key")
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ConnectionError("failed with private-test-key")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="recovered", tool_calls=None),
            finish_reason="stop")], usage=None, model=kwargs["model"])
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    response = client.complete([{"role": "user", "content": "hello"}])
    assert response.finish_reason == "error"
    assert "private-test-key" not in response.content
    assert response.usage == {"prompt_tokens": 0, "completion_tokens": 0}
    assert client.is_mock
    assert not response.tool_calls
    assert client.api_key == "private-test-key"
    assert config.load()["api_key"] == "private-test-key"
    recovered = client.complete([{"role": "user", "content": "retry"}])
    assert recovered.content == "recovered"
    assert len(calls) == 2
    assert client.api_key == "private-test-key" and not client.is_mock


def test_openrouter_usage_extension_is_sent_through_sdk_request_body():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "gen-test", "object": "chat.completion", "created": 0,
            "model": "z-ai/glm-5.3-flash", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 2,
            "total_tokens": 6, "cost": 0.00002}})
    transport_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(api_key="test-key")
    client._openai_client = OpenAI(api_key="test-key", base_url=client.base_url, http_client=transport_client)
    try:
        result = client.complete([{"role": "user", "content": "hello"}])
    finally:
        client._openai_client.close()
    assert result.content == "ok"
    assert requests[0]["usage"] == {"include": True}
    assert result.usage["cost_usd"] == 0.00002


def test_reasoning_parameter_rejection_retries_same_model_without_thinking():
    client = LLMClient(api_key="test-key")
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            error = ValueError("Provider returned error: messages.9.content.0.type expected thinking; when thinking is enabled, an assistant message must start with a thinking block.")
            error.status_code = 400
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer", tool_calls=None),
            finish_reason="stop")], usage=None, model=kwargs["model"])
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = client.complete([{"role": "user", "content": "solve"}], model="anthropic/claude-3.7-sonnet",
        reasoning_effort="high", reasoning_budget_tokens=16000)
    assert response.content == "answer"
    assert response.metadata["thinking_fallback"] is True
    assert len(requests) == 2
    assert requests[0]["model"] == requests[1]["model"] == "anthropic/claude-3.7-sonnet"
    assert "reasoning" not in requests[1].get("extra_body", {})
    assert requests[1]["temperature"] == 0.2


def test_claude_history_without_tool_thinking_uses_default_mode_safely():
    client = LLMClient(api_key="test-key")
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None),
            finish_reason="stop")], usage=None, model=kwargs["model"])
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    messages = [{"role": "user", "content": "old task"},
                {"role": "assistant", "content": "running", "tool_calls": [{"id": "1"}]},
                {"role": "user", "content": "new task"}]
    response = client.complete(messages, model="anthropic/claude-sonnet-4", reasoning_effort="high",
                               reasoning_budget_tokens=16000)
    assert response.content == "ok"
    assert response.metadata["thinking_fallback_reason"] == "legacy_tool_history"
    assert "reasoning" not in captured.get("extra_body", {})


@pytest.mark.anyio
async def test_copy_shortcut_copies_latest_agent_output(tmp_path, monkeypatch):
    app = AdaptiveHarnessApp(workspace_root=str(tmp_path), db_path=tmp_path / "ui.db", config_dir=tmp_path / "prefs")
    copied = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._last_agent_content = "Useful answer with exact result."
        app._handle_slash_command("/copy")
        assert copied == ["Useful answer with exact result."]


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


def test_agent_preserves_provider_reasoning_blocks_between_tool_steps(tmp_path):
    captured = []
    class Client:
        def complete(self, **kwargs):
            captured.append(kwargs["messages"])
            if len(captured) == 1:
                return LLMResponse(content="Checking", metadata={"reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "opaque-signature"}]},
                    tool_calls=[ToolCall(id="1", name="run_bash", arguments={"command": "echo ok"})])
            return LLMResponse(content="Done")
    agent = DeveloperAgent(llm_client=Client(), workspace_root=str(tmp_path))
    list(agent.run_stream("run command echo ok", max_steps=2))
    assistant = next(message for message in captured[1] if message.get("role") == "assistant")
    assert assistant["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "opaque-signature"}]


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
