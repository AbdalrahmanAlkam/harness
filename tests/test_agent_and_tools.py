"""Unit tests for Phase 2: OpenRouter LLM client, developer tools, pervasive classifiers, agent, and TUI."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import pytest

from adaptive_harness.agent.agent import AgentEvent, DeveloperAgent
from adaptive_harness.classifiers.ambiguity_classifier import AmbiguityClassifier
from adaptive_harness.classifiers.complexity_router import ComplexityRouter
from adaptive_harness.classifiers.skill_classifier import SkillClassifier
from adaptive_harness.classifiers.verification_classifier import VerificationClassifier
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.llm.client import LLMClient, MODEL_TIERS
from adaptive_harness.llm.mock_client import MockLLMClient
from adaptive_harness.tools.base import ToolResult
from adaptive_harness.tools.bash import RunBashTool
from adaptive_harness.tools.clarification import AskUserTool
from adaptive_harness.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from adaptive_harness.tools.testing import RunPytestTool
from adaptive_harness.tools.workspace import ListDirectoryTool, SearchFilesTool
from adaptive_harness.tui.app import AdaptiveHarnessApp
from adaptive_harness.tui.widgets import ClarificationModal
from adaptive_harness.data.sessions import SessionStore
from adaptive_harness.agent.skills import SkillCatalog
from adaptive_harness.llm.mock_client import LLMResponse
from adaptive_harness.llm.mock_client import ToolCall
from adaptive_harness.classifiers.risk_classifier import ToolRiskClassifier
from adaptive_harness.tools.science import CalculateTool
from adaptive_harness.tools.research import WebSearchTool
from adaptive_harness.classifiers.engine import (BaseClassifierBackend, Classification, create_backend,
    _parse_response, semif_weights_cached)
from adaptive_harness.classifiers.semif_engine import SemIfEngine
from adaptive_harness.classifiers.domain_classifier import DomainClassifier, DomainMode
from adaptive_harness.classifiers.thinking_classifier import ThinkingClassifier, ThinkingLevel
import time


def test_llm_client_mock_mode():
    client = LLMClient(force_mock=True)
    assert client.is_mock is True
    assert client.get_model_for_tier("fast") == MODEL_TIERS["fast"]
    assert client.get_model_for_tier("reasoning") == MODEL_TIERS["reasoning"]

    # Test conversational completion
    resp = client.complete(messages=[{"role": "user", "content": "What is the harness?"}])
    assert resp.content is not None
    assert resp.finish_reason == "stop"
    assert resp.usage["prompt_tokens"] > 0

    # Test tool completion generation from user intent
    resp_tool = client.complete(messages=[{"role": "user", "content": "run tests with pytest"}])
    assert len(resp_tool.tool_calls) == 1
    assert resp_tool.tool_calls[0].name == "run_bash"


def test_local_task_endpoint_does_not_receive_environment_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-cloud-key")
    client = LLMClient(base_url="http://localhost:8080/v1")
    assert client.api_key == "local"
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://127.0.0.1:8080/v1")
    assert LLMClient().api_key == "local"


def test_openrouter_reasoning_request_uses_effort_without_temperature():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None),
                                                      finish_reason="stop")], usage=None, model=kwargs["model"])

    client = LLMClient(force_mock=True)
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = client.complete([{"role": "user", "content": "Solve a proof"}],
                               model=MODEL_TIERS["reasoning"], reasoning_effort="high")
    assert response.content == "ok"
    assert captured["model"] == MODEL_TIERS["reasoning"]
    assert captured["extra_body"] == {"reasoning": {"effort": "high"}}
    assert "temperature" not in captured


def test_run_bash_tool(tmp_path: Path):
    bash_tool = RunBashTool(workspace_root=tmp_path)

    # Safe command
    res = bash_tool.execute(command="echo 'harness test output'")
    assert res.success is True
    assert "harness test output" in res.output
    assert res.metadata.get("exit_code") == 0

    # Dangerous command filter
    danger_res = bash_tool.execute(command="rm -rf /")
    assert danger_res.success is False
    assert "blocked by safety filter" in str(danger_res.error)

    # Failing command
    fail_res = bash_tool.execute(command="sh -c 'exit 7'")
    assert fail_res.success is False
    assert fail_res.metadata.get("exit_code") == 7


def test_file_ops_tools(tmp_path: Path):
    writer = WriteFileTool(workspace_root=tmp_path)
    reader = ReadFileTool(workspace_root=tmp_path)
    editor = EditFileTool(workspace_root=tmp_path)

    # 1. Write file
    content = "line 1: hello\nline 2: target\nline 3: world\n"
    res_w = writer.execute(path="test_file.txt", content=content)
    assert res_w.success is True
    assert (tmp_path / "test_file.txt").exists()

    # 2. Read file full & sliced
    res_r = reader.execute(path="test_file.txt")
    assert res_r.success is True
    assert "target" in res_r.output
    assert res_r.metadata["total_lines"] == 3

    res_slice = reader.execute(path="test_file.txt", start_line=2, end_line=2)
    assert res_slice.success is True
    assert "target" in res_slice.output
    assert "hello" not in res_slice.output

    # 3. Read non-existent file
    res_nonexistent = reader.execute(path="missing.txt")
    assert res_nonexistent.success is False
    assert "File not found" in str(res_nonexistent.error)

    # 4. Edit file
    res_e = editor.execute(
        path="test_file.txt",
        target_text="line 2: target\n",
        replacement_text="line 2: REPLACED\n",
    )
    assert res_e.success is True
    assert "Diff:" in res_e.output
    updated = (tmp_path / "test_file.txt").read_text()
    assert "REPLACED" in updated
    assert "target" not in updated

    # 5. Edit file failure (target not found)
    res_fail = editor.execute(
        path="test_file.txt",
        target_text="nonexistent text block",
        replacement_text="replacement",
    )
    assert res_fail.success is False
    assert "Target text was not found" in str(res_fail.error)


def test_workspace_tools(tmp_path: Path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "a.py").write_text("def find_me():\n    return 42\n")
    (tmp_path / "b.txt").write_text("plain text file")

    list_tool = ListDirectoryTool(workspace_root=tmp_path)
    res_list = list_tool.execute(path=".")
    assert res_list.success is True
    assert "subdir" in res_list.output
    assert "b.txt" in res_list.output

    search_tool = SearchFilesTool(workspace_root=tmp_path)
    res_search = search_tool.execute(pattern="find_me", file_extension="py")
    assert res_search.success is True
    assert "a.py" in res_search.output
    assert res_search.metadata["match_count"] >= 1


def test_clarification_and_schema():
    # Callback execution
    asked = {}

    def mock_cb(question, options, context):
        asked["q"] = question
        asked["opts"] = options
        return options[1]

    tool = AskUserTool(callback=mock_cb)
    res = tool.execute(
        question="Which database?",
        options=["SQLite", "PostgreSQL"],
        context="Storage design",
    )
    assert res.success is True
    assert res.metadata["answer"] == "PostgreSQL"
    assert asked["q"] == "Which database?"

    # OpenAI schema test
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "ask_user"
    assert "question" in schema["function"]["parameters"]["properties"]


def test_skill_classifier():
    clf = SkillClassifier()

    # Test distinct task intents
    res_test = clf.classify("run pytest on tests/test_harness.py")
    assert res_test.primary_skill == "testing"
    assert res_test.confidence > 0.4

    res_edit = clf.classify("write file src/engine.py and create new class")
    assert res_edit.primary_skill == "code_edit"

    res_cmd = clf.classify("git status and check branch")
    assert res_cmd.primary_skill == "run_command"

    res_explore = clf.classify("search files for TaskClassifier in codebase")
    assert res_explore.primary_skill == "search_explore"

    # Test probability distribution properties
    assert pytest.approx(sum(res_test.probabilities.values()), 0.01) == 1.0
    assert len(res_test.ranked_skills) == len(clf.pipeline.named_steps["clf"].classes_)


def test_ambiguity_classifier():
    skill_clf = SkillClassifier()
    ambig_clf = AmbiguityClassifier(entropy_threshold=1.2, margin_threshold=0.25)

    # 1. High risk action
    s_risk = skill_clf.classify("rm -rf output/ and wipe all traces")
    res_risk = ambig_clf.evaluate("rm -rf output/ and wipe all traces", s_risk)
    assert res_risk.should_ask_question is True
    assert res_risk.risk_level == "high"
    assert "potentially destructive" in res_risk.reason
    assert res_risk.suggested_question is not None

    # 2. A request for judgment should be handled by the agent.
    s_ambig = skill_clf.classify("either use sqlite or postgresql, what do you think?")
    res_ambig = ambig_clf.evaluate("either use sqlite or postgresql, what do you think?", s_ambig)
    assert res_ambig.should_ask_question is False

    # A genuinely unspecified task still needs a target.
    s_missing = skill_clf.classify("fix it")
    res_missing = ambig_clf.evaluate("fix it", s_missing)
    assert res_missing.should_ask_question is True

    # 3. Clear, unambiguous task
    s_clear = skill_clf.classify("run pytest tests/ -v")
    res_clear = ambig_clf.evaluate("run pytest tests/ -v", s_clear)
    assert res_clear.should_ask_question is False
    assert res_clear.risk_level == "low"

    # 4. to_dict serialization test
    d = res_risk.to_dict()
    assert d["should_ask_question"] is True
    assert d["risk_level"] == "high"


def test_complexity_router():
    router = ComplexityRouter()

    # Reasoning tier
    res_reason = router.route("architect a concurrent distributed actor system to avoid deadlock and race conditions")
    assert res_reason.tier == "reasoning"
    assert res_reason.recommended_model == MODEL_TIERS["reasoning"]

    # Fast tier
    res_fast = router.route("git status")
    assert res_fast.tier == "fast"
    assert res_fast.recommended_model == MODEL_TIERS["fast"]

    # Standard tier
    res_std = router.route("write a function to parse user json logs and add test cases")
    assert res_std.tier == "standard"
    assert res_std.recommended_model == MODEL_TIERS["standard"]


def test_verification_classifier():
    verifier = VerificationClassifier()

    # Success case
    ok_res = ToolResult(success=True, output="Tests passed")
    v_ok = verifier.evaluate("run_pytest", ok_res)
    assert v_ok.status == "SUCCESS"
    assert v_ok.needs_retry is False
    assert v_ok.recommended_action == "PROCEED"

    # Syntax error case
    syntax_res = ToolResult(success=False, output="", error="SyntaxError: invalid syntax line 12")
    v_syntax = verifier.evaluate("edit_file", syntax_res)
    assert v_syntax.status == "SYNTAX_ERROR"
    assert v_syntax.needs_retry is True
    assert v_syntax.recommended_action == "AUTO_RETRY_SYNTAX_FIX"

    # Test failure case
    test_res = ToolResult(success=False, output="FAILED test_harness.py::test_foo", error="AssertionError")
    v_test = verifier.evaluate("run_pytest", test_res)
    assert v_test.status == "TEST_FAILURE"
    assert v_test.needs_retry is True
    assert v_test.recommended_action == "AUTO_RETRY_TEST_FIX"

    # File error case
    file_res = ToolResult(success=False, output="", error="FileNotFoundError: [Errno 2] No such file or directory: 'abc.txt'")
    v_file = verifier.evaluate("read_file", file_res)
    assert v_file.status == "FILE_ERROR"
    assert v_file.needs_retry is True
    assert v_file.recommended_action == "VERIFY_PATH_OR_SEARCH"


def test_developer_agent_stream(tmp_path: Path):
    db_path = tmp_path / "test_exp.db"
    repo = ExperienceRepository(db_path)
    client = LLMClient(force_mock=True)

    clarified = False

    def clarification_hook(q, opts, ctx):
        nonlocal clarified
        clarified = True
        return opts[0] if opts else "Approved"

    agent = DeveloperAgent(
        llm_client=client,
        repository=repo,
        clarification_callback=clarification_hook,
        workspace_root=str(tmp_path),
    )

    # Run unambiguous task
    events = list(agent.run_stream("list files in directory"))
    event_types = [e.event_type for e in events]

    assert "skill_classification" in event_types
    assert "ambiguity_assessment" in event_types
    assert "model_routing" in event_types
    assert "thought" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "verification" in event_types
    assert "response" in event_types

    # Verify trace stored in SQLite
    traces = repo.get_recent_traces(limit=5)
    assert len(traces) == 1
    assert "list files" in traces[0]["task_text"]

    # Run ambiguous / high-risk task to trigger clarification
    clarified = False
    events_risk = list(agent.run_stream("delete all logs or format disk"))
    assert clarified is True
    risk_event_types = [e.event_type for e in events_risk]
    assert "clarification_needed" in risk_event_types
    assert "clarification_answered" in risk_event_types


@pytest.mark.anyio
async def test_tui_app_headless(tmp_path: Path):
    db_path = tmp_path / "tui_test.db"
    app = AdaptiveHarnessApp(db_path=db_path)

    async with app.run_test() as pilot:
        # Verify app mounted and widgets exist
        assert app.is_running is True
        assert app.query_one("#chat-log") is not None
        assert app.query_one("#telemetry") is not None
        assert app.query_one("#prompt-input") is not None

        # Test slash commands
        app._handle_slash_command("/key sk-openrouter-test-key-12345")
        assert app.agent.llm_client.api_key == "sk-openrouter-test-key-12345"

        app._handle_slash_command("/tier fast")
        assert app.agent.llm_client.default_model == MODEL_TIERS["fast"]

        app._handle_slash_command("/model google/gemini-2.5-pro")
        assert app.agent.llm_client.default_model == "google/gemini-2.5-pro"
        assert app.agent.explicit_model == "google/gemini-2.5-pro"

        app._handle_slash_command("/classifier ollama qwen2.5:1.5b")
        assert app.agent.classifier_backend.name == "ollama"
        app._handle_slash_command("/classifier semif Qwen/Qwen2.5-3B-Instruct")
        assert app.agent.classifier_backend.name == "semif"
        assert app.agent.classifier_backend.model == "Qwen/Qwen2.5-3B-Instruct"
        app._handle_slash_command("/classifier sklearn")
        assert app.agent.classifier_backend.name == "sklearn"

        app._handle_slash_command("/model auto")
        assert app.agent.explicit_model is None

        app._handle_slash_command("/help")
        app._handle_slash_command("/clear")


def test_model_selection_precedence(tmp_path: Path):
    client = LLMClient(force_mock=True)
    agent = DeveloperAgent(llm_client=client, explicit_model="custom/forced", workspace_root=str(tmp_path))
    events = list(agent.run_stream("git status", max_steps=1))
    route = next(e.payload for e in events if e.event_type == "model_routing")
    assert route["model"] == "custom/forced"
    assert route["selection"] == "forced"
    assert next(e.payload for e in events if e.event_type == "thought")["model"] == "custom/forced"
    agent.explicit_model = None
    route = next(e.payload for e in agent.run_stream("git status", max_steps=1) if e.event_type == "model_routing")
    assert route["model"] == MODEL_TIERS["fast"]
    assert route["selection"] == "auto"


def test_classifier_backends_and_modes():
    result = create_backend().classify("run pytest", ["testing", "code_edit"])
    assert result.label == "testing"
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    parsed = _parse_response('{"label":"coding","confidence":0.8,"reasoning":"code task"}',
                             ["coding", "research"], time.perf_counter())
    assert parsed.probabilities["coding"] == pytest.approx(0.8)
    assert DomainClassifier().classify("audit for SQL injection vulnerabilities").mode == DomainMode.AUDIT
    assert ThinkingClassifier().classify("git status").level == ThinkingLevel.NONE
    assert ThinkingClassifier().classify("find a deadlock in concurrency").level == ThinkingLevel.DEEP


@pytest.mark.anyio
async def test_clarification_modal_keyboard():
    app = AdaptiveHarnessApp(db_path=":memory:")
    async with app.run_test() as pilot:
        selected = []
        app.push_screen(ClarificationModal("Choose one", ["First", "Second"], "Reason"), callback=selected.append)
        await pilot.pause()
        assert app.screen.focused.id == "opt-0"
        await pilot.press("2")
        await pilot.pause()
        assert selected == ["Second"]
        app.push_screen(ClarificationModal("Choose one", ["First", "Second"]), callback=selected.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert selected[-1] == "Action cancelled by user"
        app.push_screen(ClarificationModal("Which file?", options=[]), callback=selected.append)
        await pilot.pause()
        assert app.screen.focused.id == "write-in-input"
        await pilot.press("f", "i", "l", "e", "space", "1", "2", "3", "enter")
        await pilot.pause()
        assert selected[-1] == "file 123"


def test_headless_destructive_task_aborts(tmp_path: Path):
    agent = DeveloperAgent(llm_client=LLMClient(force_mock=True), workspace_root=str(tmp_path))
    events = list(agent.run_stream("delete all logs or format disk"))
    assert next(e.payload for e in events if e.event_type == "response")["steps"] == 0
    assert not any(e.event_type == "tool_call" for e in events)


def test_session_store_roundtrip(tmp_path: Path):
    db = tmp_path / "sessions.db"
    store = SessionStore(db)
    session = store.create(str(tmp_path), "Research run")
    session.messages = [{"role": "user", "content": "hello"}]
    session.model = "local/model"
    session.skills = ["literature"]
    session.settings = {"classifier_backend": "ollama", "classifier_model": "qwen2.5:1.5b"}
    store.save(session)
    assert store.load(session.id) == session
    assert store.list()[0].id == session.id
    store.close()
    reopened = SessionStore(db)
    assert reopened.load(session.id) == session
    reopened.close()


def test_session_store_migrates_existing_database(tmp_path: Path):
    db = tmp_path / "old-sessions.db"
    connection = sqlite3.connect(db)
    connection.execute("""CREATE TABLE agent_sessions (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, workspace TEXT NOT NULL,
        messages TEXT NOT NULL, model TEXT, skills TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    connection.execute("INSERT INTO agent_sessions(id,title,workspace,messages,skills) VALUES (?,?,?,?,?)",
                       ("legacy", "Old session", str(tmp_path), "[]", "[]"))
    connection.commit()
    connection.close()
    store = SessionStore(db)
    assert store.load("legacy").settings == {}
    store.close()


@pytest.mark.anyio
async def test_tui_workspace_sessions_and_skills(tmp_path: Path):
    workspace = tmp_path / "project"
    skill_path = workspace / ".harness" / "skills" / "review" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("# Review\nCheck evidence.")
    app = AdaptiveHarnessApp(db_path=tmp_path / "ui.db", workspace_root=str(tmp_path))
    async with app.run_test(size=(70, 24)) as pilot:
        app._handle_slash_command(f"/workspace {workspace}")
        assert app.agent.workspace_root == workspace
        assert app.agent.tools["read_file"].workspace_root == workspace
        app._handle_slash_command("/skill review")
        assert "review" in app.agent.active_skills
        previous_id = app.session.id
        app._handle_slash_command("/session new Second")
        assert app.session.id != previous_id
        app._handle_slash_command(f"/session load {previous_id}")
        assert app.session.id == previous_id
        assert "review" in app.agent.active_skills


@pytest.mark.anyio
async def test_long_clarification_fits_small_terminal(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "small.db")
    async with app.run_test(size=(60, 20)) as pilot:
        choices = []
        app.push_screen(ClarificationModal("Long question " * 15,
                                          ["A very long architecture choice " * 8, "Alternative approach " * 8],
                                          "Detailed reason " * 12), callback=choices.append)
        await pilot.pause()
        card = app.screen.query_one("#modal-container")
        assert card.region.height <= 18
        question = app.screen.query_one("#modal-question")
        assert question.region.width < card.region.width
        assert question.region.y >= card.region.y
        await pilot.press("2")
        await pilot.pause()
        assert choices and choices[0].startswith("Alternative")
        app.push_screen(ClarificationModal("Choose", ["First", "Second"]), callback=choices.append)
        await pilot.pause()
        app.screen.query_one("#opt-text-1").scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#opt-text-1")
        await pilot.pause()
        assert choices[-1] == "Second"


def test_model_error_is_not_reported_as_success(tmp_path: Path):
    class FailingClient:
        default_model = "broken"

        def complete(self, **kwargs):
            return LLMResponse(content="API call failed", model="broken", finish_reason="error")

    events = list(DeveloperAgent(llm_client=FailingClient(), workspace_root=str(tmp_path)).run_stream("git status"))
    assert any(e.event_type == "llm_error" for e in events)
    assert next(e.payload for e in events if e.event_type == "response")["success"] is False


def test_workspace_paths_and_pytest_arguments(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    reader = ReadFileTool(tmp_path)
    writer = WriteFileTool(tmp_path)
    assert not reader.execute(path=f"../{outside.name}").success
    assert not writer.execute(path=f"../{outside.name}", content="bad").success
    assert not RunPytestTool(tmp_path).execute(test_path=f"../{outside.name}").success
    outside.write_text("outside-secret-pattern")
    (tmp_path / "linked.txt").symlink_to(outside)
    assert SearchFilesTool(tmp_path).execute(pattern="outside-secret-pattern").metadata["match_count"] == 0
    (tmp_path / "tests").mkdir()
    result = RunPytestTool(tmp_path).execute(test_path="tests", extra_args="; touch injected")
    assert not (tmp_path / "injected").exists()


def test_destructive_tool_call_is_checked(tmp_path: Path):
    class RiskyClient:
        default_model = "mock"

        def complete(self, **kwargs):
            return LLMResponse(model="mock", tool_calls=[ToolCall(id="1", name="run_bash",
                               arguments={"command": "rm -rf output"})])

    agent = DeveloperAgent(llm_client=RiskyClient(), workspace_root=str(tmp_path))
    events = list(agent.run_stream("clean old build output", max_steps=1))
    assert ToolRiskClassifier().evaluate("run_bash", {"command": "rm -rf output"})
    assert any(e.event_type == "clarification_needed" for e in events)
    assert not any(e.event_type == "tool_call" for e in events)


def test_science_and_optional_research_tools(monkeypatch):
    calculator = CalculateTool()
    assert calculator.execute("(2 + 3) * 4").metadata["value"] == 20
    assert not calculator.execute("__import__('os')").success
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert not WebSearchTool().execute("test").success


def test_selected_classifier_drives_all_three_heads(tmp_path: Path):
    class CountingBackend(BaseClassifierBackend):
        name = "counting"
        model = "tiny-local"

        def __init__(self):
            self.calls = []

        def classify(self, text, labels):
            self.calls.append(tuple(labels))
            chosen = {"coding", "research", "science", "audit"} & set(labels)
            label = "research" if chosen else "none" if "none" in labels else "code_edit"
            probabilities = {item: 1.0 if item == label else 0.0 for item in labels}
            return Classification(label, probabilities, 0.1)

    backend = CountingBackend()
    agent = DeveloperAgent(llm_client=LLMClient(force_mock=True), classifier_backend=backend,
                           workspace_root=str(tmp_path))
    events = list(agent.run_stream("summarize the papers", max_steps=1))
    assert len(backend.calls) >= 4
    assert next(e.payload for e in events if e.event_type == "domain_mode")["mode"] == "research"
    assert next(e.payload for e in events if e.event_type == "skill_classification")["backend"] == "counting"
    assert "fast" in backend.calls[1] or "standard" in backend.calls[1]


def test_semif_engine_one_forward_pass_and_temperature():
    class Tensor:
        def __init__(self, values):
            self.values = values
        def to(self, device):
            return self
        def float(self):
            return self
        def cpu(self):
            return self
        def __truediv__(self, divisor):
            return Tensor([value / divisor for value in self.values])
        def __getitem__(self, index):
            return self.values[index]
        def __iter__(self):
            return iter(self.values)

    class FakeTorch:
        @staticmethod
        def inference_mode():
            from contextlib import nullcontext
            return nullcontext()
        @staticmethod
        def softmax(tensor, dim=0):
            import math
            weights = [math.exp(value) for value in tensor.values]
            return Tensor([value / sum(weights) for value in weights])

    class Tokenizer:
        truncation_side = "right"
        def encode(self, text, add_special_tokens=False):
            return [ord(text[-1])]
        def __call__(self, prompt, **kwargs):
            return {"input_ids": Tensor([1, 2])}

    class Logits:
        def __getitem__(self, key):
            return Tensor([3.0, 1.0])

    class Model:
        calls = 0
        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])
        def __call__(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(logits=Logits())

    engine = SemIfEngine("fixture", temperature=2.0)
    engine._torch, engine._tokenizer, engine._model = FakeTorch(), Tokenizer(), Model()
    result = engine.decide("update the API", {"edit": "modify source code", "test": "run tests"})
    assert engine._model.calls == 1
    assert result.selected_option == "edit"
    assert result.probabilities["edit"] == pytest.approx(1 / (1 + 2.718281828 ** -1))
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.entropy > 0
    assert result.margin == pytest.approx(result.probabilities["edit"] - result.probabilities["test"])
    assert result.latency_ms >= 0


def test_semif_lazy_fallback_and_auto_engine(monkeypatch, tmp_path: Path):
    assert not semif_weights_cached("missing-local-checkpoint")
    assert create_backend("auto", "missing-local-checkpoint").name == "sklearn"
    backend = create_backend("semif", str(tmp_path))
    assert backend.name == "semif"  # construction does not load weights
    backend.engine._load = lambda: (_ for _ in ()).throw(RuntimeError("weights missing"))
    class OfflineClient:
        default_model = "mock"
        def complete(self, **kwargs):
            return LLMResponse(content="fallback answer", model="mock")
    agent = DeveloperAgent(llm_client=OfflineClient(), classifier_backend=backend, workspace_root=str(tmp_path))
    events = list(agent.run_stream("explain this design", max_steps=1))
    assert any(event.event_type == "classifier_error" for event in events)
    assert agent.classifier_backend.name == "sklearn"
    assert next(event.payload for event in events if event.event_type == "skill_classification")["backend"] == "sklearn"


def test_semif_heads_drive_ambiguity_tier_and_verification():
    from adaptive_harness.classifiers.skill_classifier import SkillClassificationResult

    skill = SkillClassificationResult("code_edit", 0.3,
        {"code_edit": 0.3, "run_command": 0.25, "search_explore": 0.2, "testing": 0.1,
         "ask_clarification": 0.1, "general_reasoning": 0.05},
        [("code_edit", 0.3), ("run_command", 0.25), ("search_explore", 0.2), ("testing", 0.1),
         ("ask_clarification", 0.1), ("general_reasoning", 0.05)])
    ambiguity = AmbiguityClassifier().evaluate("work on the project", skill, semantic_decision=True)
    assert ambiguity.should_ask_question
    routed = ComplexityRouter().route("simple typo", {"fast": 0.05, "standard": 0.1, "reasoning": 0.85})
    assert routed.tier == "reasoning"
    semantic_verification = VerificationClassifier().evaluate(
        "run_bash", ToolResult(success=False, output="Unexpected result"), semantic_status="test_failure")
    assert semantic_verification.status == "TEST_FAILURE"


def test_classifier_json_validation_and_local_endpoint():
    parsed = _parse_response('```json\n{"label":"coding","confidence":0.7}\n```',
                             ["coding", "research"], time.perf_counter())
    assert parsed.label == "coding"
    with pytest.raises(ValueError):
        _parse_response('{"label":"coding","confidence":"NaN"}',
                        ["coding", "research"], time.perf_counter())
    assert create_backend("local-slm", "tiny", "http://localhost:8080").endpoint == \
        "http://localhost:8080/v1/chat/completions"


def test_invalid_search_and_risky_split_flags(tmp_path: Path):
    assert not SearchFilesTool(tmp_path).execute(pattern="[").success
    risk = ToolRiskClassifier()
    assert risk.evaluate("run_bash", {"command": "rm -r -f build"})
    assert risk.evaluate("run_bash", {"command": "git clean -df"})
    assert risk.evaluate("run_bash", {"command": "git reset --hard"})
    assert risk.evaluate("run_bash", {"command": "echo okay"}) is None


def test_agent_denies_unavailable_domain_tool_and_retains_failure(tmp_path: Path):
    class AuditBackend(BaseClassifierBackend):
        name = "audit-backend"
        model = "test"

        def classify(self, text, labels):
            label = "audit" if "audit" in labels else "low" if "low" in labels else labels[0]
            return Classification(label, {item: float(item == label) for item in labels}, 0.1)

    class WriteClient:
        default_model = "test"

        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(model="test", tool_calls=[ToolCall(id="write", name="write_file",
                    arguments={"path": "should-not-exist.txt", "content": "bad"})])
            return LLMResponse(content="Could not write in audit mode", model="test")

    agent = DeveloperAgent(llm_client=WriteClient(), classifier_backend=AuditBackend(), workspace_root=str(tmp_path))
    events = list(agent.run_stream("audit this workspace", max_steps=2))
    assert not (tmp_path / "should-not-exist.txt").exists()
    result = next(e.payload for e in events if e.event_type == "tool_result")
    assert not result["success"]
    assert "unavailable" in agent.messages[-2]["content"]
    assert next(e.payload for e in events if e.event_type == "response")["success"] is False


@pytest.mark.anyio
async def test_classifier_api_key_and_narrow_layout(tmp_path: Path):
    app = AdaptiveHarnessApp(db_path=tmp_path / "layout.db", classifier_backend="openrouter")
    async with app.run_test(size=(70, 22)) as pilot:
        app._handle_slash_command("/key secret-test-key")
        assert app.agent.classifier_backend.api_key == "secret-test-key"
        assert app.query_one("#chat-log").region.width <= 70
        assert app.query_one("#chat-log").virtual_size.width <= 70
        app._handle_slash_command("/classifier sklearn")
        app._handle_slash_command("/session save")
        assert app.session_store.load(app.session.id).settings["classifier_backend"] == "sklearn"
