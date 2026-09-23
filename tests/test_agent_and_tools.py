"""Unit tests for Phase 2: OpenRouter LLM client, developer tools, pervasive classifiers, agent, and TUI."""

from __future__ import annotations

import os
from pathlib import Path
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

    # 2. Ambiguous user choice prompt
    s_ambig = skill_clf.classify("either use sqlite or postgresql, what do you think?")
    res_ambig = ambig_clf.evaluate("either use sqlite or postgresql, what do you think?", s_ambig)
    assert res_ambig.should_ask_question is True
    assert res_ambig.risk_level == "medium"
    assert len(res_ambig.suggested_options) > 0

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

        app._handle_slash_command("/help")
        app._handle_slash_command("/clear")
