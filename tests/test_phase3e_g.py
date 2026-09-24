"""Regression checks for context compaction, deterministic tools, and memory."""

from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from rich.console import Console

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.agent.compaction import compact_tool_output, rank_search_results
from adaptive_harness.data.preferences import ClarificationMemory
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.llm.client import LLMClient
from adaptive_harness.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from adaptive_harness.tools.python_repl import RunPythonReplTool, VerifyEquationTool
from adaptive_harness.tui.widgets import ClassifierTelemetryWidget
from adaptive_harness.strategies.equations import EquationStrategy
from adaptive_harness.models.domain import Task
from adaptive_harness.classifiers.verification_classifier import VerificationClassifier


def test_ast_symbol_slice_and_guard(tmp_path: Path):
    source = "import math\nfrom pathlib import Path\n" + "\n" * 500 + (
        "class Calculator:\n    def square(self, value):\n        return value * value\n")
    path = tmp_path / "sample.py"
    path.write_text(source)
    read = ReadFileTool(tmp_path)
    whole = read.execute("sample.py")
    assert "omitted" in whole.output
    focused = read.execute("sample.py", symbol="Calculator.square")
    assert focused.success
    assert "import math" in focused.output
    assert "return value * value" in focused.output
    assert len(focused.output) < len(source) // 4

    write = WriteFileTool(tmp_path)
    rejected = write.execute("new.py", "def broken(:\n pass")
    assert not rejected.success and "SyntaxError" in rejected.error
    assert not (tmp_path / "new.py").exists()
    edit = EditFileTool(tmp_path)
    rejected = edit.execute("sample.py", "return value * value", "return (")
    assert not rejected.success and "SyntaxError" in rejected.error
    assert path.read_text() == source


def test_compactor_preserves_failure_and_removes_noise():
    output = "\x1b[31mred\x1b[0m\n" + ("PASSED test_noise\n" * 150) + (
        "================ FAILURES ================\n"
        "________ test_math ________\n"
        "E       assert 2 == 3\n"
        "FAILED tests/test_math.py::test_math\n"
        "================ short test summary info ================\n"
        "1 failed, 150 passed in 1.20s\n")
    compact = compact_tool_output("run_pytest", output)
    assert "assert 2 == 3" in compact and "test_math" in compact
    assert "1 failed" in compact
    assert "PASSED test_noise" not in compact and "\x1b" not in compact
    assert len(compact) < len(output) / 2
    bash = compact_tool_output("run_bash", "line\n" * 1000)
    assert len(bash) <= 1500 and "repeated 999" in bash


def test_search_prefilter_restricts_files():
    output = "\n".join(f"module{i}.py:12: token matches here" for i in range(12))
    compact = rank_search_results(output, "inspect module9 token")
    assert "module9.py" in compact
    assert "showing 3 of 12" in compact
    assert len([line for line in compact.splitlines() if ".py:" in line]) == 3
    class Engine:
        calls = 0
        def decide(self, context, options):
            self.calls += 1
            return SimpleNamespace(probabilities={key: (1.0 if "module7.py" in value else 0.0)
                                                  for key, value in options.items()})
    engine = Engine()
    semantic = rank_search_results(output, "find relevant code", engine)
    assert engine.calls == 1
    assert "module7.py" in semantic


def test_python_repl_and_exact_equation_verification(tmp_path: Path):
    repl = RunPythonReplTool()
    if not shutil.which("bwrap"):
        assert not repl.execute("2 + 2").success
    else:
        result = repl.execute("import sympy as sp\nsp.factorint(123456)")
        if not result.success and "Operation not permitted" in (result.error or ""):
            # Some CI/container sandboxes forbid nested Bubblewrap namespaces.
            assert result.metadata["sandbox"] == "bubblewrap"
        else:
            assert result.success, result.error
            assert "2: 6" in result.output
            assert repl.execute("open('/etc/passwd').read()").success is False
    verifier = VerifyEquationTool()
    assert verifier.execute("x**2 - 2 = 0", "x", "sqrt(2)").success
    incorrect = verifier.execute("x**2 - 2 = 0", "x", "2")
    assert not incorrect.success
    assert VerificationClassifier().evaluate("verify_equation", incorrect).status == "MATH_MISMATCH"
    assert "denominator zero" in verifier.execute("x/x = 1", "x", "0").error
    strategy = EquationStrategy()
    equation = Task(text="solve x^2 - 2 = 0")
    solved = strategy.execute(equation)
    checked = strategy.verify(equation, solved)
    assert checked.success and "exact SymPy" in checked.reason
    assert set(checked.details["exact_roots"]) == {"sqrt(2)", "-sqrt(2)"}


def test_exact_verified_memory_invalidates_when_file_changes(tmp_path: Path):
    (tmp_path / "sample.py").write_text("x = 1\n")
    repository = ExperienceRepository(tmp_path / "experience.db")
    client = LLMClient(force_mock=True)
    original = client.complete
    calls = []
    def count_calls(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    client.complete = count_calls
    agent = DeveloperAgent(llm_client=client, repository=repository, workspace_root=str(tmp_path),
                           preferences_dir=tmp_path / "prefs")
    first = list(agent.run_stream("read sample.py", max_steps=3))
    assert next(e.payload for e in first if e.event_type == "response")["success"]
    assert any(e.event_type == "tool_result" and e.payload["success"] for e in first)
    count = len(calls)
    repeated = list(agent.run_stream("read sample.py", max_steps=3))
    assert len(calls) == count
    assert any(e.event_type == "memory_hit" for e in repeated)
    assert next(e.payload["usage"] for e in repeated if e.event_type == "response") == {
        "prompt_tokens": 0, "completion_tokens": 0}
    normalized = list(agent.run_stream("READ   sample.py", max_steps=3))
    assert any(e.event_type == "memory_hit" for e in normalized)
    assert len(calls) == count
    (tmp_path / "sample.py").write_text("x = 2\n")
    list(agent.run_stream("read sample.py", max_steps=3))
    assert len(calls) > count


def test_clarification_preferences_persist_but_not_destructive(tmp_path: Path):
    memory = ClarificationMemory(tmp_path / "prefs")
    assert memory.remember("Which database do you prefer?", "PostgreSQL")
    assert memory.lookup("  which DATABASE do you prefer?  ") == "PostgreSQL"
    assert memory.lookup("Which database should I use for this project?") == "PostgreSQL"
    assert memory.path.stat().st_mode & 0o777 == 0o600
    assert "PostgreSQL" in ClarificationMemory(tmp_path / "prefs").guidance()
    assert not memory.remember("Allow destructive command?", "Proceed", "Destructive tool command detected")
    assert memory.lookup("Allow destructive command?", "Destructive tool command detected") is None
    assert not memory.remember("What should I work on?", "README.md", "The request has no identifiable task target")

    asked = []
    def callback(*args):
        asked.append(args)
        return "SQLite"
    agent = DeveloperAgent(llm_client=LLMClient(force_mock=True), workspace_root=str(tmp_path),
                           preferences_dir=tmp_path / "prefs", clarification_callback=callback)
    assert agent._handle_clarification("Which linter do you prefer?", None, "Choose style") == "SQLite"
    assert agent._handle_clarification("Which linter do you prefer?", None, "Choose style") == "SQLite"
    assert len(asked) == 1
    next_agent = DeveloperAgent(llm_client=LLMClient(force_mock=True), workspace_root=str(tmp_path),
                                preferences_dir=tmp_path / "prefs",
                                clarification_callback=lambda *args: pytest.fail("asked again"))
    assert next_agent._handle_clarification("Which linter do you prefer?", None, "Choose style") == "SQLite"


def test_telemetry_distinguishes_estimates_provider_cache_and_memory():
    widget = ClassifierTelemetryWidget()
    widget.update_telemetry(tokens_saved_estimate=4250, provider_cached_tokens=900,
                            provider_prompt_tokens=1000, memory_resolution=True)
    console = Console(width=52, color_system=None)
    with console.capture() as capture:
        console.print(widget.render())
    rendered = capture.get()
    assert "~4,250" in rendered
    assert "900 tokens (90%)" in rendered
    assert "Verified memory" in rendered


def test_provider_reports_actual_cached_tokens_and_stable_prefix():
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None),
                                     finish_reason="stop")], model=kwargs["model"],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=0.0123,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=60, cache_write_tokens=8),
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=5)))
    client = LLMClient(api_key="test-key")
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    prefix = {"role": "system", "content": "Stable project instructions"}
    for text in ("first", "second"):
        response = client.complete([prefix, {"role": "user", "content": text}],
                                   model="anthropic/claude-sonnet-4")
        assert response.usage["cached_tokens"] == 60
        assert response.usage["total_tokens"] == 120
        assert response.usage["reasoning_tokens"] == 5
        assert response.usage["cache_write_tokens"] == 8
        assert response.usage["cost_usd"] == 0.0123
    assert requests[0]["messages"][0] == requests[1]["messages"][0] == prefix
    assert requests[0]["extra_body"]["cache_control"] == {"type": "ephemeral"}
    assert requests[0]["extra_body"]["usage"] == {"include": True}
