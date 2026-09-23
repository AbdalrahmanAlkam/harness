"""Tests for local craft-skill registration, routing, and scoped execution."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_harness.agent.agent import DeveloperAgent
from adaptive_harness.agent.skills import SkillCatalog
from adaptive_harness.llm.mock_client import LLMResponse
from adaptive_harness.skills.registry import BUILTIN_BY_NAME
from adaptive_harness.skills.router import SkillRouter
from adaptive_harness.skills.verifier import SkillVerifier


def test_builtin_catalog_is_complete_and_token_scoped():
    assert len(BUILTIN_BY_NAME) == 22
    assert len({skill.name for skill in BUILTIN_BY_NAME.values()}) == 22
    assert len({skill.category for skill in BUILTIN_BY_NAME.values()}) == 6
    assert all(skill.instructions and skill.tools and skill.invariants for skill in BUILTIN_BY_NAME.values())
    all_instructions = "\n".join(skill.instructions for skill in BUILTIN_BY_NAME.values())
    assert len(BUILTIN_BY_NAME["symbolic_math_solver"].instructions) < len(all_instructions) * 0.4


def test_local_routing_across_all_six_categories():
    cases = {
        "refactor_clean_code": "Refactor duplicated code in this module",
        "pytest_tdd_loop": "Write pytest regression tests for this bug",
        "symbolic_math_solver": "Solve this algebra equation with SymPy",
        "literature_synthesizer": "Synthesize academic literature and citations",
        "docker_containerizer": "Create a multi-stage Dockerfile",
        "security_audit_scanner": "Audit OWASP injection risk",
    }
    router = SkillRouter()
    for name, task in cases.items():
        selection = router.route(task, BUILTIN_BY_NAME)
        assert selection.skills[0].name == name
        assert selection.confidence >= 0.70
    assert not router.route("Hello there", BUILTIN_BY_NAME).skills


def test_semif_router_probes_all_skills_in_one_decision():
    calls = []

    class Engine:
        def decide(self, context, options):
            calls.append(options)
            probabilities = {key: (0.8 if key == "symbolic_math_solver" else 0.2 / (len(options) - 1))
                             for key in options}
            return type("Decision", (), {"probabilities": probabilities})()

    backend = type("Backend", (), {"name": "semif", "engine": Engine()})()
    selection = SkillRouter().route("solve an equation", BUILTIN_BY_NAME, backend)
    assert len(calls) == 1
    assert len(calls[0]) == 22
    assert selection.skills[0].name == "symbolic_math_solver"
    assert selection.backend == "semif"


def test_semif_skill_failure_falls_back_locally():
    class Engine:
        def decide(self, context, options):
            raise RuntimeError("weights unavailable")

    backend = type("Backend", (), {"name": "semif", "engine": Engine()})()
    selection = SkillRouter().route("Create a Dockerfile", BUILTIN_BY_NAME, backend)
    assert selection.backend == "local-fallback"
    assert selection.skills[0].name == "docker_containerizer"


def test_custom_package_and_manual_override(tmp_path: Path):
    root = tmp_path / "skills" / "my_review"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text("Inspect carefully and report exact line numbers.")
    (root / "skill.json").write_text(json.dumps({"title": "My Review", "category": "Custom",
        "trigger": "Inspect source for style", "tools": ["read_file", "search_files"],
        "invariants": ["evidence_checked"]}))
    catalog = SkillCatalog(tmp_path, tmp_path / "config")
    skill = catalog.get("my_review")
    assert skill.title == "My Review"
    assert skill.tools == ("read_file", "search_files")
    selection = SkillRouter().route("unrelated task", catalog.all(), forced=("my_review",))
    assert selection.skills == (skill,)
    assert selection.selection == "forced"


def test_skill_verifier_uses_tool_evidence():
    verifier = SkillVerifier()
    skill = BUILTIN_BY_NAME["pytest_tdd_loop"]
    assert not verifier.verify((skill,), ())[0].verified
    assert verifier.verify((skill,), ({"name": "run_pytest", "success": True},))[0].verified
    assert not verifier.verify((skill,), ({"name": "run_pytest", "success": False},))[0].verified


def test_agent_injects_only_selected_skill_and_narrows_tools(tmp_path: Path):
    requests = []

    class Client:
        def complete(self, **kwargs):
            requests.append(kwargs)
            return LLMResponse(content="Draft design.")

    agent = DeveloperAgent(llm_client=Client(), workspace_root=str(tmp_path), forced_mode="coding")
    agent.active_skills["architecture_design"] = agent.skill_catalog.read("architecture_design")
    events = list(agent.run_stream("Design an architecture for these components", max_steps=1))
    selected = next(event.payload for event in events if event.event_type == "specialized_skill")
    assert selected["skills"][0]["name"] == "architecture_design"
    request = requests[0]
    assert "Map callers and tests before editing" not in request["messages"][0]["content"]
    assert "State responsibilities, data flow" in request["messages"][0]["content"]
    names = {tool["function"]["name"] for tool in request["tools"]}
    assert "edit_file" not in names
    assert "read_file" in names
    assert any(event.event_type == "skill_verification" for event in events)
    assert next(event.payload for event in events if event.event_type == "response")["success"] is False


def test_audit_policy_allows_read_only_dependency_scanners():
    from adaptive_harness.classifiers.domain_classifier import audit_command_is_read_only
    assert audit_command_is_read_only("pip-audit")
    assert audit_command_is_read_only("safety check")
    assert not audit_command_is_read_only("pip-audit --output report.json")
    assert not audit_command_is_read_only("pip-audit; touch changed.txt")
