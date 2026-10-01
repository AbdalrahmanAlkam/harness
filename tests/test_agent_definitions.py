"""User-defined subagents.

A subagent used to be one of four names hardcoded into a tool's JSON schema, so
the agent surface was a property of the binary: adding a "database reviewer"
meant editing the source, and there was nowhere to say that a given subagent
wanted a different model, fewer permissions, or a worktree of its own.

These tests pin that a subagent is a file, that the file is honoured rather
than merely read, and that a broken one cannot stop the harness starting.

The frontmatter keys deliberately match Claude Code's documented ones, so a
definition written for either works in both -- which is the whole reason for the
naming.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.agents import (
    AgentDefinitionError,
    AgentRegistry,
    builtin_definitions,
    load_definition,
    parse_frontmatter,
    resolve,
    scaffold,
)

GOOD = """---
name: db-reviewer
description: Reviews a migration for safety and reversibility.
tools: read_file, search_files, run_bash
disallowedTools: write_file
model: fast
permissionMode: plan
maxTurns: 12
effort: high
isolation: worktree
background: true
color: blue
memory: project
---

You review database migrations.
"""


def _write(root: Path, name: str, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


# --- parsing ----------------------------------------------------------------


def test_a_definition_parses_into_frontmatter_and_a_body():
    fields, body = parse_frontmatter(GOOD, "db-reviewer.md")
    assert fields["name"] == "db-reviewer"
    assert fields["tools"] == ["read_file", "search_files", "run_bash"]
    assert fields["maxTurns"] == 12
    assert fields["background"] is True
    assert body == "You review database migrations."


def test_a_list_may_be_written_across_lines():
    text = ("---\nname: x\ndescription: d\ntools:\n  - read_file\n  - run_bash\n---\nBody\n")
    fields, _ = parse_frontmatter(text, "x.md")
    assert fields["tools"] == ["read_file", "run_bash"]


def test_a_bracketed_list_is_accepted():
    text = "---\nname: x\ndescription: d\ntools: [read_file, run_bash]\n---\n"
    fields, _ = parse_frontmatter(text, "x.md")
    assert fields["tools"] == ["read_file", "run_bash"]


def test_a_file_with_no_frontmatter_is_refused_with_a_reason():
    with pytest.raises(AgentDefinitionError) as excinfo:
        parse_frontmatter("just prose\n", "x.md")
    assert "---" in str(excinfo.value), "the error should show the expected shape"


def test_a_description_is_required_because_the_model_reads_it():
    """It is how the model decides when to spawn this agent, so an empty one is
    never chosen and should not be accepted."""
    with pytest.raises(AgentDefinitionError) as excinfo:
        load_definition(_write(Path("/tmp"), "nodesc", "---\nname: nodesc\n---\nBody\n"))
    assert "description" in str(excinfo.value)


def test_an_unreadable_isolation_is_refused():
    bad = "---\nname: x\ndescription: d\nisolation: container\n---\n"
    with pytest.raises(AgentDefinitionError) as excinfo:
        load_definition(_write(Path("/tmp"), "badiso", bad))
    assert "worktree" in str(excinfo.value)


def test_a_non_numeric_max_turns_is_refused():
    bad = "---\nname: x\ndescription: d\nmaxTurns: lots\n---\n"
    with pytest.raises(AgentDefinitionError):
        load_definition(_write(Path("/tmp"), "badturns", bad))


def test_an_unknown_key_is_carried_rather_than_dropped():
    """A file written for a newer tool should still run here."""
    definition = load_definition(_write(Path("/tmp"), "future",
                                       "---\nname: future\ndescription: d\ncacheTtl: 1h\n---\nB\n"))
    assert "cacheTtl" in definition.extra
    assert definition.name == "future"


# --- what the definition controls --------------------------------------------


def test_the_tool_list_becomes_the_agents_tool_surface(tmp_path: Path):
    definition = load_definition(_write(tmp_path, "db-reviewer", GOOD))
    assert definition.tools == ("read_file", "search_files", "run_bash")
    assert definition.disallowed_tools == ("write_file",)


def test_a_plan_agent_is_not_write_capable():
    """`permissionMode: plan` must mean it, not merely suggest it."""
    definition = load_definition(_write(Path("/tmp"), "planner",
                                       "---\nname: p\ndescription: d\n"
                                       "permissionMode: plan\ntools: read_file, write_file\n---\n"))
    assert not definition.is_write_capable


def test_a_write_capable_agent_is_detected():
    definition = load_definition(_write(Path("/tmp"), "builder",
                                       "---\nname: b\ndescription: d\n"
                                       "tools: read_file, write_file\n---\n"))
    assert definition.is_write_capable


def test_model_turns_and_isolation_survive_the_round_trip(tmp_path: Path):
    definition = load_definition(_write(tmp_path, "db-reviewer", GOOD))
    assert definition.model == "fast"
    assert definition.max_turns == 12
    assert definition.effort == "high"
    assert definition.isolation == "worktree"
    assert definition.background is True
    assert definition.color == "blue"


# --- discovery --------------------------------------------------------------


def test_a_project_definition_is_discovered(tmp_path: Path):
    _write(tmp_path / ".harness" / "agents", "db-reviewer", GOOD)
    registry = AgentRegistry(tmp_path)
    assert "db-reviewer" in registry.discover()
    assert registry.get("db-reviewer").description.startswith("Reviews a migration")


def test_a_broken_definition_does_not_hide_the_working_ones(tmp_path: Path):
    root = tmp_path / ".harness" / "agents"
    _write(root, "db-reviewer", GOOD)
    _write(root, "broken", "no frontmatter at all\n")
    registry = AgentRegistry(tmp_path)
    found = registry.discover()
    assert "db-reviewer" in found
    assert "broken" not in found
    assert registry.warnings, "a broken definition must be reported, not silently dropped"


def test_a_project_definition_overrides_a_user_one(tmp_path: Path):
    """The project's is the more specific and the more deliberate."""
    _write(tmp_path / ".harness" / "agents", "db-reviewer",
           "---\nname: db-reviewer\ndescription: The project's version.\n---\n")
    registry = AgentRegistry(tmp_path)
    registry.discover()
    assert registry.get("db-reviewer").description == "The project's version."


def test_the_built_in_roles_still_resolve_with_no_files_on_disk(tmp_path: Path):
    """Nothing that worked before may stop working."""
    registry = AgentRegistry(tmp_path)
    registry.discover()
    for name in ("architect", "coder", "reviewer", "security"):
        assert resolve(name, registry) is not None
        assert registry.get(name) is None, "a built-in is not a discovered file"


def test_a_user_definition_wins_over_a_built_in_of_the_same_name(tmp_path: Path):
    _write(tmp_path / ".harness" / "agents", "coder",
           "---\nname: coder\ndescription: The team's coder, with their rules.\n---\n")
    registry = AgentRegistry(tmp_path)
    registry.discover()
    assert "team's" in resolve("coder", registry).description


# --- the listing ------------------------------------------------------------


def test_the_listing_shows_what_matters_per_agent(tmp_path: Path):
    _write(tmp_path / ".harness" / "agents", "db-reviewer", GOOD)
    registry = AgentRegistry(tmp_path)
    registry.discover()
    described = registry.describe()
    assert "db-reviewer" in described
    for flag in ("background", "worktree", "fast", "plan"):
        assert flag in described, f"{flag} is declared but not shown"


def test_an_empty_listing_says_how_to_make_one(tmp_path: Path):
    described = AgentRegistry(tmp_path).describe()
    assert "No subagents" in described
    assert "init" in described, "the empty view should say how to get one"


def test_a_carried_key_is_flagged_as_not_acted_on(tmp_path: Path):
    _write(tmp_path / ".harness" / "agents", "future",
           "---\nname: future\ndescription: d\ncacheTtl: 1h\n---\n")
    registry = AgentRegistry(tmp_path)
    registry.discover()
    assert "not acted on" in registry.describe()


# --- scaffolding ------------------------------------------------------------


def test_a_scaffolded_agent_is_usable_as_written(tmp_path: Path):
    """The first thing a user does must not need fixing first."""
    path = scaffold(tmp_path / ".harness" / "agents", "db-reviewer", "Reviews a migration.")
    definition = load_definition(path)
    assert definition.name == "db-reviewer"
    assert definition.description == "Reviews a migration."
    assert definition.tools, "a scaffold with no tools cannot do anything"
    assert definition.system_prompt


def test_scaffolding_over_an_existing_agent_is_refused(tmp_path: Path):
    scaffold(tmp_path / ".harness" / "agents", "x")
    with pytest.raises(AgentDefinitionError):
        scaffold(tmp_path / ".harness" / "agents", "x")


def test_a_scaffolded_agent_appears_in_discovery(tmp_path: Path):
    scaffold(tmp_path / ".harness" / "agents", "perf", "Audits hot paths.")
    registry = AgentRegistry(tmp_path)
    assert "perf" in registry.discover()


# --- the delegation tool honours the definition -----------------------------


def test_delegation_runs_a_named_definition_and_keeps_the_builtins():
    import tempfile

    from adaptive_harness.llm.client import LLMResponse
    from adaptive_harness.tools.delegation import DelegateSubagentTool

    class Client:
        default_model = "mock"
        provider = "openrouter"

        def __init__(self):
            self.done = False

        def complete(self, messages, **kwargs):
            if not self.done:
                # A write-capable agent has to produce the file it was asked
                # for, or the run correctly reports missing changes.
                self.done = True
                from adaptive_harness.llm.mock_client import ToolCall
                return LLMResponse(model="mock", content="", tool_calls=[ToolCall(
                    id="w1", name="write_file",
                    arguments={"path": "review.md", "content": "safe"})])
            return LLMResponse(model="mock", content="Safe: additive and reversible.")

    root = Path(tempfile.mkdtemp())
    # No isolation here: this test is about a named definition running, and a
    # worktree needs a repository, which a tmp_path is not.
    _write(root / ".harness" / "agents", "db-reviewer",
           "---\nname: db-reviewer\ndescription: Reviews a migration.\n"
           "tools: read_file, write_file, run_bash\n---\n"
           "You build database migrations.\n")
    (root / "m.sql").write_text("ALTER TABLE t ADD COLUMN c int;")

    tool = DelegateSubagentTool(root, llm_client_factory=Client)
    result = tool.execute(role="db-reviewer", task="review m.sql")
    assert result.success, result.error
    assert "Safe" in result.output

    # And a built-in role still runs, unchanged. `coder` with a client that
    # writes: a `reviewer` deliberately needs test evidence and an `architect`
    # is read-only, so neither is a fair check with a stub that writes a file.
    class Writer:
        default_model = "mock"
        provider = "openrouter"

        def __init__(self):
            self.done = False

        def complete(self, messages, **kwargs):
            if not self.done:
                self.done = True
                from adaptive_harness.llm.mock_client import ToolCall
                return LLMResponse(model="mock", content="", tool_calls=[ToolCall(
                    id="w2", name="write_file",
                    arguments={"path": "plan.md", "content": "plan"})])
            return LLMResponse(model="mock", content="Wrote the plan.")

    tool.llm_client_factory = Writer
    builtin = tool.execute(role="coder", task="write the plan")
    assert builtin.success, builtin.error


def test_an_unknown_agent_name_is_actionable(tmp_path: Path):
    from adaptive_harness.tools.delegation import DelegateSubagentTool

    class Client:
        default_model = "mock"
        provider = "openrouter"

        def complete(self, messages, **kwargs):
            raise AssertionError("should not be called for an unknown name")

    tool = DelegateSubagentTool(tmp_path, llm_client_factory=Client)
    result = tool.execute(role="not-real", task="x")
    assert not result.success
    assert ".harness/agents/" in result.error, (
        "the error should say where to define one")
    assert "reviewer" in result.error, "and what already exists"


def test_an_agent_declaration_reaches_the_worker(tmp_path: Path):
    """The file is honoured, not merely read: its system prompt is what the
    subagent is actually given."""
    from adaptive_harness.llm.client import LLMResponse
    from adaptive_harness.tools.delegation import DelegateSubagentTool

    seen: list[str] = []

    class Client:
        default_model = "mock"
        provider = "openrouter"

        def complete(self, messages, **kwargs):
            seen.append(str(messages[0].get("content", "")))
            return LLMResponse(model="mock", content="ok")

    # A plan agent is stripped of mutating tools, so this one may write; and no
    # isolation, since a worktree needs a repository and tmp_path is not one.
    _write(tmp_path / ".harness" / "agents", "builder",
           "---\nname: builder\ndescription: Builds things.\n"
           "tools: read_file, write_file, run_bash\n---\n"
           "You build database migrations.\n")
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=Client)
    tool.execute(role="builder", task="review m.sql")
    assert seen, "the agent never called the model"
    assert any("You build database migrations." in text for text in seen), (
        "the declared system prompt was not used")


def test_a_dropped_tool_reaches_the_caller_rather_than_being_silent(tmp_path: Path):
    """One unavailable tool costs that tool, not the run -- but the user has to
    be told, or the agent quietly is not what its file declared."""
    from adaptive_harness.llm.client import LLMResponse
    from adaptive_harness.tools.delegation import DelegateSubagentTool

    seen: list[dict] = []

    class Client:
        default_model = "mock"
        provider = "openrouter"

        def __init__(self):
            self.done = False

        def complete(self, messages, **kwargs):
            if not self.done:
                self.done = True
                return LLMResponse(model="mock", content="",
                                   tool_calls=[__import__(
                                       "adaptive_harness.llm.mock_client",
                                       fromlist=["ToolCall"]).ToolCall(
                                       id="w1", name="write_file",
                                       arguments={"path": "out.md",
                                                  "content": "done"})])
            return LLMResponse(model="mock", content="done")

    # A write-capable agent, so the run is judged on the file it produces
    # rather than on the tool it could not have.
    _write(tmp_path / ".harness" / "agents", "mixed", "---\nname: mixed\n"
           "description: d\ntools: read_file, write_file, teleport\n---\nBody\n")
    tool = DelegateSubagentTool(tmp_path, llm_client_factory=Client,
                                on_event=lambda e: seen.append(e.payload))
    result = tool.execute(role="mixed", task="do the thing")
    assert result.success, result.error
    assert result.metadata["unavailable_tools"] == ["teleport"]
    assert any(event.get("tools") == ["teleport"] for event in seen), (
        "the dropped tool was never announced")
