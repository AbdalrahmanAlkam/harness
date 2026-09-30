"""The agents-md plugin: the project-memory loader, and what it refuses to do.

Two properties are tested here and they are not the same one.

**Correct reading.** ``memory_load`` returns ``AGENTS.md`` and
``.harness/instructions/*.md`` in a stated precedence order, expands
``@include`` directives, and says which files contributed.

**Not writing.** The plugin declares only ``tools``, because it only reads.
The tests assert that directly against the manifest, and then assert it
behaviourally: a snapshot of every file in the workspace -- content and
mtime -- is taken before the tools are called and compared afterwards, and
a workspace with no instructions must still have no instructions afterwards.

That second one is the reason this plugin exists in the shape it does. A
memory loader that quietly creates ``AGENTS.md`` in a repository the user
merely pointed the agent at has changed something nobody asked it to change,
and the user is the one who has to notice. ``memory_check`` prints the
scaffold and says out loud that it did not create it; these tests hold it to
that.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from adaptive_harness.plugins.host import PluginHost
from plugin_paths import plugin_path  # noqa: E402

AGENTS_MD = """\
# Project rules

- Run `pytest tests/ -q` before calling anything done.
- Never edit files under `vendor/` by hand.
"""

TOPIC_TESTING = """\
## Testing

The suite is slow. Use `-k` rather than running everything twice.
"""

TOPIC_STYLE = """\
## Style

Four-space indent, no tabs, 90 columns.
"""


class AgentsMd:
    """A loaded agents-md plugin, with its module."""

    def __init__(self, host: PluginHost) -> None:
        self.host = host
        self.plugin = next(p for p in host.plugins if p.name == "agents-md")
        matches = [name for name in sys.modules
                   if name.startswith("_adaptive_plugin_")
                   and self.plugin.directory.name in name]
        assert len(matches) == 1, f"expected one imported module, got {matches}"
        self.module = sys.modules[matches[0]]
        self.tools = {tool.name: tool for tool in self.plugin.tools}

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return self.tools[name].handler(**kwargs)


def _load(workspace: Path) -> AgentsMd:
    host = PluginHost(project_root=workspace)
    host.discovery_roots = lambda: [plugin_path("agents-md").parent]
    host.discover()
    return AgentsMd(host)


def _workspace(tmp_path: Path) -> Path:
    """A project with a root file and two topic files, in precedence order."""
    (tmp_path / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    instructions = tmp_path / ".harness" / "instructions"
    instructions.mkdir(parents=True)
    (instructions / "10-testing.md").write_text(TOPIC_TESTING, encoding="utf-8")
    (instructions / "20-style.md").write_text(TOPIC_STYLE, encoding="utf-8")
    return tmp_path


def _snapshot(root: Path) -> dict[str, tuple[str, float, int]]:
    """Every file under ``root``: relative path -> (content, mtime, size).

    The whole tree, not just the instruction files, because a writer that
    creates a *new* file would leave the old ones untouched and still be a
    writer. The mtime is there because a write that restores the content is
    still a write.
    """
    seen: dict[str, tuple[str, float, int]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        key = str(path.relative_to(root))
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = "<binary>"
        seen[key] = (content, stat.st_mtime, stat.st_size)
    return seen


@pytest.fixture
def agents_md(tmp_path: Path) -> AgentsMd:
    return _load(_workspace(tmp_path))


@pytest.fixture
def bare(tmp_path: Path) -> AgentsMd:
    """A plugin pointed at a workspace with no instructions at all."""
    return _load(tmp_path)


# --- it loads ---------------------------------------------------------------


def test_the_plugin_loads_with_its_two_tools(agents_md: AgentsMd):
    assert agents_md.plugin.ok, agents_md.plugin.error
    assert set(agents_md.tools) == {"memory_load", "memory_check"}


def test_the_manifest_asks_only_for_tools(agents_md: AgentsMd):
    """It only reads, so it must not ask for anything that could write."""
    assert agents_md.plugin.permissions == frozenset({"tools"})
    manifest = json.loads((agents_md.plugin.directory / "plugin.plugin.json")
                          .read_text(encoding="utf-8"))
    assert manifest["permissions"] == ["tools"]
    for forbidden in ("subprocess", "fs_home", "env", "net", "write"):
        assert forbidden not in manifest["permissions"]


def test_both_tools_declare_read_risk(agents_md: AgentsMd):
    # Risk drives the strict-profile gate. A reader that claimed to be a write
    # would ask permission for nothing; one that claimed exec could reach a
    # subprocess it has no permission to run.
    assert {tool.risk for tool in agents_md.plugin.tools} == {"read"}


def test_the_module_cannot_write(agents_md: AgentsMd):
    """A static check of the obvious mistake, so it cannot come back unnoticed."""
    source = (agents_md.plugin.directory / "plugin.py").read_text(encoding="utf-8")
    for forbidden in ('"w"', "'w'", '"a"', "'a'", '"wb"', "os.remove",
                      "os.unlink", "os.mkdir", "os.makedirs", "shutil.rmtree",
                      "open("):
        assert forbidden not in source, f"{forbidden!r} appears in the plugin"


# --- memory_load: precedence ------------------------------------------------


def test_load_returns_both_the_root_and_the_topic_files(agents_md: AgentsMd,
                                                        tmp_path: Path):
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert result["success"]
    assert result["metadata"]["count"] == 3
    assert "Run `pytest tests/ -q`" in result["output"]
    assert "The suite is slow" in result["output"]
    assert "90 columns" in result["output"]


def test_load_returns_them_in_precedence_order(agents_md: AgentsMd, tmp_path: Path):
    """AGENTS.md first, then the numbered topic files in sorted order.

    This is the contract, not an accident of directory listing: a later file
    is the more specific layer and wins where two disagree.
    """
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert result["metadata"]["relative"] == [
        "AGENTS.md",
        ".harness/instructions/10-testing.md",
        ".harness/instructions/20-style.md",
    ]
    assert result["metadata"]["precedence"] == [1, 2, 3]

    output = result["output"]
    assert output.index("Run `pytest tests/ -q`") < output.index("The suite is slow") \
        < output.index("90 columns")


def test_load_states_the_precedence_rule_in_its_header(agents_md: AgentsMd,
                                                       tmp_path: Path):
    output = agents_md.call("memory_load", path=str(tmp_path))["output"]
    assert "precedence order" in output
    assert "more specific layer and win" in output
    # A reader who cannot see the order cannot tell which file won a conflict.
    assert "--- [1] AGENTS.md" in output
    assert "--- [2]" in output
    assert "--- [3]" in output


def test_load_reports_which_files_contributed(agents_md: AgentsMd, tmp_path: Path):
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert result["metadata"]["files"] == [
        str(tmp_path / "AGENTS.md"),
        str(tmp_path / ".harness" / "instructions" / "10-testing.md"),
        str(tmp_path / ".harness" / "instructions" / "20-style.md"),
    ]
    assert result["metadata"]["problems"] == []


def test_load_follows_the_sort_order_not_the_mtime(agents_md: AgentsMd,
                                                   tmp_path: Path):
    """Precedence has to be visible in the repository, not in a timestamp."""
    style = tmp_path / ".harness" / "instructions" / "20-style.md"
    testing = tmp_path / ".harness" / "instructions" / "10-testing.md"
    now = time.time()
    os.utime(style, (now, now))            # the *later* name touched most recently
    os.utime(testing, (now - 3600, now - 3600))
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert result["metadata"]["relative"][1].endswith("10-testing.md")
    assert result["metadata"]["relative"][2].endswith("20-style.md")


def test_load_with_only_a_root_file_still_works(bare: AgentsMd, tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    result = bare.call("memory_load", path=str(tmp_path))
    assert result["metadata"]["count"] == 1
    assert result["metadata"]["relative"] == ["AGENTS.md"]


def test_load_with_only_topic_files_still_works(bare: AgentsMd, tmp_path: Path):
    instructions = tmp_path / ".harness" / "instructions"
    instructions.mkdir(parents=True)
    (instructions / "10-testing.md").write_text(TOPIC_TESTING, encoding="utf-8")
    result = bare.call("memory_load", path=str(tmp_path))
    assert result["metadata"]["relative"] == [".harness/instructions/10-testing.md"]
    # The absence of the root file is the user's business, and memory_check
    # is where it is reported -- not something load invents content for.
    assert result["metadata"]["precedence"] == [1]


def test_load_ignores_non_markdown_files_in_the_instructions_directory(
        bare: AgentsMd, tmp_path: Path):
    instructions = tmp_path / ".harness" / "instructions"
    instructions.mkdir(parents=True)
    (instructions / "notes.txt").write_text("not an instruction", encoding="utf-8")
    (instructions / "10-testing.md").write_text(TOPIC_TESTING, encoding="utf-8")
    result = bare.call("memory_load", path=str(tmp_path))
    assert result["metadata"]["relative"] == [".harness/instructions/10-testing.md"]


def test_load_on_a_workspace_with_nothing_says_so_honestly(bare: AgentsMd,
                                                           tmp_path: Path):
    result = bare.call("memory_load", path=str(tmp_path))
    assert result["success"]
    assert result["metadata"]["count"] == 0
    assert result["metadata"]["files"] == []
    # An empty string would read as instructions that happened to say nothing.
    assert "No project instructions" in result["output"]
    assert "AGENTS.md" in result["output"]
    assert "Nothing was created" in result["output"]


def test_load_on_a_missing_directory_fails_loudly(bare: AgentsMd, tmp_path: Path):
    result = bare.call("memory_load", path=str(tmp_path / "nope"))
    assert result["success"] is False
    assert "Not a directory" in result["error"]


def test_load_reports_when_it_cut_the_text(agents_md: AgentsMd, tmp_path: Path):
    result = agents_md.call("memory_load", path=str(tmp_path), max_chars=500)
    assert result["metadata"]["truncated"] is True
    assert "cut at the max_chars you asked for" in result["output"]


# --- the include chain ------------------------------------------------------


def test_an_include_is_expanded(agents_md: AgentsMd, tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "style.md").write_text("Tabs are forbidden.\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text(
        AGENTS_MD + "\n@docs/style.md\n", encoding="utf-8")
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert "Tabs are forbidden" in result["output"]
    assert "<!-- included from docs/style.md -->" in result["output"]


def test_an_include_is_resolved_against_the_including_file(agents_md: AgentsMd,
                                                           tmp_path: Path):
    instructions = tmp_path / ".harness" / "instructions"
    (instructions / "shared").mkdir()
    (instructions / "shared" / "style.md").write_text("Shared style.\n", encoding="utf-8")
    (instructions / "10-testing.md").write_text(
        TOPIC_TESTING + "\n@shared/style.md\n", encoding="utf-8")
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert "Shared style." in result["output"]


def test_an_include_chain_is_followed_recursively(agents_md: AgentsMd,
                                                  tmp_path: Path):
    (tmp_path / "a.md").write_text("First level.\n@b.md\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("Second level.\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("@a.md\n", encoding="utf-8")
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert "First level." in result["output"]
    assert "Second level." in result["output"]


def test_an_include_cycle_is_reported_not_followed_forever(agents_md: AgentsMd,
                                                           tmp_path: Path):
    (tmp_path / "a.md").write_text("A.\n@b.md\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("B.\n@a.md\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("@a.md\n", encoding="utf-8")
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert result["success"]
    assert "A." in result["output"] and "B." in result["output"]
    assert any("cycle" in problem for problem in result["metadata"]["problems"])


def test_an_include_that_escapes_the_workspace_is_refused(agents_md: AgentsMd,
                                                          tmp_path: Path):
    outside = tmp_path.parent / "outside-secret.md"
    outside.write_text("SECRET", encoding="utf-8")
    try:
        (tmp_path / "AGENTS.md").write_text("@../outside-secret.md\n", encoding="utf-8")
        result = agents_md.call("memory_load", path=str(tmp_path))
        assert "SECRET" not in result["output"]
        assert any("outside the workspace" in problem
                   for problem in result["metadata"]["problems"])
    finally:
        outside.unlink()


def test_an_include_that_does_not_exist_is_reported(agents_md: AgentsMd,
                                                    tmp_path: Path):
    # Appended, not substituted: the point is that the rest of the file, and
    # the rest of the workspace, still load.
    with (tmp_path / "AGENTS.md").open("a", encoding="utf-8") as stream:
        stream.write("\n@nowhere.md\n")
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert result["success"]
    assert any("no such file" in problem for problem in result["metadata"]["problems"])
    assert "Run `pytest tests/ -q`" in result["output"]
    assert result["metadata"]["count"] == 3


def test_a_line_that_is_not_alone_is_not_an_include(agents_md: AgentsMd,
                                                    tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("see @notes.md for more\n", encoding="utf-8")
    result = agents_md.call("memory_load", path=str(tmp_path))
    assert result["metadata"]["problems"] == []
    assert "see @notes.md for more" in result["output"]


# --- memory_check -----------------------------------------------------------


def test_check_reports_what_exists_and_in_what_order(agents_md: AgentsMd,
                                                     tmp_path: Path):
    result = agents_md.call("memory_check", path=str(tmp_path))
    assert result["success"]
    assert result["metadata"]["has_agents_md"] is True
    assert result["metadata"]["has_instructions_dir"] is True
    assert result["metadata"]["count"] == 3
    output = result["output"]
    assert "[1] AGENTS.md" in output
    assert "[2] .harness/instructions/10-testing.md" in output
    assert "precedence order" in output


def test_check_with_no_instructions_prints_the_scaffold(bare: AgentsMd,
                                                        tmp_path: Path):
    result = bare.call("memory_check", path=str(tmp_path))
    assert result["success"]
    assert result["metadata"]["has_agents_md"] is False
    assert result["metadata"]["scaffold_shown"] is True
    output = result["output"]
    assert "No project instructions found" in output
    # A scaffold a user can act on: it says what to fill in, not just that
    # something is missing.
    assert "## Commands" in output
    assert "<the exact command>" in output
    assert "## Conventions" in output


def test_check_says_plainly_that_it_created_nothing(bare: AgentsMd,
                                                    tmp_path: Path):
    """The sentence the whole design turns on."""
    result = bare.call("memory_check", path=str(tmp_path))
    output = result["output"]
    assert "NOTHING WAS CREATED" in output
    assert "did not write a file here" in output
    assert "this tool cannot" in output
    assert result["metadata"]["wrote"] is False
    # And the scaffold is labelled as not written, twice.
    assert output.count("not written") >= 2


def test_check_notes_a_root_file_with_no_topic_files(agents_md: AgentsMd,
                                                     tmp_path: Path):
    (tmp_path / ".harness" / "instructions" / "20-style.md").unlink()
    (tmp_path / ".harness" / "instructions" / "10-testing.md").unlink()
    result = agents_md.call("memory_check", path=str(tmp_path))
    assert result["success"]
    # The directory is still there and is empty; the two cases are told apart
    # because "empty" and "absent" are different situations to fix.
    assert "the .harness/instructions/ directory is empty" in result["output"]


def test_check_notes_a_missing_instructions_directory(bare: AgentsMd,
                                                      tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    result = bare.call("memory_check", path=str(tmp_path))
    assert result["success"]
    assert "there is no .harness/instructions/ directory" in result["output"]


def test_check_notes_topic_files_with_no_root_file(bare: AgentsMd, tmp_path: Path):
    (tmp_path / ".harness" / "instructions").mkdir(parents=True)
    (tmp_path / ".harness" / "instructions" / "10-testing.md").write_text(
        TOPIC_TESTING, encoding="utf-8")
    result = bare.call("memory_check", path=str(tmp_path))
    assert result["success"]
    assert "There is no AGENTS.md" in result["output"]


def test_check_on_a_missing_directory_fails_loudly(bare: AgentsMd, tmp_path: Path):
    result = bare.call("memory_check", path=str(tmp_path / "nope"))
    assert result["success"] is False
    assert "Not a directory" in result["error"]


# --- the property that matters: it never writes -----------------------------


def test_nothing_is_created_in_a_workspace_with_no_instructions(bare: AgentsMd,
                                                                tmp_path: Path):
    """The scaffold is printed, not written. This is the test for that."""
    before = _snapshot(tmp_path)
    assert before == {}

    result = bare.call("memory_check", path=str(tmp_path))
    assert result["success"]
    assert "## Commands" in result["output"]

    after = _snapshot(tmp_path)
    assert after == {}, f"the plugin created files: {sorted(after)}"
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / ".harness").exists()


def test_nothing_is_changed_in_a_workspace_with_instructions(agents_md: AgentsMd,
                                                              tmp_path: Path):
    before = _snapshot(tmp_path)
    assert len(before) == 3

    agents_md.call("memory_check", path=str(tmp_path))
    agents_md.call("memory_load", path=str(tmp_path))
    agents_md.call("memory_load", path=str(tmp_path))
    agents_md.call("memory_check", path=str(tmp_path))

    after = _snapshot(tmp_path)
    assert sorted(after) == sorted(before)
    for key in before:
        assert after[key] == before[key], f"{key} was modified"


def test_calling_both_tools_repeatedly_changes_nothing(agents_md: AgentsMd,
                                                       tmp_path: Path):
    """Idempotent reads: a memory loader that rewrites is not a reader."""
    before = _snapshot(tmp_path)
    for _ in range(5):
        agents_md.call("memory_load", path=str(tmp_path))
        agents_md.call("memory_check", path=str(tmp_path))
    assert _snapshot(tmp_path) == before


def test_a_broken_include_does_not_create_its_target(agents_md: AgentsMd,
                                                     tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("@absent.md\n", encoding="utf-8")
    before = _snapshot(tmp_path)
    agents_md.call("memory_load", path=str(tmp_path))
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "absent.md").exists()
