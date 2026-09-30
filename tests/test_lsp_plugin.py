"""The lsp plugin: language tooling that works with no language server.

The property under test throughout is **honest degradation**. A machine with
no ``pyright``, no ``pylsp`` and no ``node`` must still get useful results from
this plugin, and every one of those results must say what it did and did not
check. A tool that reports "no problems found" when it ran nothing is worse
than a tool that refuses, because the model cannot tell the two apart.

So the fixture here deletes the language servers from ``PATH`` and the tests
still pass. Nothing in this file installs, configures or requires a language
server, and the subprocess seam is replaced rather than reached.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from adaptive_harness.plugins.host import PluginHost
from plugin_paths import plugin_path  # noqa: E402

#: Resolved at import, before any fixture clears PATH. The fixture below
#: deliberately removes every language server from PATH, so a test that wants
#: a real `node` has to have held on to where it was.
REAL_NODE = shutil.which("node")

BROKEN_PY = '''\
def greet(name:
    return "hello " + name
'''

GOOD_PY = '''\
"""A tiny module."""


def greet(name: str) -> str:
    """Return a greeting for `name`."""
    return "hello " + name


class Greeter:
    """Greets people."""

    def __init__(self, prefix: str = "hello") -> None:
        self.prefix = prefix

    def go(self, name: str) -> str:
        return greet(name)
'''

USER_PY = '''\
from helpers import greet


def run(name: str) -> str:
    return greet(name).upper()
'''


class Lsp:
    """A loaded lsp plugin, with its module for reaching the seams."""

    def __init__(self, host: PluginHost) -> None:
        self.host = host
        self.plugin = next(p for p in host.plugins if p.name == "lsp")
        matches = [name for name in sys.modules
                   if name.startswith("_adaptive_plugin_")
                   and self.plugin.directory.name in name]
        assert len(matches) == 1, f"expected one imported module, got {matches}"
        self.module = sys.modules[matches[0]]
        self.tools = {tool.name: tool for tool in self.plugin.tools}

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return self.tools[name].handler(**kwargs)


@pytest.fixture
def lsp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lsp:
    # A workspace with no language server anywhere on PATH. If these tests
    # pass here, the plugin never depended on one.
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    monkeypatch.delenv("ADAPTIVE_HARNESS_LSP_SERVER", raising=False)
    host = PluginHost(project_root=tmp_path)
    host.discovery_roots = lambda: [plugin_path("lsp").parent]
    host.discover()
    return Lsp(host)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "helpers.py").write_text(GOOD_PY, encoding="utf-8")
    (tmp_path / "user.py").write_text(USER_PY, encoding="utf-8")
    return tmp_path


# --- it loads ---------------------------------------------------------------


def test_the_plugin_loads_with_exactly_its_five_tools(lsp: Lsp):
    assert lsp.plugin.ok, lsp.plugin.error
    assert set(lsp.tools) == {"lsp_status", "lsp_diagnostics", "lsp_definitions",
                              "lsp_references", "lsp_hover"}


def test_the_manifest_declares_only_what_it_uses(lsp: Lsp):
    """`subprocess` is for `node --check`, and nothing else by default."""
    assert lsp.plugin.permissions == frozenset({"tools", "subprocess"})
    manifest = json.loads((lsp.plugin.directory / "plugin.plugin.json")
                          .read_text(encoding="utf-8"))
    assert manifest["permissions"] == ["tools", "subprocess"]


def test_every_tool_reads_except_the_one_that_may_spawn_node(lsp: Lsp):
    # Risk drives the strict-profile gate, so a tool that might run a program
    # must not claim to be a read.
    risks = {tool.name: tool.risk for tool in lsp.plugin.tools}
    assert risks["lsp_diagnostics"] == "exec"
    assert all(risk == "read" for name, risk in risks.items() if name != "lsp_diagnostics")


# --- lsp_status: honest about absence ---------------------------------------


def test_status_reports_every_server_as_missing_when_none_is_installed(lsp: Lsp):
    result = lsp.call("lsp_status")
    assert result["success"]
    assert result["metadata"]["any_server_installed"] is False
    assert result["metadata"]["installed"] == []
    # Every known server is named, and named as missing. A status that listed
    # only the working ones would be the report nobody can act on.
    assert len(result["metadata"]["missing"]) == len(lsp.module.SERVERS)
    for server in result["metadata"]["servers"]:
        assert server["installed"] is False
        assert server["path"] == ""
        assert server["install"]


def test_status_names_the_install_command_for_a_missing_server(lsp: Lsp):
    output = lsp.call("lsp_status")["output"]
    assert "NOT INSTALLED" in output
    assert "pip install python-lsp-server" in output
    assert "npm install -g typescript-language-server typescript" in output


def test_status_says_no_server_is_configured_and_will_not_run_one(lsp: Lsp):
    result = lsp.call("lsp_status")
    assert result["metadata"]["configured_server"]["configured"] == ""
    output = result["output"]
    assert "No language server is configured" in output
    # Finding a binary is not permission to run it. This sentence is the
    # difference between "installed" and "used".
    assert "not permission to run it" in output
    assert "ADAPTIVE_HARNESS_LSP_SERVER" in output


def test_status_reports_a_server_that_is_installed(lsp: Lsp, tmp_path: Path):
    """The other half of honest: an installed server is reported as installed."""
    fake = tmp_path / "empty-bin" / "pylsp"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    result = lsp.call("lsp_status")
    assert "pylsp" in result["metadata"]["installed"]
    row = next(r for r in result["metadata"]["servers"] if r["server"] == "pylsp")
    assert row["installed"] is True
    assert row["path"] == str(fake)


def test_status_names_what_works_without_any_server(lsp: Lsp):
    output = lsp.call("lsp_status")["output"]
    assert "What this plugin does without any language server" in output
    assert "lsp_definitions" in output
    assert "a regex hit is not a type check" in output


# --- lsp_diagnostics: a syntax check, and only a syntax check --------------


def test_diagnostics_finds_a_syntax_error_in_a_broken_file(lsp: Lsp, tmp_path: Path):
    broken = tmp_path / "broken.py"
    broken.write_text(BROKEN_PY, encoding="utf-8")
    result = lsp.call("lsp_diagnostics", path=str(broken))

    assert result["success"]
    assert result["metadata"]["checked"] is True
    assert result["metadata"]["clean"] is False
    assert len(result["metadata"]["diagnostics"]) == 1
    problem = result["metadata"]["diagnostics"][0]
    assert problem["line"] == 1
    assert "greet" in result["output"]
    assert "1 problem(s)" in result["output"]


def test_diagnostics_finds_nothing_in_a_good_file(lsp: Lsp, tmp_path: Path):
    good = tmp_path / "good.py"
    good.write_text(GOOD_PY, encoding="utf-8")
    result = lsp.call("lsp_diagnostics", path=str(good))

    assert result["success"]
    assert result["metadata"]["clean"] is True
    assert result["metadata"]["diagnostics"] == []
    assert "No problems found" in result["output"]


def test_a_clean_result_states_what_was_not_checked(lsp: Lsp, tmp_path: Path):
    """A syntax check must not read as a clean bill of health."""
    good = tmp_path / "good.py"
    good.write_text(GOOD_PY, encoding="utf-8")
    output = lsp.call("lsp_diagnostics", path=str(good))["output"]
    assert "not a type check" in output
    assert "not a language server" in output


def test_diagnostics_uses_no_subprocess_for_python(lsp: Lsp, tmp_path: Path):
    """Python is parsed in process, so nothing is spawned and nothing is written."""
    calls: list = []
    lsp.module._RUNNER = lambda *a, **k: calls.append(a)  # type: ignore[attr-defined]
    try:
        good = tmp_path / "good.py"
        good.write_text(GOOD_PY, encoding="utf-8")
        result = lsp.call("lsp_diagnostics", path=str(good))
    finally:
        lsp.module._RUNNER = __import__("subprocess").run  # type: ignore[attr-defined]
    assert result["metadata"]["used_subprocess"] is False
    assert calls == []
    # No __pycache__: `python -m py_compile` would have written one.
    assert list(tmp_path.glob("__pycache__")) == []


def test_diagnostics_on_javascript_says_it_was_not_checked_when_node_is_absent(
        lsp: Lsp, tmp_path: Path):
    """The distinction that matters: not checked is not clean."""
    script = tmp_path / "app.js"
    script.write_text("function f( { return 1; }\n", encoding="utf-8")
    result = lsp.call("lsp_diagnostics", path=str(script))

    assert result["success"]
    assert result["metadata"]["checked"] is False
    assert result["metadata"]["clean"] is False
    assert "NOT checked" in result["output"]
    assert "not checked is not the same as clean" in result["output"]


def test_diagnostics_on_javascript_uses_node_when_it_is_present(
        lsp: Lsp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import os
    import stat

    fake_bin = tmp_path / "empty-bin"
    node = fake_bin / "node"
    node.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    node.chmod(node.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(fake_bin))
    os.environ["PATH"] = str(fake_bin)

    script = tmp_path / "app.js"
    script.write_text("function f( { return 1; }\n", encoding="utf-8")
    result = lsp.call("lsp_diagnostics", path=str(script))
    assert result["metadata"]["checker"] == "node --check"
    assert result["metadata"]["used_subprocess"] is True


@pytest.mark.skipif(REAL_NODE is None, reason="node is not installed")
def test_diagnostics_reports_a_real_javascript_syntax_error(lsp: Lsp, tmp_path: Path,
                                                            monkeypatch):
    """With a real node on PATH, a real error is found and reported."""
    import os
    import stat

    real = REAL_NODE
    fake_bin = tmp_path / "empty-bin"
    link = fake_bin / "node"
    link.symlink_to(real)
    link.chmod(link.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(fake_bin))
    os.environ["PATH"] = str(fake_bin)

    # `.mjs` rather than `.js`: Node classifies a plain .js file itself, and
    # some errors in a .js are not reported because of that. See the caveat
    # the tool emits.
    script = tmp_path / "bad.mjs"
    script.write_text("export function greet( { return 1; }\n", encoding="utf-8")
    result = lsp.call("lsp_diagnostics", path=str(script))
    assert result["metadata"]["checked"] is True
    assert result["metadata"]["clean"] is False
    assert result["metadata"]["diagnostics"][0]["message"]
    assert "1 problem(s)" in result["output"]


@pytest.mark.skipif(REAL_NODE is None, reason="node is not installed")
def test_diagnostics_warns_that_node_classifies_a_js_file_itself(
        lsp: Lsp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A clean verdict from node --check is only as good as its classification."""
    import os
    import stat

    real = REAL_NODE
    fake_bin = tmp_path / "empty-bin"
    link = fake_bin / "node"
    link.symlink_to(real)
    link.chmod(link.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(fake_bin))
    os.environ["PATH"] = str(fake_bin)

    good = tmp_path / "app.js"
    good.write_text("export function greet(n) {\n  return n;\n}\n", encoding="utf-8")
    result = lsp.call("lsp_diagnostics", path=str(good))
    assert result["metadata"]["clean"] is True
    assert "node --check classified this file itself" in result["output"]
    assert ".mjs" in result["output"]


def test_diagnostics_on_an_unknown_language_says_it_ran_nothing(
        lsp: Lsp, tmp_path: Path):
    rust = tmp_path / "main.rs"
    rust.write_text("fn main() { let x = ; }\n", encoding="utf-8")
    result = lsp.call("lsp_diagnostics", path=str(rust))

    assert result["success"]
    assert result["metadata"]["checked"] is False
    assert "no parser for .rs is available" in result["output"]


def test_diagnostics_on_a_missing_file_fails_loudly(lsp: Lsp, tmp_path: Path):
    result = lsp.call("lsp_diagnostics", path=str(tmp_path / "nope.py"))
    assert result["success"] is False
    assert "No such file" in result["error"]


# --- lsp_definitions: works with no language server -------------------------


def test_definitions_finds_a_function_in_the_file_it_was_given(lsp: Lsp, project: Path):
    result = lsp.call("lsp_definitions", path=str(project / "helpers.py"),
                      symbol="greet")
    assert result["success"]
    assert result["metadata"]["count"] >= 1
    names = {d["signature"] for d in result["metadata"]["definitions"]}
    assert any("greet" in name for name in names)
    # The AST gives the signature and the docstring, which a grep could not.
    hit = next(d for d in result["metadata"]["definitions"] if "def greet" in d["signature"])
    assert hit["docstring"].startswith("Return a greeting")
    assert hit["kind"] == "function"


def test_definitions_finds_a_class_with_its_line(lsp: Lsp, project: Path):
    result = lsp.call("lsp_definitions", path=str(project / "helpers.py"),
                      symbol="Greeter")
    hit = next(d for d in result["metadata"]["definitions"] if d["kind"] == "class")
    assert hit["file"] == "helpers.py"
    assert hit["line"] > 0
    assert hit["docstring"] == "Greets people."


def test_definitions_finds_a_symbol_in_another_file_of_the_workspace(
        lsp: Lsp, project: Path):
    """The search is a workspace search, not a single-file lookup."""
    result = lsp.call("lsp_definitions", path=str(project / "user.py"),
                      symbol="greet")
    files = {d["file"] for d in result["metadata"]["definitions"]}
    assert "helpers.py" in files


def test_definitions_respects_an_explicit_directory(lsp: Lsp, project: Path):
    elsewhere = project / "pkg"
    elsewhere.mkdir()
    (elsewhere / "extra.py").write_text("def greet(x):\n    return x\n", encoding="utf-8")
    narrowed = lsp.call("lsp_definitions", path=str(project / "helpers.py"),
                        symbol="greet", directory=str(elsewhere))
    assert {d["file"] for d in narrowed["metadata"]["definitions"]} == {"extra.py"}


def test_definitions_of_an_unknown_symbol_says_what_not_found_means(lsp: Lsp, project: Path):
    result = lsp.call("lsp_definitions", path=str(project / "helpers.py"),
                      symbol="no_such_symbol_anywhere")
    assert result["success"]
    assert result["metadata"]["count"] == 0
    assert "None found" in result["output"]
    # Not found by a parser is not proven absent. That sentence is the point.
    assert "not proven absent" in result["output"]


def test_definitions_uses_the_ast_for_python_and_says_so(lsp: Lsp, project: Path):
    result = lsp.call("lsp_definitions", path=str(project / "helpers.py"), symbol="greet")
    assert result["metadata"]["used_ast"] is True
    assert "AST parse" in result["output"]


def test_definitions_skips_a_file_it_cannot_parse_and_counts_it(lsp: Lsp, project: Path):
    (project / "broken.py").write_text(BROKEN_PY, encoding="utf-8")
    result = lsp.call("lsp_definitions", path=str(project / "helpers.py"), symbol="greet")
    assert result["metadata"]["unparseable"] >= 1
    assert "could not be parsed" in result["output"]
    # And the good files were still searched.
    assert result["metadata"]["count"] >= 1


def test_definitions_skips_vendored_directories(lsp: Lsp, project: Path):
    vendored = project / "node_modules" / "leftpad"
    vendored.mkdir(parents=True)
    (vendored / "index.js").write_text("function greet() {}\n", encoding="utf-8")
    result = lsp.call("lsp_definitions", path=str(project / "user.py"), symbol="greet")
    files = {d["file"] for d in result["metadata"]["definitions"]}
    assert not any("node_modules" in name for name in files)


def test_references_for_javascript_searches_the_whole_language(lsp: Lsp,
                                                                tmp_path: Path):
    """A symbol crosses extensions within a language, not within one suffix.

    A `.mjs` caller of a `.js` definition is exactly what a search should find;
    restricting the walk to the one suffix of the file that was opened would
    report "not referenced anywhere" while the reference sits in the next file.
    """
    (tmp_path / "lib.js").write_text("export function greet(n) {\n  return n;\n}\n",
                                     encoding="utf-8")
    (tmp_path / "caller.mjs").write_text("import { greet } from './lib.js';\n"
                                         "greet('x');\n", encoding="utf-8")
    result = lsp.call("lsp_references", path=str(tmp_path / "lib.js"), symbol="greet")
    files = {r["file"] for r in result["metadata"]["references"]}
    assert "caller.mjs" in files


def test_definitions_needs_no_subprocess(lsp: Lsp, project: Path):
    calls: list = []
    lsp.module._RUNNER = lambda *a, **k: calls.append(a)  # type: ignore[attr-defined]
    try:
        result = lsp.call("lsp_definitions", path=str(project / "helpers.py"),
                          symbol="greet")
    finally:
        lsp.module._RUNNER = __import__("subprocess").run  # type: ignore[attr-defined]
    assert result["metadata"]["count"] >= 1
    assert calls == []


# --- non-python search: honest about being a regex --------------------------


def test_definitions_for_javascript_are_reported_as_unconfirmed(
        lsp: Lsp, tmp_path: Path):
    script = tmp_path / "app.js"
    script.write_text("export function greet(name) {\n  return name;\n}\n",
                      encoding="utf-8")
    result = lsp.call("lsp_definitions", path=str(script), symbol="greet")
    assert result["metadata"]["used_ast"] is False
    hit = result["metadata"]["definitions"][0]
    assert "unconfirmed" in hit["kind"]
    assert result["metadata"]["count"] == 1


# --- lsp_references ---------------------------------------------------------


def test_references_finds_the_uses_of_a_symbol(lsp: Lsp, project: Path):
    result = lsp.call("lsp_references", path=str(project / "helpers.py"),
                      symbol="greet")
    assert result["success"]
    assert result["metadata"]["count"] >= 2
    sources = [r["source"] for r in result["metadata"]["references"]]
    assert any("from helpers import greet" in s for s in sources)
    assert any("return greet(name)" in s for s in sources)


def test_references_crosses_files(lsp: Lsp, project: Path):
    result = lsp.call("lsp_references", path=str(project / "user.py"), symbol="greet")
    files = {r["file"] for r in result["metadata"]["references"]}
    assert {"user.py", "helpers.py"} <= files


def test_references_counts_the_limit_and_says_it_cut(lsp: Lsp, project: Path):
    body = "\n".join(f"x{greet} = greet({greet})" for greet in range(0, 20))
    (project / "many.py").write_text(f"greet = 1\n{body}\n", encoding="utf-8")
    result = lsp.call("lsp_references", path=str(project / "many.py"),
                      symbol="greet", limit=3)
    assert result["metadata"]["shown"] == 3
    assert result["metadata"]["count"] > 3
    assert "showing the first 3 of" in result["output"]


def test_references_of_nothing_says_it_is_a_lower_bound(lsp: Lsp, project: Path):
    result = lsp.call("lsp_references", path=str(project / "helpers.py"),
                      symbol="absent_symbol")
    assert result["success"]
    assert result["metadata"]["count"] == 0
    assert "would not be seen" in result["output"]


# --- lsp_hover --------------------------------------------------------------


def test_hover_describes_the_enclosing_function(lsp: Lsp, project: Path):
    lines = GOOD_PY.splitlines()
    line = next(i for i, text in enumerate(lines, start=1) if "return greet(name)" in text)
    result = lsp.call("lsp_hover", path=str(project / "helpers.py"), line=line,
                      symbol="greet")
    assert result["success"]
    assert result["metadata"]["enclosing"]["kind"] == "function"
    assert "def go(self, name: str) -> str" in result["metadata"]["enclosing"]["signature"]
    assert "return greet(name)" in result["output"]


def test_hover_includes_the_docstring(lsp: Lsp, project: Path):
    lines = GOOD_PY.splitlines()
    line = next(i for i, text in enumerate(lines, start=1) if 'return "hello "' in text)
    result = lsp.call("lsp_hover", path=str(project / "helpers.py"), line=line)
    assert "Return a greeting for" in result["output"]


def test_hover_shows_the_comment_block_above_a_line(lsp: Lsp, tmp_path: Path):
    script = tmp_path / "app.js"
    script.write_text("// adds one\n// keeps it integer\nfunction inc(x) {\n  return x + 1;\n}\n",
                      encoding="utf-8")
    result = lsp.call("lsp_hover", path=str(script), line=3, symbol="inc")
    assert "keeps it integer" in result["output"]
    assert "function inc(x) {" in result["output"]


def test_hover_says_when_the_symbol_is_not_on_the_line(lsp: Lsp, project: Path):
    """Describing a line is not the same as describing the symbol you guessed."""
    result = lsp.call("lsp_hover", path=str(project / "helpers.py"), line=1,
                      symbol="not_here")
    assert result["success"]
    assert "does not appear on line 1" in result["output"]


def test_hover_says_it_infers_nothing(lsp: Lsp, project: Path):
    output = lsp.call("lsp_hover", path=str(project / "helpers.py"), line=5)["output"]
    assert "No type was inferred" in output


def test_hover_past_the_end_of_the_file_fails_loudly(lsp: Lsp, project: Path):
    result = lsp.call("lsp_hover", path=str(project / "helpers.py"), line=9999)
    assert result["success"] is False
    assert "past the end" in result["error"]


def test_hover_reports_an_unparseable_file_instead_of_inventing_context(
        lsp: Lsp, tmp_path: Path):
    broken = tmp_path / "broken.py"
    broken.write_text(BROKEN_PY, encoding="utf-8")
    result = lsp.call("lsp_hover", path=str(broken), line=1)
    assert result["success"]
    assert "does not parse" in result["output"]


# --- the shape of an absence ------------------------------------------------


def test_no_tool_ever_claims_a_server_ran(lsp: Lsp, project: Path):
    """The whole plugin, called end to end, names no server that did not run."""
    calls = [
        lsp.call("lsp_status"),
        lsp.call("lsp_diagnostics", path=str(project / "helpers.py")),
        lsp.call("lsp_definitions", path=str(project / "helpers.py"), symbol="greet"),
        lsp.call("lsp_references", path=str(project / "helpers.py"), symbol="greet"),
        lsp.call("lsp_hover", path=str(project / "helpers.py"), line=6, symbol="greet"),
    ]
    for result in calls:
        assert result["success"], result
        assert result["metadata"].get("configured_server", {}).get("configured", "") == "" \
            if "configured_server" in result.get("metadata", {}) else True
    # And nothing anywhere in the output names a language server as having run.
    for result in calls:
        assert "pylsp did" not in result["output"]
        assert "pyright found" not in result["output"]
