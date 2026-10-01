"""The MCP host, against a real server subprocess.

An MCP server is a process speaking JSON-RPC over stdio. A mock cannot tell you
whether the framing, the initialize handshake, the required notification, the
tool-listing round trip, or the shutdown path are right — the whole risk in this
module is in the transport. So these tests run an actual server: a stdlib script
under `tests/fixtures/` that behaves like a real one, including printing a
banner to stderr, which a real server may do and a client must tolerate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from adaptive_harness.plugins.host import PluginHost
from adaptive_harness.plugins.mcp import McpError, McpHost, _render_content
from adaptive_harness.plugins.types import McpServerSpec

SERVER = Path(__file__).resolve().parent / "fixtures" / "mcp_server.py"


def _spec(**overrides) -> McpServerSpec:
    return McpServerSpec(name="test", command=[sys.executable, str(SERVER)], **overrides)


@pytest.fixture
def host():
    """A started host, always stopped afterwards."""
    instance = McpHost()
    yield instance
    instance.stop_all()


def test_a_real_server_completes_the_handshake_and_lists_its_tools(host: McpHost):
    host.start(_spec())
    assert host.errors == [], host.errors
    assert sorted(host.connections["test"].tools) == ["add", "boom", "echo"]


def test_a_remote_tool_becomes_a_callable_with_the_usual_result_shape(host: McpHost):
    host.start(_spec())
    handlers = host.build_handlers()
    assert sorted(handlers) == ["test__add", "test__boom", "test__echo"]

    result = handlers["test__echo"](text="hello")
    assert result["success"] is True
    assert result["output"] == "hello"
    assert result["metadata"] == {"server": "test", "tool": "echo"}


def test_arguments_reach_the_server_verbatim(host: McpHost):
    host.start(_spec())
    assert host.build_handlers()["test__add"](a=2, b=3)["output"] == "5"


def test_a_tool_that_fails_is_reported_as_a_failure_not_a_crash(host: McpHost):
    host.start(_spec())
    result = host.build_handlers()["test__boom"]()
    assert result["success"] is False
    assert "always fails" in result["output"]


def test_calling_a_tool_the_server_does_not_offer_is_an_error(host: McpHost):
    host.start(_spec())
    connection = host.connections["test"]
    with pytest.raises(McpError):
        connection.call("not_a_tool", {})


def test_a_server_that_cannot_start_is_skipped_and_reported(tmp_path: Path):
    instance = McpHost()
    instance.start(McpServerSpec(name="broken", command=["/nonexistent/binary"]))
    assert instance.skipped == ["broken"]
    assert instance.errors and "broken" in instance.errors[0]
    assert instance.connections == {}
    instance.stop_all()


def test_a_server_that_exits_immediately_is_reported_not_hung(tmp_path: Path):
    """A server that dies during the handshake must time out or error, never
    leave the harness waiting forever."""
    failing = tmp_path / "dies.py"
    failing.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    instance = McpHost()
    instance.start(McpServerSpec(name="dies", command=[sys.executable, str(failing)],
                                 startup_timeout_s=5.0))
    assert "dies" in instance.skipped
    assert instance.errors
    instance.stop_all()


def test_stopping_twice_is_safe(host: McpHost):
    host.start(_spec())
    host.stop_all()
    host.stop_all()
    assert host.connections == {}


def test_a_stale_connection_reports_that_it_is_not_running(host: McpHost):
    host.start(_spec())
    connection = host.connections["test"]
    connection.stop()
    with pytest.raises(McpError):
        connection.call("echo", {"text": "x"})


@pytest.mark.parametrize("block,expected", [
    ({"type": "text", "text": "hello"}, "hello"),
    ({"type": "image", "data": "AAAA"}, "image content omitted"),
    ({"type": "resource", "resource": {"uri": "file://x", "text": "body"}},
     "[resource file://x: body]"),
    ({"type": "something-new"}, "[something-new content]"),
])
def test_content_blocks_flatten_to_something_the_model_can_read(block, expected):
    """Nothing is silently dropped: a resource or an image is described rather
    than vanishing from the transcript."""
    assert expected in _render_content({"content": [block]})


def test_a_result_with_no_content_falls_back_to_structured_output():
    assert _render_content({"structuredContent": {"answer": 42}}) == '{"answer": 42}'
    assert _render_content({}) == "(no content)"


def test_a_remote_tool_reaches_the_agent_through_the_existing_adapter(host: McpHost,
                                                                      tmp_path: Path):
    """The whole point: an MCP tool is an ordinary tool. It is wrapped by
    PluginToolAdapter and needs no special case in the agent loop."""
    host.start(_spec())
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    manifest = {
        "name": "mcp-demo", "version": "1.0",
        "permissions": ["tools", "subprocess", "net"],
        "mcp": [{"name": "test", "command": [sys.executable, str(SERVER)],
                 "startup_timeout_s": 15.0}],
    }
    plugin_host = PluginHost(project_root=tmp_path, allow_project_plugins=True)
    directory = tmp_path / ".harness" / "plugins" / "mcp-demo"
    directory.mkdir(parents=True)
    (directory / "plugin.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    plugin_host.discovery_roots = lambda: [tmp_path / ".harness" / "plugins"]
    plugin_host.discover()
    assert plugin_host.mcp_servers()[0].risk == "net"

    # Supervise the declared server and expose its tools through the adapter.
    host.start(plugin_host.mcp_servers()[0], workspace_root=tmp_path)
    assert "test__echo" in host.build_handlers()
    host.stop_all()
