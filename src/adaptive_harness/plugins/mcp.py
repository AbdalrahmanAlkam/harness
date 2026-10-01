"""MCP server supervision.

An MCP server is a subprocess speaking JSON-RPC 2.0 over stdio. This module owns
its lifecycle -- start, initialize, list tools, call tool, shut down -- and hands
each discovered tool to the *existing* :class:`PluginToolAdapter`, so a remote
tool reaches the agent loop through exactly the same path as a local one. That is
why the agent loop needs no knowledge of MCP at all.

No third-party library: the core has no MCP dependency, and adding one for a
newline-delimited JSON protocol would be a poor trade. Everything here is stdlib.

**Tools are network-facing.** A remote tool's data crosses a process boundary to
code the user did not write, so its risk is pinned to ``net`` and the manifest
cannot downgrade it. That is the single most important property of this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from adaptive_harness.plugins.types import McpServerSpec

#: JSON-RPC protocol version this client speaks.
PROTOCOL_VERSION = "2024-11-05"

CLIENT_INFO = {"name": "adaptive-harness", "version": "2.0.0"}


@dataclass
class RemoteTool:
    """One tool an MCP server advertises."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def as_parameters(self) -> Dict[str, Any]:
        """Present the remote schema in the shape the model already expects."""
        schema = dict(self.input_schema or {})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema


class McpError(RuntimeError):
    """A server failed to start, misbehaved, or returned an error."""


class McpServerConnection:
    """One supervised MCP server process."""

    def __init__(self, spec: McpServerSpec) -> None:
        self.spec = spec
        self.process: Optional[subprocess.Popen[str]] = None
        self.tools: Dict[str, RemoteTool] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._stderr: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Launch the server and complete the MCP initialize handshake."""
        environment = os.environ.copy()
        environment.update(self.spec.env)
        try:
            self.process = subprocess.Popen(
                self.spec.command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=environment,
                cwd=self.spec.cwd or None, start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise McpError(f"Could not start MCP server {self.spec.name!r}: {exc}") from exc

        response = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        }, timeout=self.spec.startup_timeout_s)
        if "error" in response:
            raise McpError(
                f"MCP server {self.spec.name!r} refused to initialize: "
                f"{response['error'].get('message', response['error'])}")
        # The notification is required by the protocol; servers wait for it
        # before serving requests.
        self._notify("notifications/initialized", {})
        self._load_tools()

    def _load_tools(self) -> None:
        response = self._request("tools/list", {}, timeout=self.spec.startup_timeout_s)
        if "error" in response:
            raise McpError(
                f"MCP server {self.spec.name!r} could not list tools: "
                f"{response['error'].get('message', response['error'])}")
        for entry in response.get("result", {}).get("tools", []):
            name = entry.get("name")
            if not name:
                continue
            self.tools[name] = RemoteTool(
                name=name,
                description=entry.get("description", ""),
                input_schema=entry.get("inputSchema") or entry.get("input_schema") or {},
            )

    def call(self, tool_name: str, arguments: Dict[str, Any],
             timeout: float = 60.0) -> Dict[str, Any]:
        """Invoke a remote tool and return its result payload."""
        if self.process is None or self.process.poll() is not None:
            raise McpError(f"MCP server {self.spec.name!r} is not running")
        if tool_name not in self.tools:
            raise McpError(f"MCP server {self.spec.name!r} does not offer {tool_name!r}")
        response = self._request("tools/call",
                                 {"name": tool_name, "arguments": arguments},
                                 timeout=timeout)
        if "error" in response:
            raise McpError(f"{self.spec.name}.{tool_name} failed: "
                           f"{response['error'].get('message', response['error'])}")
        return response.get("result", {})

    def stop(self) -> None:
        """Shut the server down, escalating to a kill if it does not exit."""
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        finally:
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except OSError:
                    pass
            self.process = None

    # -- JSON-RPC ----------------------------------------------------------

    def _next_request_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _write(self, payload: Dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise McpError("MCP server is not running")
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise McpError(f"MCP server {self.spec.name!r} closed its input: {exc}") from exc

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _read(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Read one JSON line, or None on timeout."""
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except ValueError:
                # A server may print banners; skip anything that is not JSON
                # rather than treating noise as a protocol failure.
                continue
        return None

    def _request(self, method: str, params: Dict[str, Any],
                 timeout: float) -> Dict[str, Any]:
        request_id = self._next_request_id()
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self._read(max(0.1, deadline - time.monotonic()))
            if message is None:
                break
            if message.get("id") != request_id:
                continue  # a notification or a late reply to something else
            return message
        raise McpError(f"MCP server {self.spec.name!r} did not answer {method!r} "
                       f"within {timeout:g}s")


class McpHost:
    """Supervises every MCP server a plugin declared.

    A server that fails to start is reported and skipped. One broken integration
    must not take down the harness, and it certainly must not silently look like
    a working one.
    """

    def __init__(self) -> None:
        self.connections: Dict[str, McpServerConnection] = {}
        self.errors: List[str] = []
        self.skipped: List[str] = []

    def start_all(self, specs, workspace_root: Path | str | None = None) -> None:
        for spec in specs:
            self.start(spec, workspace_root)

    def start(self, spec: McpServerSpec, workspace_root: Path | str | None = None) -> None:
        # A server's command is plugin-supplied, so it is confined to the
        # workspace the same way a tool's path arguments are.
        if workspace_root is not None and spec.cwd:
            from adaptive_harness.tools.base import workspace_path

            spec.cwd = str(workspace_path(Path(workspace_root), spec.cwd))
        connection = McpServerConnection(spec)
        try:
            connection.start()
        except McpError as exc:
            self.errors.append(str(exc))
            self.skipped.append(spec.name)
            return
        self.connections[spec.name] = connection

    def stop_all(self) -> None:
        for connection in list(self.connections.values()):
            try:
                connection.stop()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass
        self.connections.clear()

    def build_handlers(self) -> Dict[str, Callable[..., Any]]:
        """A ``name -> callable`` map for every discovered remote tool.

        The returned callables accept the same arguments the model sends and
        return the same result shape a local tool does, so the adapter wraps
        them with no special-casing.
        """
        handlers: Dict[str, Callable[..., Any]] = {}
        for server_name, connection in self.connections.items():
            for tool_name in connection.tools:
                qualified = f"{server_name}__{tool_name}"
                handlers[qualified] = self._make_handler(connection, tool_name, qualified)
        return handlers

    @staticmethod
    def _make_handler(connection: McpServerConnection, tool_name: str,
                      qualified: str) -> Callable[..., Any]:
        def invoke(**kwargs: Any) -> Dict[str, Any]:
            try:
                result = connection.call(tool_name, kwargs)
            except McpError as exc:
                return {"success": False, "error": str(exc)}
            return {"success": not result.get("isError", False),
                    "output": _render_content(result),
                    "metadata": {"server": connection.spec.name, "tool": tool_name}}
        invoke.__name__ = qualified
        return invoke


def _render_content(result: Dict[str, Any]) -> str:
    """Flatten an MCP tool result into text the model can read.

    MCP returns a list of typed content blocks. Text is passed through; anything
    else is described rather than dropped, so a resource or an image is visible
    to the model as "there was something here" instead of vanishing.
    """
    parts: List[str] = []
    for block in result.get("content", []) or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind in {"image", "audio"}:
            parts.append(f"[{kind} content omitted: {len(str(block.get('data', '')))} bytes]")
        elif kind == "resource":
            resource = block.get("resource", {})
            parts.append(f"[resource {resource.get('uri', '?')}: "
                         f"{str(resource.get('text', ''))[:500]}]")
        else:
            parts.append(f"[{kind} content]")
    if not parts:
        structured = result.get("structuredContent")
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False, default=str)
        return "(no content)"
    return "\n".join(parts)
