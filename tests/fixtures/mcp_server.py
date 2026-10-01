"""A real MCP server, in stdlib only, used to exercise the host end to end.

This is an actual server speaking JSON-RPC 2.0 over stdio -- not a mock. The
point of the MCP host is that it supervises a *subprocess*, and a mock cannot
tell you whether the framing, the handshake, the notification, or the shutdown
path is right. Running this for real is the only honest test.

It is used by the tests and by the bundled demo plugin, never by the harness at
runtime.
"""

from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "echo",
        "description": "Return the text it was given.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string",
                                    "description": "What to echo back."}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "boom",
        "description": "Always fails, so error handling is exercised.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def reply(message_id, result=None, error=None) -> dict:
    payload = {"jsonrpc": "2.0", "id": message_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def call_tool(name: str, arguments: dict) -> dict:
    if name == "echo":
        return {"content": [{"type": "text", "text": str(arguments.get("text", ""))}]}
    if name == "add":
        total = float(arguments.get("a", 0)) + float(arguments.get("b", 0))
        # An integer sum should not come back as "4.0".
        text = str(int(total)) if total == int(total) else str(total)
        return {"content": [{"type": "text", "text": text}]}
    if name == "boom":
        return {"content": [{"type": "text", "text": "this tool always fails"}],
                "isError": True}
    raise KeyError(name)


def main() -> int:
    # A banner on stdout would be a protocol violation, so it goes to stderr --
    # which also proves the client tolerates a noisy server.
    print("mcp test server starting", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue

        method = message.get("method")
        message_id = message.get("id")

        if method == "initialize":
            print(json.dumps(reply(message_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-server", "version": "1.0.0"},
            })), flush=True)
        elif method == "notifications/initialized":
            continue  # a notification; no reply
        elif method == "tools/list":
            print(json.dumps(reply(message_id, {"tools": TOOLS})), flush=True)
        elif method == "tools/call":
            params = message.get("params", {})
            try:
                result = call_tool(params.get("name", ""), params.get("arguments", {}))
            except KeyError as exc:
                print(json.dumps(reply(message_id, error={
                    "code": -32601, "message": f"Unknown tool: {exc}"})), flush=True)
                continue
            print(json.dumps(reply(message_id, result)), flush=True)
        elif message_id is not None:
            print(json.dumps(reply(message_id, error={
                "code": -32601, "message": f"Unknown method: {method}"})), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
