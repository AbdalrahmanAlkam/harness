"""A plugin-contributed tool, presented to the agent like any built-in.

The adapter exists so a plugin's tool is indistinguishable from a built-in at
the call site: same ``Tool`` interface, same schema, same result type. The core
does not branch on where a tool came from, which is what keeps the plugin
surface from leaking special cases into the agent loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_harness.tools.base import Tool, ToolResult, workspace_path


class PluginToolAdapter(Tool):
    """Wraps a :class:`~adaptive_harness.plugins.host.PluginTool` as a `Tool`."""

    def __init__(self, spec: Any, plugin: Any, workspace_root: Path | str) -> None:
        self.spec = spec
        self.plugin = plugin
        self.workspace_root = Path(workspace_root).resolve()
        self.name = spec.name
        self.description = (f"[{plugin.name}] {spec.description}" if spec.description
                            else f"A tool provided by the {plugin.name} plugin.")
        self.parameters = spec.parameters

    def execute(self, **kwargs: Any) -> ToolResult:
        # A plugin receives only arguments that resolve inside the workspace.
        # This is not a sandbox -- a plugin with the `subprocess` permission can
        # do anything -- it is the same containment the built-in file tools
        # apply, so a path argument cannot silently walk out of the project.
        try:
            self._confine_paths(kwargs)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        try:
            outcome = self.spec.handler(**kwargs)
        except Exception as exc:  # noqa: BLE001 - a plugin's failure is its own
            return ToolResult(success=False, output="",
                              error=f"{self.plugin.name}.{self.name} failed: "
                                    f"{type(exc).__name__}: {exc}")
        if isinstance(outcome, ToolResult):
            return outcome
        if isinstance(outcome, dict):
            return ToolResult(success=bool(outcome.get("success", True)),
                              output=str(outcome.get("output", outcome.get("text", ""))),
                              error=outcome.get("error"),
                              # A plugin that omits metadata, or sets it to null,
                              # is normal; only a non-dict is a mistake worth
                              # rejecting the result over.
                              metadata=outcome.get("metadata") or {})
        return ToolResult(success=True, output="" if outcome is None else str(outcome))

    def _confine_paths(self, arguments: dict[str, Any]) -> None:
        """Resolve path arguments against the workspace, refusing escapes.

        The value is passed on as an absolute path inside the workspace, which
        is what a plugin's handler will actually open. A plugin that reaches for
        the raw relative string would otherwise resolve against whatever
        directory it happened to be imported in.
        """
        for key in ("path", "file_path", "directory", "target", "filename"):
            value = arguments.get(key)
            if not isinstance(value, str) or not value:
                continue
            arguments[key] = str(workspace_path(self.workspace_root, value))

    @property
    def risk(self) -> str:
        """The risk the plugin declared, read by the strict-profile gate."""
        return self.spec.risk
