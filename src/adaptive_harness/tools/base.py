"""Base interface and schema converter for agent developer tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Outcome produced by executing an agent tool."""

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Abstract developer tool invokable by the agent."""

    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Executes the tool with the provided arguments."""
        ...

    def to_openai_schema(self) -> Dict[str, Any]:
        """Converts tool specification to OpenAI / OpenRouter function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def workspace_path(root: Path, value: str) -> Path:
    """Resolve a file path inside the selected workspace, including symlinks."""
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Path is outside the workspace: {value}")
    return candidate
