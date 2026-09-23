"""Interactive clarification tool for querying the developer in the middle of development."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from adaptive_harness.tools.base import Tool, ToolResult


class AskUserTool(Tool):
    """Halts execution to ask the developer a targeted question during development."""

    name = "ask_user"
    description = (
        "Asks the user a clarifying question when requirements are ambiguous, "
        "multiple conflicting approaches exist, or approval is required."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The specific question to ask the user.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of multiple choice options for quick user selection.",
            },
            "context": {
                "type": "string",
                "description": "Brief explanation of why clarification is needed.",
            },
        },
        "required": ["question"],
    }

    def __init__(self, callback: Optional[Callable[[str, Optional[List[str]], Optional[str]], str]] = None):
        self.callback = callback

    def execute(
        self,
        question: str,
        options: Optional[List[str]] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        if self.callback is not None:
            # Delegate to UI modal / interactive handler
            answer = self.callback(question, options, context)
            return ToolResult(
                success=True,
                output=f"User responded: {answer}",
                metadata={"question": question, "options": options, "answer": answer},
            )

        # Fallback for headless CLI: prompt terminal
        print("\n" + "=" * 60)
        print(f"[AGENT ASKS A QUESTION]: {question}")
        if context:
            print(f"Context: {context}")
        if options:
            print("Options:")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
        print("=" * 60)

        try:
            user_input = input("Your answer (or press enter for default): ").strip()
            answer = user_input or (options[0] if options else "Proceed with recommended approach")
        except (EOFError, KeyboardInterrupt):
            answer = options[0] if options else "Approved"

        return ToolResult(
            success=True,
            output=f"User responded: {answer}",
            metadata={"question": question, "options": options, "answer": answer},
        )
