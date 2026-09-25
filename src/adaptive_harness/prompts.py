"""Central registry of every prompt and prompt injection sent to models.

All model-facing text lives here as editable templates, so prompts can be
inspected and maintained in one place. Users override any template without
touching code by writing a JSON file that maps prompt names to new text:

* ``<workspace>/.harness/prompts.json``   (project level, wins over user level)
* ``~/.config/adaptive-harness/prompts.json`` (user level)
* ``$ADAPTIVE_PROMPTS_FILE``              (explicit file, wins over both)

Inspect the available names with ``adaptive-harness prompts list`` or the
TUI ``/prompts`` command; ``adaptive-harness prompts export`` writes the full
defaults to a JSON file ready for editing.

Templates may contain ``str.format`` placeholders such as ``{target}``; the
call sites pass those values. Placeholders are only substituted when values
are provided, so plain text templates are returned verbatim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from adaptive_harness.data.config import DEFAULT_CONFIG_DIR
except Exception:  # pragma: no cover - config is optional for library use
    DEFAULT_CONFIG_DIR = Path.home() / ".config" / "adaptive-harness"


# --- Default prompts -------------------------------------------------------
# Names are stable identifiers; keep them when editing or overriding.

DEFAULT_PROMPTS: dict[str, str] = {
    # System prompts (base and assembled fragments).
    "system.default": (
        "You are Adaptive Agent, an expert assistant for software engineering, research, science, and mathematics.\n"
        "Inspect relevant evidence, use tools when needed, and distinguish verified results from assumptions.\n"
        "For code changes, run appropriate checks and inspect failures before reporting success.\n"
        "Make routine decisions yourself; ask only when the task target is missing or an action is destructive.\n"
        "You are an active software engineer with direct file tools. You MUST invoke write_file, edit_file, or "
        "run_bash to implement changes. NEVER output code in chat and claim it is done. If you write code, write "
        "it to disk with tools. Create missing project folders with write_file paths or run_bash.\n"
    ),
    "system.tool_guidance": (
        "Use read_file(symbol=...) for focused Python code. For mathematics and science, run deterministic "
        "calculations and verify proposed roots before claiming exactness. Use plot_terminal for useful visual "
        "comparisons."
    ),
    "system.exemplar_notice": (
        "\nVerified prior example from this workspace (use as evidence, then verify current files): "
    ),
    "system.workspace_line": "\nWorkspace: ",
    "system.mode_line": "\nOperating mode: ",
    "system.preferences_line": "\nUser preferences:\n",
    "system.swarm_coordinator": (
        "You are the Swarm Coordinator. You have the capability and tools to spawn and coordinate subagents. "
        "When asked to use multiple agents, decompose the task and call delegate_subagent with architect, "
        "coder, and reviewer roles. Do not claim that delegation is unavailable."
    ),

    # Domain guidance appended to the system prompt per mode.
    "domain.guidance.coding": (
        "Inspect relevant code, make focused edits, validate Python syntax, run appropriate tests, "
        "and review the final diff before claiming success."
    ),
    "domain.guidance.research": (
        "Compare primary sources and retain their URLs or document references. Separate evidence "
        "from inference. Present findings, uncertainty, and citations in a structured Markdown report. "
        "Keep concise research notes when the task spans several sources."
    ),
    "domain.guidance.science": (
        "State assumptions and units. Check algebra or numerical results with the calculator or "
        "reproducible code. Use check_convergence to test observed numerical tail stability, and "
        "check boundary cases and precision. A finite sample does not prove mathematical convergence."
    ),
    "domain.guidance.audit": (
        "Inspect code and diffs without modifying the target. Check injection, authentication, "
        "memory safety, and unsafe command patterns. Verify each suspected finding and report "
        "severity, evidence, and a concrete mitigation."
    ),

    # Classifier / runtime-overseer prompt injections (system role messages).
    "intervention.stop_circling": (
        "CLASSIFIER STOP-CIRCLING DIRECTIVE: The classifier detects unnecessary repeated tool calls. "
        "Stop going around in circles. Do not repeat any previous command or edit. Either take one "
        "genuinely different, verified step that completes the task, or finalize your answer now with "
        "the best verified result and state clearly what remains. A further repeat of the same action "
        "will end the run."
    ),
    "intervention.looping": (
        "OVERSEER INTERVENTION: Repeated tool actions are looping on {target!r}. Stop repeating the same "
        "edit or command. Read the relevant module afresh, identify why prior attempts failed, and use a "
        "different strategy before editing again."
    ),
    "intervention.hallucination": (
        "OVERSEER INTERVENTION: Reality check failed. The available tool evidence says {evidence!r}. "
        "Do not claim success or assume file contents. Use read_file or search_files to ground the next step."
    ),
    "intervention.stalled": (
        "OVERSEER INTERVENTION: Three attempts made no verified progress. Isolate the smallest failing case, "
        "inspect its error, and change strategy. Use run_python_repl only when numerical or symbolic "
        "reproduction is appropriate."
    ),
    "intervention.drift": (
        "OVERSEER INTERVENTION: Scope may have drifted. Re-anchor the next action to the user's original "
        "task: {task!r}. Explain how any proposed edit serves it."
    ),
    "intervention.termination_verdict": (
        "OVERSEER TERMINATION VERDICT: The classifier detects no path to progress after repeated "
        "interventions ({state}). Further tool calls are unnecessary. The run is stopped here; report the "
        "verified results obtained so far and the concrete blocker that prevented completion."
    ),

    # Harness prompt injections (user role nudges during execution).
    "harness.missing_file_changes": (
        "The task requires real file changes. Use write_file or edit_file now. "
        "Do not claim completion until a file tool succeeds."
    ),
    "harness.continuation": (
        "Continue your previous response from exactly where it stopped. Do not repeat "
        "prior text; finish the requested work and clearly report what remains."
    ),
    "harness.verification_repair": (
        "The task is not complete. Repair failed tools and perform the missing checks before answering. "
        "Failed tools: {failures}. Missing checks: {checks}. "
        "If a check cannot be completed, state the concrete blocker instead of claiming success."
    ),

    # Swarm subagent prompts.
    "swarm.role.read_only": (
        "You are a read-only planning or review subagent. Do not attempt write_file or edit_file. "
        "Inspect available evidence and return a useful plan or review with the tools you have."
    ),
    "swarm.role.security": (
        "You are a security reviewer subagent. Do not attempt write_file or edit_file. "
        "Inspect the changed files for security problems and report concrete findings."
    ),
    "swarm.workspace_limit": "Only use available tools within the assigned workspace.",
    "swarm.context_header": "\nPrior verified artifacts (JSON):\n",
    "swarm.instruction.architect.plan": (
        "Inspect relevant interfaces and produce a concise implementation plan with risks. Do not edit files."
    ),
    "swarm.instruction.qa.plan": (
        "Inspect relevant tests and propose concrete regression cases for the task. Do not edit files."
    ),
    "swarm.instruction.coder.implement": (
        "Implement the task in the assigned workspace. Create the requested directories and files with "
        "write_file or edit_file. Run relevant checks."
    ),
    "swarm.instruction.qa.verify": (
        "Independently inspect the implementation and run relevant tests. Report exact test evidence; "
        "do not edit files."
    ),
    "swarm.instruction.security.verify": (
        "Review the changed files for injection, path traversal, unsafe shell use, and credential leaks. "
        "Do not edit files."
    ),

    # LLM classification backend prompts.
    "classifier.ollama": (
        "Classify the task into exactly one of {labels}. Return JSON with label, confidence (0-1), "
        "reasoning. Task: {text}"
    ),
    "classifier.local_http": (
        "Classify into {labels}. Return JSON with label, confidence, reasoning. Task: {text}"
    ),
    "classifier.openrouter": (
        "Classify this task into {labels}. Return JSON with label, confidence (0-1), reasoning. Task: {text}"
    ),
}


class PromptRegistry:
    """Named prompt templates with layered user overrides."""

    def __init__(self, overrides: dict[str, str] | None = None):
        self.defaults = dict(DEFAULT_PROMPTS)
        self.overrides: dict[str, str] = {}
        self.warnings: list[str] = []
        if overrides:
            self._merge(overrides)

    @classmethod
    def for_workspace(cls, workspace: str | Path | None = None) -> "PromptRegistry":
        """Registry with user, environment, and project overrides applied."""
        registry = cls()
        registry.load_overrides(workspace)
        return registry

    def load_overrides(self, workspace: str | Path | None = None) -> None:
        candidates: list[Path] = [DEFAULT_CONFIG_DIR / "prompts.json"]
        if workspace:
            candidates.append(Path(workspace) / ".harness" / "prompts.json")
        env_file = os.getenv("ADAPTIVE_PROMPTS_FILE")
        if env_file:
            candidates.append(Path(env_file))
        for path in candidates:
            data = _read_override_file(path, self.warnings)
            if data is not None:
                self._merge(data)

    def _merge(self, data: dict[str, Any]) -> None:
        for name, text in data.items():
            if not isinstance(name, str) or not isinstance(text, str):
                self.warnings.append(f"Ignored non-string prompt override: {name!r}")
                continue
            self.overrides[name] = text

    def get(self, name: str, **values: Any) -> str:
        """Return a prompt by name, substituting template values when given."""
        if name in self.overrides:
            template = self.overrides[name]
        elif name in self.defaults:
            template = self.defaults[name]
        else:
            raise KeyError(f"Unknown prompt name: {name!r}. Run `adaptive-harness prompts list`.")
        return template.format(**values) if values else template

    def names(self) -> list[str]:
        return sorted(self.defaults)

    def is_overridden(self, name: str) -> bool:
        return name in self.overrides

    def export(self) -> dict[str, str]:
        """All prompts as plain text (overrides applied) for editing."""
        return {name: self.get(name) for name in self.names()}


def _read_override_file(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            warnings.append(f"Prompt override file must be a JSON object: {path}")
            return None
        return data
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read prompt overrides {path}: {exc}")
        return None


_default_registry: PromptRegistry | None = None


def get_default_registry() -> PromptRegistry:
    """Shared registry with user-level and environment overrides (no workspace)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry.for_workspace()
    return _default_registry


def reset_default_registry() -> None:
    """Drop the cached registry (used by tests and after editing override files)."""
    global _default_registry
    _default_registry = None
