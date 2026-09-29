"""Task domain selection and domain-specific operating guidance."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shlex


class DomainMode(str, Enum):
    CODING = "coding"
    RESEARCH = "research"
    SCIENCE = "science"
    AUDIT = "audit"
    #: Plan mode is a mode, not a personality. The agent may read, search and
    #: delegate; every mutating tool is intercepted and the run ends with a plan
    #: for the operator to approve, edit, or reject.
    PLAN = "plan"


from adaptive_harness.prompts import DEFAULT_PROMPTS

# The editable source of truth for domain guidance is prompts.py (domain.guidance.*).
DOMAIN_GUIDANCE = {
    DomainMode.CODING: DEFAULT_PROMPTS["domain.guidance.coding"],
    DomainMode.RESEARCH: DEFAULT_PROMPTS["domain.guidance.research"],
    DomainMode.SCIENCE: DEFAULT_PROMPTS["domain.guidance.science"],
    DomainMode.AUDIT: DEFAULT_PROMPTS["domain.guidance.audit"],
    DomainMode.PLAN: DEFAULT_PROMPTS["domain.guidance.plan"],
}


def parse_domain_mode(value: str | DomainMode | None) -> DomainMode | None:
    """Map the public security spelling to the existing audit mode."""
    if value is None:
        return None
    if isinstance(value, DomainMode):
        return value
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    if normalized == "security":
        normalized = "audit"
    try:
        return DomainMode(normalized)
    except ValueError as exc:
        raise ValueError(
            "Mode must be coding, research, science, plan, security, or auto") from exc


def audit_command_is_read_only(command: str) -> bool:
    """Limit shell access during an audit to plain Git inspection commands."""
    if not command or any(character in command for character in ";&|><`$\\\n\r"):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if tokens == ["pip-audit"] or tokens == ["safety", "check"]:
        return True
    if len(tokens) < 2 or tokens[0] != "git" or tokens[1] not in {"status", "diff", "show", "log"}:
        return False
    return not any(token.startswith(("--output", "--ext-diff", "--config", "--exec-path"))
                   for token in tokens[2:])


#: Commands that only observe. Used by plan mode, where investigating the
#: workspace is the whole point -- refusing `ls` would make the mode useless
#: for the job it exists to do. A command not listed here is treated as
#: mutating, which is the safe direction: a false refusal costs one tool call.
_READ_ONLY_COMMANDS = frozenset({
    "ls", "pwd", "cat", "head", "tail", "wc", "file", "stat", "du", "df",
    "grep", "rg", "find", "which", "type", "echo", "true", "date", "whoami",
    "env", "printenv", "diff", "sort", "uniq", "basename", "dirname", "realpath",
    "tree", "less", "more", "man", "history",
})
#: Subcommands that only observe, for commands that also have mutating forms.
_READ_ONLY_SUBCOMMANDS = {
    "git": frozenset({"status", "log", "show", "diff", "branch", "remote", "blame",
                      "describe", "rev-parse", "ls-files", "shortlog", "config"}),
    "python": frozenset({"--version", "-V"}),
    "node": frozenset({"--version"}),
    "pip": frozenset({"list", "show", "freeze"}),
    "docker": frozenset({"ps", "images", "inspect", "logs"}),
    "kubectl": frozenset({"get", "describe", "logs", "explain"}),
}
#: Commands whose subcommand is not optional -- `git` alone does nothing
#: useful, and guessing its intent is how a mutating form slips through.
_NEEDS_SUBCOMMAND = frozenset(_READ_ONLY_SUBCOMMANDS)


def plan_command_is_read_only(command: str) -> bool:
    """Whether a shell command only observes, for plan mode.

    Fails closed: a command not recognised is treated as mutating. A false
    refusal costs the model one tool call and teaches it the boundary; a false
    permission changes the workspace during a run that promised not to.
    """
    if not command or any(character in command for character in ";&|><`$\\\n\r"):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    program = Path(tokens[0]).name
    if program in _READ_ONLY_COMMANDS:
        # `find` and `grep` can both delete with a flag, and `echo` can write
        # with a redirect, so a leading flag is checked rather than trusted.
        return not any(token.startswith("-delete") or token.startswith("--delete")
                       for token in tokens[1:])
    if program in _READ_ONLY_SUBCOMMANDS:
        if program in _NEEDS_SUBCOMMAND:
            if len(tokens) < 2:
                return False
            return tokens[1] in _READ_ONLY_SUBCOMMANDS[program]
        return True
    return False


@dataclass
class DomainAssessment:
    mode: DomainMode
    confidence: float


class DomainClassifier:
    CUES = {
        DomainMode.RESEARCH: ("research", "literature", "paper", "citation", "sources", "hypothesis", "bibliography"),
        DomainMode.SCIENCE: ("equation", "numerical", "convergence", "simulation", "theorem", "proof", "statistical", "algorithm"),
        DomainMode.AUDIT: ("security", "audit", "vulnerability", "owasp", "injection", "threat", "review for bugs"),
        DomainMode.CODING: ("code", "implement", "fix", "refactor", "test", "file", "function", "build"),
    }

    def classify(self, text: str) -> DomainAssessment:
        low = text.lower()
        scores = {domain: sum(1 for cue in cues if cue in low) for domain, cues in self.CUES.items()}
        mode = max(scores, key=scores.get)
        return DomainAssessment(mode, 0.5 if scores[mode] == 0 else min(0.95, 0.6 + scores[mode]*0.1))
