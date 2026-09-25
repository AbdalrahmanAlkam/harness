"""Task domain selection and domain-specific operating guidance."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import shlex


class DomainMode(str, Enum):
    CODING = "coding"
    RESEARCH = "research"
    SCIENCE = "science"
    AUDIT = "audit"


from adaptive_harness.prompts import DEFAULT_PROMPTS

# The editable source of truth for domain guidance is prompts.py (domain.guidance.*).
DOMAIN_GUIDANCE = {
    DomainMode.CODING: DEFAULT_PROMPTS["domain.guidance.coding"],
    DomainMode.RESEARCH: DEFAULT_PROMPTS["domain.guidance.research"],
    DomainMode.SCIENCE: DEFAULT_PROMPTS["domain.guidance.science"],
    DomainMode.AUDIT: DEFAULT_PROMPTS["domain.guidance.audit"],
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
        raise ValueError("Mode must be coding, research, science, security, or auto") from exc


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
