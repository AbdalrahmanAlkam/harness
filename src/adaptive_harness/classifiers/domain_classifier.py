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


DOMAIN_GUIDANCE = {
    DomainMode.CODING: ("Inspect relevant code, make focused edits, validate Python syntax, run appropriate tests, "
                        "and review the final diff before claiming success."),
    DomainMode.RESEARCH: ("Compare primary sources and retain their URLs or document references. Separate evidence "
                          "from inference. Present findings, uncertainty, and citations in a structured Markdown report. "
                          "Keep concise research notes when the task spans several sources."),
    DomainMode.SCIENCE: ("State assumptions and units. Check algebra or numerical results with the calculator or "
                         "reproducible code. Use check_convergence to test observed numerical tail stability, and "
                         "check boundary cases and precision. A finite sample does not prove mathematical convergence."),
    DomainMode.AUDIT: ("Inspect code and diffs without modifying the target. Check injection, authentication, "
                       "memory safety, and unsafe command patterns. Verify each suspected finding and report "
                       "severity, evidence, and a concrete mitigation."),
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
