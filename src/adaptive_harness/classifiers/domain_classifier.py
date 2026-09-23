"""Task domain selection and domain-specific operating guidance."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DomainMode(str, Enum):
    CODING = "coding"
    RESEARCH = "research"
    SCIENCE = "science"
    AUDIT = "audit"


DOMAIN_GUIDANCE = {
    DomainMode.CODING: "Inspect relevant code, make focused changes, then run tests, compiler or AST checks, and review the diff.",
    DomainMode.RESEARCH: "Compare sources, extract citations, distinguish evidence from inference, and write structured notes. Use web search only when available.",
    DomainMode.SCIENCE: "Show assumptions, check units and boundary cases, and verify numerical convergence where applicable.",
    DomainMode.AUDIT: "Inspect attack surfaces and compliance requirements, verify suspected findings, and report severity with evidence.",
}


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
