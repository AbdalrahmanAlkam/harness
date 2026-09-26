"""Divisions, leaders, and worker roles for the autonomous research swarm.

The organisation mirrors an academic research institute: an Executive Director
owns the objective and the convergence loop; four Division Leaders each own a
methodologically distinct line of attack; every Leader may spawn an unbounded
number of Workers with a narrow directive and an explicit tool budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Mapping


class Division(str, Enum):
    """A methodologically distinct line of attack."""

    LITERATURE = "literature"
    THEORY = "theory"
    EMPIRICAL = "empirical"
    ADVERSARIAL = "adversarial"
    FORMAL = "formal"


class Clearance(str, Enum):
    """Adversarial disposition of a claim after red-team review."""

    PENDING = "PENDING"
    CLEARED = "CLEARED"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"


@dataclass(frozen=True)
class DivisionSpec:
    """Static description of a division and its default worker roles."""

    division: Division
    leader_title: str
    objective: str
    worker_titles: tuple[str, ...]
    default_tools: tuple[str, ...]


DIVISION_SPECS: Mapping[Division, DivisionSpec] = {
    Division.LITERATURE: DivisionSpec(
        Division.LITERATURE, "Literature Lead",
        "Establish which prior results constrain the problem, and record every "
        "external claim as a citable evidence record rather than an assertion.",
        ("Scout", "Surveyor", "Citation Auditor"),
        ("web_search", "read_file", "write_file", "list_directory", "search_files")),
    Division.THEORY: DivisionSpec(
        Division.THEORY, "Theoretical Lead",
        "Convert the objective into exact, self-contained derivations whose scripts "
        "execute to exit code 0 with no floating-point approximation.",
        ("SymPy Prover", "Lean Formalist", "Lemma Hunter", "Bound Analyst"),
        ("read_file", "write_file", "edit_file", "run_bash", "run_python_repl",
         "run_lean_proof", "calculate")),
    Division.EMPIRICAL: DivisionSpec(
        Division.EMPIRICAL, "Empirical Lead",
        "Reproduce every theoretical prediction under a pinned seed and report the "
        "95% interval, including the runs that contradict the theory.",
        ("Simulation Worker", "Benchmark Worker", "Statistics Auditor"),
        ("read_file", "write_file", "edit_file", "list_directory", "search_files",
         "run_bash", "run_python_repl", "plot_terminal")),
    Division.ADVERSARIAL: DivisionSpec(
        Division.ADVERSARIAL, "Adversarial Lead",
        "Attempt to falsify every claim by constructing counterexamples, boundary "
        "cases, and unstated assumptions; grant clearance only when attempts fail.",
        ("Red Team Auditor", "Falsifier", "Assumption Hunter"),
        ("read_file", "search_files", "list_directory", "run_bash", "run_python_repl",
         "run_lean_proof")),
    Division.FORMAL: DivisionSpec(
        Division.FORMAL, "Formal Proof Lead",
        "Formalise each proposition in Lean 4 and have the kernel machine-check it, so the "
        "published claim rests on a verified derivation and not only on symbolic computation.",
        ("Lean Formalist", "Tactic Specialist", "Axiom Auditor"),
        ("read_file", "write_file", "edit_file", "run_python_repl", "run_bash",
         "run_lean_proof")),
}

# Tool sets for the interactive subagent loop, per division. These differ from the
# static DivisionSpec lists above: a *worker* that iterates in a tool loop needs
# the instruments of its methodology, and the theorist additionally needs the Lean
# prover so a formal proof can be machine-checked mid-iteration.
WORKER_TOOLS: Mapping[Division, tuple[str, ...]] = {
    Division.LITERATURE: ("read_file", "search_files", "list_directory", "write_file",
                          "run_bash", "web_search", "compile_typst"),
    Division.THEORY: ("read_file", "write_file", "edit_file", "run_python_repl",
                      "run_bash", "run_lean_proof"),
    Division.EMPIRICAL: ("read_file", "write_file", "edit_file", "run_python_repl", "run_bash"),
    Division.ADVERSARIAL: ("search_files", "read_file", "write_file", "edit_file",
                           "run_python_repl", "run_bash", "run_lean_proof"),
    Division.FORMAL: ("read_file", "write_file", "edit_file", "run_python_repl",
                      "run_bash", "run_lean_proof"),
}

# A falsification attempt must return this shape. Prose alone is ambiguous: read
# strictly, every reply becomes a counterexample and live mode can never
# converge; read generously, a claim nobody examined gets cleared.
FALSIFICATION_VERDICT = ('Return ONLY a JSON object: {"falsified": true|false, '
                         '"finding": "<the counterexample, or why the search was empty>"}')


@dataclass(frozen=True)
class FalsificationVerdict:
    """Structured outcome of one red-team attempt."""

    status: Clearance
    finding: str

    @property
    def conclusive(self) -> bool:
        """Whether the attempt produced a usable verdict rather than prose."""
        return self.status is not Clearance.PENDING

    @classmethod
    def parse(cls, raw: str | None) -> "FalsificationVerdict":
        """Interpret an author response as a verdict, defaulting to inconclusive.

        An unparseable response is PENDING rather than CLEARED: granting
        clearance on ambiguity is a rubber stamp, and declaring a claim false on
        ambiguity is a false accusation. Declining to adjudicate leaves the gate
        unsatisfied, which is the honest outcome.
        """
        text = (raw or "").strip()
        if not text:
            return cls(Clearance.PENDING, "the falsification attempt returned nothing")
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return cls(Clearance.PENDING,
                       "the falsification attempt did not return the required JSON verdict")
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            return cls(Clearance.PENDING, f"the falsification verdict was not valid JSON: {exc}")
        if not isinstance(payload, dict) or "falsified" not in payload:
            return cls(Clearance.PENDING, "the falsification verdict had no 'falsified' field")
        finding = str(payload.get("finding", "")).strip()
        if payload["falsified"] is True:
            return cls(Clearance.COUNTEREXAMPLE, finding or "counterexample reported without detail")
        if payload["falsified"] is False:
            return cls(Clearance.CLEARED, finding or "no counterexample found")
        return cls(Clearance.PENDING,
                   f"'falsified' must be a boolean, got {payload['falsified']!r}")


@dataclass
class ResearchAgent:
    """A live member of the swarm: a Leader or a spawned Worker."""

    agent_id: str
    role_name: str
    directive: str
    allowed_tools: tuple[str, ...]
    budget_tokens: int = 16000
    parent_id: str | None = None
    division: Division | None = None
    clearance: Clearance = Clearance.PENDING
    children: list[str] = field(default_factory=list)
    # Explicit rather than inferred from the role name. A title heuristic silently
    # demoted any lead whose name did not happen to end in "Lead" — and promoted
    # nothing, so the failure was a quiet loss of hierarchy, not a visible error.
    is_leader: bool = False

    @property
    def depth(self) -> int:
        return 0 if self.is_leader else 1

    def to_dict(self) -> dict[str, object]:
        return {"agent_id": self.agent_id, "role_name": self.role_name,
                "parent_id": self.parent_id,
                "division": self.division.value if self.division else None,
                "directive": self.directive[:400],
                "allowed_tools": list(self.allowed_tools),
                "budget_tokens": self.budget_tokens,
                "clearance": self.clearance.value,
                "is_leader": self.is_leader,
                "children": list(self.children)}


def leader_agent_id(division: Division) -> str:
    return f"{division.value}_lead_01"


def worker_agent_id(division: Division, index: int) -> str:
    spec = DIVISION_SPECS[division]
    slug = spec.worker_titles[(index - 1) % len(spec.worker_titles)].lower().replace(" ", "_")
    return f"{division.value}_{slug}_{index:02d}"
