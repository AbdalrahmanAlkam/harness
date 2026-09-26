"""The Invariant Gate and the Relentless Convergence Loop.

The loop is a state machine that refuses to report success until four
independent conditions hold simultaneously:

1. **Mathematical soundness** — every proof script exits 0 and is exact.
2. **Empirical replication** — every seeded experiment reproduces its declared
   prediction inside a 95% interval and emits a hashed data artifact.
3. **Adversarial clearance** — the red team recorded no open counterexample.
4. **Document integrity** — ``paper.typ`` compiles to ``paper.pdf`` with no
   Typst warnings.

The loop is *not* turn-limited. It is **stagnation-limited**: if a cycle leaves
the gate fingerprint unchanged, the Director escalates the worker budget, and
if escalation still yields no progress the run aborts as ``STAGNATION_ABORT``
with the state of the last cycle recorded. That distinction matters — a turn
cap is an arbitrary guess that punishes hard problems, whereas stagnation is
a proof that further identical work cannot help.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


class Invariant(str, Enum):
    """The conditions that jointly define a settled, publishable result."""

    MATHEMATICAL_SOUNDNESS = "mathematical_soundness"
    EMPIRICAL_REPLICATION = "empirical_replication"
    ADVERSARIAL_CLEARANCE = "adversarial_clearance"
    DOCUMENT_INTEGRITY = "document_integrity"
    CLAIM_ADJUDICATION = "claim_adjudication"
    FORMAL_VERIFICATION = "formal_verification"


class StopReason(str, Enum):
    """Why the convergence loop ended."""

    CONVERGED = "CONVERGED"
    STAGNATION_ABORT = "STAGNATION_ABORT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    EXTERNAL_STOP = "EXTERNAL_STOP"
    FATAL = "FATAL"


@dataclass(frozen=True)
class InvariantStatus:
    """Evaluation of one invariant in one cycle."""

    invariant: Invariant
    satisfied: bool
    detail: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"invariant": self.invariant.value, "satisfied": self.satisfied,
                "detail": self.detail, "evidence": list(self.evidence)}


@dataclass(frozen=True)
class GateReport:
    """Aggregate verdict of the Invariant Gate."""

    statuses: tuple[InvariantStatus, ...]
    fingerprint: str
    cycle: int

    @property
    def satisfied(self) -> bool:
        return all(status.satisfied for status in self.statuses)

    @property
    def gaps(self) -> tuple[InvariantStatus, ...]:
        return tuple(status for status in self.statuses if not status.satisfied)

    def to_dict(self) -> dict[str, Any]:
        return {"cycle": self.cycle, "satisfied": self.satisfied,
                "fingerprint": self.fingerprint,
                "statuses": [status.to_dict() for status in self.statuses]}

    def render(self) -> str:
        lines = [f"[gate] cycle {self.cycle}: {'SOLVED' if self.satisfied else 'GAPS DETECTED'}"]
        for status in self.statuses:
            mark = "PASS" if status.satisfied else "FAIL"
            lines.append(f"  [{mark}] {status.invariant.value}: {status.detail}")
        return "\n".join(lines)


class InvariantGate:
    """Evaluate the definition of solved from the current artifact state.

    ``evaluators`` maps an :class:`Invariant` to a callable returning
    ``(satisfied, detail, evidence)``. Plugging the real proof/experiment/
    typst runners in keeps the gate itself free of I/O.
    """

    def __init__(self, evaluators: Mapping[Invariant, Callable[[], tuple[bool, str, Sequence[str]]]]):
        missing = set(Invariant) - set(evaluators)
        if missing:
            raise ValueError(f"Invariant gate is missing evaluators for: {sorted(item.value for item in missing)}")
        self._evaluators = dict(evaluators)

    def evaluate(self, cycle: int = 0) -> GateReport:
        statuses: list[InvariantStatus] = []
        for invariant in Invariant:
            satisfied, detail, evidence = self._evaluators[invariant]()
            statuses.append(InvariantStatus(invariant, bool(satisfied), detail, tuple(evidence)))
        # The fingerprint encodes only the pass/fail shape, not the prose, so
        # re-running an unchanged artifact set counts as no progress.
        shape = "|".join(f"{status.invariant.value}:{int(status.satisfied)}" for status in statuses)
        fingerprint = hashlib.sha256(shape.encode("utf-8")).hexdigest()[:16]
        return GateReport(tuple(statuses), fingerprint, cycle)


@dataclass
class ConvergenceCycle:
    """One pass of the loop, retained for the audit appendix."""

    index: int
    gaps: tuple[str, ...]
    diagnosis: str
    workers_spawned: tuple[str, ...] = ()
    gate_fingerprint: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "gaps": list(self.gaps), "diagnosis": self.diagnosis,
                "workers_spawned": list(self.workers_spawned),
                "gate_fingerprint": self.gate_fingerprint, "elapsed_s": round(self.elapsed_s, 2)}


@dataclass
class ConvergenceOutcome:
    """Terminal state of the Relentless Convergence Loop."""

    solved: bool
    stop_reason: StopReason
    cycles: int
    final_report: GateReport
    history: tuple[ConvergenceCycle, ...]
    workers_spawned: int

    def to_dict(self) -> dict[str, Any]:
        return {"solved": self.solved, "stop_reason": self.stop_reason.value,
                "cycles": self.cycles, "workers_spawned": self.workers_spawned,
                "gate": self.final_report.to_dict(),
                "history": [cycle.to_dict() for cycle in self.history]}

    def render(self) -> str:
        lines = [self.final_report.render(), "",
                 f"[loop] {self.stop_reason.value} after {self.cycles} cycle(s); "
                 f"{self.workers_spawned} worker(s) spawned"]
        if not self.solved:
            lines.append("[loop] UNSOLVED — remaining gaps:")
            lines.extend(f"  - {gap}" for gap in self.final_report.gaps)
        return "\n".join(lines)


class StagnationError(RuntimeError):
    """The loop cannot progress and must escalate or abort."""


@dataclass
class RelentlessConvergenceLoop:
    """Iterate diagnosis and remediation until the gate passes or progress stops.

    ``cycle_fn`` performs one full research cycle and returns the workers it
    spawned. ``diagnose_fn`` explains the current gaps for the audit log. The
    loop owns escalation: each consecutive no-progress cycle raises the worker
    budget handed to ``cycle_fn`` through ``escalate_fn``.
    """

    gate: InvariantGate
    cycle_fn: Callable[[GateReport, int, int], Sequence[str]]
    diagnose_fn: Callable[[GateReport], str] = lambda report: (
        "; ".join(f"{status.invariant.value}: {status.detail}" for status in report.gaps) or "none")
    escalate_fn: Callable[[int], int] = lambda budget: max(2, budget * 2)
    stagnation_patience: int = 2
    budget_ceiling: int = 32
    absolute_ceiling: int = 64
    on_cycle: Callable[[ConvergenceCycle], None] | None = None
    history: list[ConvergenceCycle] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.stagnation_patience < 1:
            raise ValueError("stagnation_patience must be at least 1")
        if self.budget_ceiling < 1:
            raise ValueError("budget_ceiling must be at least 1")
        if self.absolute_ceiling < 1:
            raise ValueError("absolute_ceiling must be at least 1")

    def run(self, max_cycles: int | None = None,
            should_stop: Callable[[GateReport], bool] | None = None) -> ConvergenceOutcome:
        """Drive the loop to convergence.

        The loop is *stagnation*-limited, not turn-limited: it exits on
        convergence, or when it can prove that further identical cycles cannot
        help. Two independent guarantees make that exit unreachable-in-practice
        but always finite:

        * **Monotone progress.** Progress is reaching a *new minimum* in the
          number of outstanding gaps, tracked as a high-water mark. Comparing
          against only the previous cycle would be exploitable — a single
          flapping invariant oscillating 4,3,4,3 would look like progress every
          other cycle and reset the stagnation counter forever. A high-water
          mark admits at most ``len(Invariants)`` progress events in total.
        * **Absolute ceiling.** ``absolute_ceiling`` caps total cycles as a
          last-resort net. It is a safety net, never a convergence criterion:
          hitting it reports ``BUDGET_EXHAUSTED`` and an honest UNSOLVED
          verdict, never a false success.

        ``max_cycles`` is a separate operator-set valve that may be lower still;
        it likewise yields ``BUDGET_EXHAUSTED`` rather than a false success.
        """
        report = self.gate.evaluate(cycle=0)
        self.history.clear()
        workers_total = 0
        stale_cycles = 0
        last_fingerprint = report.fingerprint
        best_gap_count = len(report.gaps)
        budget = 2
        cycle_index = 0
        stop_reason = StopReason.STAGNATION_ABORT
        solved = report.satisfied

        while True:
            if should_stop is not None and should_stop(report):
                stop_reason = StopReason.EXTERNAL_STOP
                break
            if report.satisfied:
                stop_reason = StopReason.CONVERGED
                solved = True
                break
            if cycle_index >= self.absolute_ceiling or (
                    max_cycles is not None and cycle_index >= max_cycles):
                stop_reason = StopReason.BUDGET_EXHAUSTED
                break

            cycle_index += 1
            started = time.perf_counter()
            spawned = tuple(self.cycle_fn(report, cycle_index, budget) or ())
            workers_total += len(spawned)
            report = self.gate.evaluate(cycle=cycle_index)

            # Progress is a new minimum in outstanding gaps, never a regression
            # from a single flapping cycle. See the docstring for why.
            gap_count = len(report.gaps)
            progressed = gap_count < best_gap_count
            if progressed:
                best_gap_count = gap_count
            if progressed:
                last_fingerprint = report.fingerprint
                stale_cycles = 0
                budget = 2
            else:
                stale_cycles += 1
                if stale_cycles >= self.stagnation_patience:
                    # Escalate before conceding: a wider worker pool is the only
                    # lever that changes what the next cycle can even attempt.
                    # Once escalation would exceed the ceiling, no lever remains
                    # and further identical cycles are provably futile.
                    escalated = self.escalate_fn(budget)
                    stale_cycles = 0
                    if escalated > self.budget_ceiling or escalated <= budget:
                        self._record(ConvergenceCycle(
                            cycle_index, tuple(status.invariant.value for status in report.gaps),
                            self.diagnose_fn(report), spawned, report.fingerprint,
                            time.perf_counter() - started))
                        stop_reason = StopReason.STAGNATION_ABORT
                        break
                    budget = escalated

            self._record(ConvergenceCycle(
                cycle_index, tuple(status.invariant.value for status in report.gaps),
                self.diagnose_fn(report), spawned, report.fingerprint,
                time.perf_counter() - started))

        solved = stop_reason is StopReason.CONVERGED
        return ConvergenceOutcome(solved, stop_reason, cycle_index, report,
                                  tuple(self.history), workers_total)

    def _record(self, cycle: ConvergenceCycle) -> None:
        self.history.append(cycle)
        if self.on_cycle is not None:
            self.on_cycle(cycle)


def write_cycle_history(path: str | Path, outcome: ConvergenceOutcome) -> Path:
    """Persist the loop history as the machine-readable audit trail."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(outcome.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
