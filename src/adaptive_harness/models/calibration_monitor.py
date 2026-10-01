"""Classifier self-calibration.

A classifier that silently degrades is a harness that silently rots. Nothing
in the product notices when `SkillClassifier` starts routing 40% of coding tasks
to `general_reasoning`, because the run still completes -- it is just worse, and
the cost of being worse is invisible until someone notices the bill.

So each classifier's decisions are tracked against outcomes, and a decision
threshold is adjusted within configured bounds to hold a target accuracy. The
bounds are the point: an auto-adjusting threshold that could swing anywhere
would be a classifier quietly rewriting its own behaviour with no ceiling. Drift
is emitted as an event rather than logged, because a log line nobody reads is
not a signal.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: The default accuracy a classifier is expected to hold. Below it, the
#: threshold moves; above it, it relaxes back toward the default so the system
#: does not drift into permanently refusing.
TARGET_ACCURACY = 0.75

#: A threshold may move this far from its default, and no further. Without the
#: clamp, feedback on a noisy signal ratchets the threshold one way forever.
MIN_THRESHOLD = 0.30
MAX_THRESHOLD = 0.90

#: Decisions are tracked in a bounded window, so a classifier that is good now
#: is not judged on decisions from a year ago.
WINDOW = 500

#: Below this many decisions, no adjustment is made. A threshold moved on two
#: samples is noise, and a system that reacts to noise is worse than one that
#: waits.
MIN_DECISIONS = 40


@dataclass
class Decision:
    """One classifier decision, and what actually happened."""

    name: str
    predicted: str
    confidence: float
    #: The correct label, filled in once the outcome is known. `None` while the
    #: outcome is still unknown, which is why an unfilled decision is not
    #: counted as a failure.
    actual: Optional[str] = None
    at: float = field(default_factory=time.time)

    @property
    def resolved(self) -> bool:
        return self.actual is not None

    @property
    def correct(self) -> Optional[bool]:
        if not self.resolved:
            return None
        return self.predicted == self.actual

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "predicted": self.predicted,
                "confidence": round(self.confidence, 4), "actual": self.actual,
                "correct": self.correct, "at": self.at}


@dataclass
class ClassifierHealth:
    """What one classifier has been doing, and what has been done about it."""

    name: str
    decisions: int = 0
    resolved: int = 0
    correct: int = 0
    threshold: float = 0.5
    default_threshold: float = 0.5
    adjustments: int = 0
    last_drift: float = 0.0

    @property
    def accuracy(self) -> Optional[float]:
        """Accuracy over *resolved* decisions only.

        An unresolved decision is not a failure: the outcome simply has not
        been observed yet, and counting it as wrong would make the tracker look
        broken rather than busy.
        """
        if not self.resolved:
            return None
        return self.correct / self.resolved

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "decisions": self.decisions,
                "resolved": self.resolved, "correct": self.correct,
                "accuracy": self.accuracy, "threshold": round(self.threshold, 4),
                "default_threshold": self.default_threshold,
                "adjustments": self.adjustments, "last_drift": self.last_drift}


class CalibrationMonitor:
    """Tracks every classifier's decisions and holds them near target accuracy."""

    def __init__(self, *, target: float = TARGET_ACCURACY,
                 window: int = WINDOW,
                 on_drift: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.target = target
        self.window = window
        self._on_drift = on_drift
        self._lock = threading.Lock()
        self._decisions: Dict[str, List[Decision]] = {}
        self._health: Dict[str, ClassifierHealth] = {}

    # -- recording ---------------------------------------------------------

    def record(self, name: str, predicted: str, confidence: float,
               actual: Optional[str] = None) -> Decision:
        """Record one decision, optionally with its outcome already known."""
        decision = Decision(name=name, predicted=str(predicted),
                            confidence=float(confidence), actual=actual)
        with self._lock:
            entries = self._decisions.setdefault(name, [])
            entries.append(decision)
            if len(entries) > self.window:
                del entries[: len(entries) - self.window]
            health = self._health.setdefault(name, ClassifierHealth(name=name))
            health.decisions += 1
            if decision.resolved:
                health.resolved += 1
                if decision.correct:
                    health.correct += 1
        return decision

    def resolve(self, decision: Decision, actual: str) -> None:
        """Fill in an outcome that was not known when the call was made."""
        with self._lock:
            decision.actual = str(actual)
            health = self._health.setdefault(decision.name,
                                             ClassifierHealth(name=decision.name))
            # Recomputed from the window rather than incremented, so resolving
            # a decision twice cannot inflate the count.
            health.resolved = sum(1 for item in self._decisions.get(decision.name, [])
                                  if item.resolved)
            health.correct = sum(1 for item in self._decisions.get(decision.name, [])
                                 if item.correct is True)

    # -- adjusting ---------------------------------------------------------

    def register(self, name: str, threshold: float = 0.5) -> None:
        """Note a classifier's default threshold, so drift is measured from it."""
        with self._lock:
            health = self._health.setdefault(name, ClassifierHealth(name=name))
            health.default_threshold = float(threshold)
            health.threshold = float(threshold)

    def adjust(self, name: str) -> Optional[Dict[str, Any]]:
        """Move one classifier's threshold toward the target accuracy.

        Returns the drift record when something moved, else None. Nothing is
        adjusted before `MIN_DECISIONS` resolved decisions: a threshold moved on
        a handful of samples is noise, and a system that reacts to noise is
        worse than one that waits.
        """
        with self._lock:
            health = self._health.get(name)
            if health is None:
                return None
            accuracy = health.accuracy
            if accuracy is None or health.resolved < MIN_DECISIONS:
                return None
            error = 1.0 - accuracy
            if abs(error - (1.0 - self.target)) < 0.02:
                return None  # already on target; leave it alone

            before = health.threshold
            # Above target: relax, so an easy classifier does not become
            # permanently reluctant. Below target: tighten.
            direction = -1.0 if accuracy > self.target else 1.0
            step = 0.05 * direction
            health.threshold = _clamp(health.threshold + step,
                                      max(MIN_THRESHOLD, health.default_threshold - 0.2),
                                      min(MAX_THRESHOLD, health.default_threshold + 0.2))
            if abs(health.threshold - before) < 1e-9:
                return None  # already at its bound
            health.adjustments += 1
            health.last_drift = round(health.threshold - before, 4)
            record = {"classifier": name, "accuracy": round(accuracy, 4),
                      "target": self.target, "from": round(before, 4),
                      "to": round(health.threshold, 4), "resolved": health.resolved}
        if self._on_drift is not None:
            try:
                self._on_drift(record)
            except Exception:  # noqa: BLE001 - a listener must not break tracking
                pass
        return record

    def adjust_all(self) -> List[Dict[str, Any]]:
        return [record for name in list(self._health)
                if (record := self.adjust(name)) is not None]

    # -- reporting ---------------------------------------------------------

    def health(self, name: str) -> Optional[ClassifierHealth]:
        with self._lock:
            return self._health.get(name)

    def report(self) -> Dict[str, Any]:
        with self._lock:
            return {name: health.to_dict() for name, health in self._health.items()}

    def describe(self) -> str:
        report = self.report()
        if not report:
            return "No classifier decisions recorded yet."
        lines = ["Classifier health"]
        for name, entry in sorted(report.items()):
            accuracy = entry["accuracy"]
            shown = f"{accuracy:.1%}" if accuracy is not None else "not yet known"
            drift = entry["last_drift"]
            drift_text = "" if not drift else (
                f", threshold {entry['threshold']:.2f} ({drift:+.2f})")
            flag = "" if accuracy is None or accuracy >= self.target else "  ← below target"
            lines.append(f"  {name}: {shown} over {entry['resolved']} resolved "
                         f"of {entry['decisions']} decisions{drift_text}{flag}")
        return "\n".join(lines)

    # -- durability --------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Persist, so a restart does not reset the picture.

        A monitor that forgets everything on restart cannot detect a slow
        decline, which is exactly the failure it exists to catch.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # Built inline rather than by calling report(): report() takes the
            # same non-reentrant lock, so calling it here deadlocks.
            payload = {
                "target": self.target,
                "health": {name: health.to_dict() for name, health in self._health.items()},
                "decisions": {name: [item.to_dict() for item in entries[-self.window:]]
                              for name, entries in self._decisions.items()},
            }
        handle, temp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, default=str)
            os.replace(temp, target)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
        return target

    def load(self, path: Path | str) -> bool:
        target = Path(path)
        if not target.is_file():
            return False
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        with self._lock:
            self.target = float(payload.get("target", self.target))
            for name, entry in (payload.get("health") or {}).items():
                health = ClassifierHealth(name=name)
                health.decisions = int(entry.get("decisions", 0))
                health.resolved = int(entry.get("resolved", 0))
                health.correct = int(entry.get("correct", 0))
                health.threshold = float(entry.get("threshold", 0.5))
                health.default_threshold = float(entry.get("default_threshold", 0.5))
                health.adjustments = int(entry.get("adjustments", 0))
                self._health[name] = health
            for name, entries in (payload.get("decisions") or {}).items():
                restored = []
                for entry in entries:
                    if isinstance(entry, dict):
                        restored.append(Decision(
                            name=str(entry.get("name", name)),
                            predicted=str(entry.get("predicted", "")),
                            confidence=float(entry.get("confidence", 0.0)),
                            actual=entry.get("actual"),
                            at=float(entry.get("at", 0.0))))
                if restored:
                    self._decisions[name] = restored
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._health)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
