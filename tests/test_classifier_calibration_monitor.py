"""Phase 5.2 — classifier self-calibration.

A classifier that silently degrades is a harness that silently rots: the run
still completes, it is just worse, and the cost of being worse is invisible
until someone notices the bill. These tests hold the properties that make the
auto-adjustment safe rather than merely active:

- nothing moves on a handful of samples, because a threshold moved on noise is
  worse than one that waits;
- the threshold is clamped, so feedback on a noisy signal cannot ratchet it one
  way forever;
- an unresolved decision is not counted as a failure — the outcome simply has
  not been observed yet;
- state survives a restart, because a monitor that forgets cannot detect a slow
  decline, which is the failure it exists to catch.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from adaptive_harness.models.calibration_monitor import (
    MAX_THRESHOLD,
    MIN_DECISIONS,
    MIN_THRESHOLD,
    TARGET_ACCURACY,
    CalibrationMonitor,
)


def _feed(monitor: CalibrationMonitor, name: str, *, correct: int, wrong: int,
          predicted: str = "a", actual: str = "a") -> None:
    for index in range(correct + wrong):
        monitor.record(name,
                       predicted if index < correct else f"wrong{index}",
                       0.6, actual if index < correct else "b")


# --- nothing moves too early ------------------------------------------------


def test_nothing_is_adjusted_before_enough_decisions():
    """A threshold moved on a handful of samples is noise, and a system that
    reacts to noise is worse than one that waits."""
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=2, wrong=2)
    assert monitor.adjust("skill") is None
    assert monitor.health("skill").threshold == 0.5


def test_nothing_is_adjusted_without_outcomes():
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    for _ in range(100):
        monitor.record("skill", "a", 0.6)  # no actual
    assert monitor.health("skill").accuracy is None
    assert monitor.adjust("skill") is None


def test_an_unresolved_decision_is_not_a_failure():
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    for _ in range(50):
        monitor.record("skill", "a", 0.6)  # nothing resolved
    assert monitor.health("skill").decisions == 50
    assert monitor.health("skill").resolved == 0
    assert monitor.health("skill").accuracy is None, (
        "an unresolved decision was counted as a failure")


def test_resolving_a_decision_later_updates_the_health():
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    decision = monitor.record("skill", "code_edit", 0.6)
    assert monitor.health("skill").accuracy is None
    monitor.resolve(decision, "code_edit")
    assert monitor.health("skill").accuracy == 1.0


def test_resolving_the_same_decision_twice_cannot_inflate_the_count():
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    decision = monitor.record("skill", "a", 0.6)
    monitor.resolve(decision, "a")
    monitor.resolve(decision, "a")
    assert monitor.health("skill").resolved == 1, "a decision was counted twice"


# --- adjustment direction ---------------------------------------------------


def test_a_degrading_classifier_has_its_threshold_raised():
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=10, wrong=40)
    record = monitor.adjust("skill")
    assert record is not None
    assert record["to"] > record["from"], (
        "a classifier below target had its threshold lowered, which would make "
        "it worse")


def test_a_strong_classifier_relaxes_back_toward_its_default():
    """A system that only ever tightens becomes permanently reluctant, which is
    its own kind of rot."""
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=60, wrong=0)
    before = monitor.health("skill").threshold
    record = monitor.adjust("skill")
    assert record is not None
    assert record["to"] < before or before < 0.5


def test_an_on_target_classifier_is_left_alone():
    monitor = CalibrationMonitor(target=TARGET_ACCURACY)
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=40, wrong=0)  # 100% vs a 75% target
    # Far above target is still a reason to relax, so use a genuinely on-target
    # split for the "leave it alone" case.
    monitor2 = CalibrationMonitor(target=TARGET_ACCURACY)
    monitor2.register("skill", 0.5)
    _feed(monitor2, "skill", correct=37, wrong=13)  # ~74%, within 2 points
    assert monitor2.adjust("skill") is None


# --- the threshold is bounded -----------------------------------------------


def test_the_threshold_cannot_run_away():
    """Feedback on a noisy signal would otherwise ratchet it one way forever."""
    monitor = CalibrationMonitor(target=0.99)
    monitor.register("c", 0.5)
    _feed(monitor, "c", correct=0, wrong=200)
    for _ in range(40):
        monitor.adjust("c")
    assert monitor.health("c").threshold <= MAX_THRESHOLD
    assert monitor.health("c").threshold >= MIN_THRESHOLD


def test_adjustments_are_counted_so_the_movement_is_visible():
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=10, wrong=40)
    for _ in range(5):
        monitor.adjust("skill")
    assert monitor.health("skill").adjustments > 0


def test_an_unregistered_classifier_is_ignored_rather_than_crashing():
    monitor = CalibrationMonitor()
    assert monitor.adjust("never-seen") is None


# --- reporting --------------------------------------------------------------


def test_the_report_flags_a_classifier_below_target():
    monitor = CalibrationMonitor(target=0.75)
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=10, wrong=40)
    monitor.adjust("skill")
    described = monitor.describe()
    assert "below target" in described, (
        "a degraded classifier must be visible, not merely tracked")
    assert "skill" in described


def test_the_report_says_so_when_there_is_nothing_yet():
    assert "No classifier decisions" in CalibrationMonitor().describe()


def test_a_drift_listener_is_told_about_a_movement():
    seen = []
    monitor = CalibrationMonitor(on_drift=seen.append)
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=10, wrong=40)
    monitor.adjust("skill")
    assert seen and seen[0]["classifier"] == "skill"


def test_a_listener_that_raises_does_not_break_tracking():
    def exploding(_record):
        raise RuntimeError("listener is broken")

    monitor = CalibrationMonitor(on_drift=exploding)
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=10, wrong=40)
    monitor.adjust("skill")  # must not raise
    assert monitor.health("skill").adjustments == 1


# --- durability -------------------------------------------------------------


def test_health_survives_a_restart(tmp_path: Path):
    """A monitor that forgets on restart cannot detect a slow decline."""
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=20, wrong=20)
    monitor.adjust("skill")
    path = monitor.save(tmp_path / "calibration.json")

    restored = CalibrationMonitor()
    assert restored.load(path)
    assert restored.health("skill").resolved == 40
    assert restored.health("skill").accuracy == pytest.approx(0.5)
    assert restored.health("skill").adjustments == 1


def test_a_missing_or_corrupt_file_is_not_fatal(tmp_path: Path):
    monitor = CalibrationMonitor()
    assert monitor.load(tmp_path / "absent.json") is False
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert monitor.load(corrupt) is False


def test_the_saved_file_is_readable_json(tmp_path: Path):
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)
    _feed(monitor, "skill", correct=5, wrong=5)
    payload = json.loads(monitor.save(tmp_path / "c.json").read_text(encoding="utf-8"))
    assert "health" in payload and "skill" in payload["health"]


# --- concurrency ------------------------------------------------------------


def test_recording_is_thread_safe():
    monitor = CalibrationMonitor()
    monitor.register("skill", 0.5)

    def feed(start: int) -> None:
        for index in range(50):
            monitor.record("skill", "a" if index % 2 else "b", 0.6, "a")

    threads = [threading.Thread(target=feed, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    health = monitor.health("skill")
    assert health.decisions == 400
    assert health.resolved == 400, "a concurrent update was lost"
    assert health.correct == 200
