"""Numerical verification of the calibration metrics.

Each value is checked against a hand computation or an independently written
reference rather than against the code's own output, so a change that moves a
number has to justify itself.

The bin-boundary tests are a regression test for a real defect: bin edges were
built with ``np.linspace(0, 1, n_bins + 1)``, whose 0.7 edge is
``0.7000000000000001``. A model confidence of exactly 0.7 therefore failed the
``>= low`` test and was filed into the bin *below* the boundary, so the
reliability diagram reported the point in a bin it does not belong to. The
edges are now ``i / n_bins``, which is exactly the decimal boundary a caller
writes.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from adaptive_harness.evaluation.metrics import (
    brier_score_loss,
    confidence_margin,
    cross_entropy_loss,
    expected_calibration_error,
    shannon_entropy,
)
from adaptive_harness.models.calibration import compute_brier_score, compute_ece

CLASSES = ["a", "b"]

# Two samples at confidence 0.9 (one right, one wrong) and two at 0.7 (one
# right, one wrong). With 10 bins over [0,1]:
#   bin 7 = [0.7,0.8): n=2, acc=0.5, conf=0.7 -> (2/4)*0.2 = 0.1
#   bin 9 = [0.9,1.0): n=2, acc=0.5, conf=0.9 -> (2/4)*0.4 = 0.2
#   ECE = 0.3
PROBS = np.array([[0.9, 0.1], [0.9, 0.1], [0.3, 0.7], [0.3, 0.7]])
Y = ["a", "b", "b", "a"]


def reference_ece(y_true, probs, n_bins=10):
    """Textbook ECE, written independently of the project's binning code."""
    names = [f"c{i}" for i in range(probs.shape[1])]
    index = {name: i for i, name in enumerate(names)}
    truth = np.array([index[y] for y in y_true])
    correct = (probs.argmax(axis=1) == truth).astype(float)
    conf = probs.max(axis=1)
    edges = np.arange(n_bins + 1) / n_bins
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf <= hi) if hi == 1.0 else (conf >= lo) & (conf < hi)
        n = int(sel.sum())
        if n:
            total += (n / len(conf)) * abs(correct[sel].mean() - conf[sel].mean())
    return total


def test_ece_matches_a_hand_computation():
    """A fixture small enough to total on paper."""
    ece, detail = expected_calibration_error(Y, PROBS, CLASSES, n_bins=10)
    assert ece == pytest.approx(0.3, abs=1e-12)
    assert detail["bin_counts"][7] == 2
    assert detail["bin_accuracies"][7] == pytest.approx(0.5)
    assert detail["bin_confidences"][7] == pytest.approx(0.7)
    assert detail["bin_counts"][9] == 2
    assert sum(detail["bin_counts"]) == len(Y), "every sample must land in one bin"


def test_brier_matches_a_hand_computation():
    # (0.9-1)^2+(0.1-0)^2=0.02 | (0.9-0)^2+(0.1-1)^2=1.62
    # (0.3-0)^2+(0.7-1)^2=0.18 | (0.3-1)^2+(0.7-0)^2=0.98  ->  2.80 / 4 = 0.70
    assert brier_score_loss(Y, PROBS, CLASSES) == pytest.approx(0.7, abs=1e-12)
    assert brier_score_loss(["a", "b"], np.eye(2), CLASSES) == pytest.approx(0.0)
    assert brier_score_loss(["a", "b"],
                            np.array([[0.0, 1.0], [1.0, 0.0]]), CLASSES) == pytest.approx(2.0)


@pytest.mark.parametrize("n_bins", [5, 10, 20])
def test_ece_matches_an_independent_reference(n_bins):
    """Randomised, fixed seed, so a failure is reproducible."""
    rng = np.random.default_rng(20260927)
    for _ in range(40):
        k = int(rng.integers(2, 6))
        names = [f"c{i}" for i in range(k)]
        p = rng.dirichlet(np.ones(k), size=int(rng.integers(1, 40)))
        y = [names[i] for i in rng.integers(0, k, size=len(p))]
        got, _ = compute_ece(y, p, names, n_bins=n_bins)
        assert got == pytest.approx(reference_ece(y, p, n_bins), abs=1e-12)


def test_the_two_ece_implementations_agree_exactly():
    """evaluation.metrics and models.calibration duplicate this; keep them honest."""
    rng = np.random.default_rng(11)
    for _ in range(60):
        k = int(rng.integers(2, 6))
        names = [f"c{i}" for i in range(k)]
        p = rng.dirichlet(np.ones(k), size=int(rng.integers(1, 40)))
        y = [names[i] for i in rng.integers(0, k, size=len(p))]
        a, da = expected_calibration_error(y, p, names, n_bins=10)
        b, db = compute_ece(y, p, names, n_bins=10)
        assert a == b
        assert da == db


@pytest.mark.parametrize("n_bins", [5, 10, 20, 50])
def test_a_confidence_exactly_on_a_bin_edge_is_not_filed_one_bin_low(n_bins):
    """Regression: ``linspace`` put the 0.7 edge one ULP high.

    Walks every interior edge plus every confidence a uniform distribution can
    produce (1/K), because a row's confidence is its maximum.
    """
    misfiled = []
    probes = []
    for k in range(2, 21):
        boundary = 1.0 / k
        if 0.0 < boundary < 1.0:
            # Exact rational arithmetic: 0.58 * 50 is 28.999... in float, so an
            # int() of the product would predict the wrong bin.
            exact = Fraction(1, k) * n_bins
            probes.append((k, boundary, exact.numerator // exact.denominator))
    for i in range(1, n_bins):
        probes.append((0, i / n_bins, i))

    for k, boundary, want_bin in probes:
        if k == 0:
            # Pad so the row max is the boundary; padding must stay at or below
            # it, which needs (1-b)/(k-1) <= b, i.e. k >= 1/b.
            k = max(2, math.ceil(1.0 / boundary - 1e-12))
            rest = (1.0 - boundary) / (k - 1)
            row = [boundary] + [rest] * (k - 1)
            assert max(row) == boundary
        else:
            row = [1.0 / k] * k
        names = [f"c{i}" for i in range(k)]
        _, detail = compute_ece([names[0]], np.array([row]), names, n_bins=n_bins)
        placed = [i for i, count in enumerate(detail["bin_counts"]) if count]
        if placed != [want_bin]:
            misfiled.append((n_bins, k, boundary, placed, want_bin))
    assert not misfiled, f"misfiled at bin edges: {misfiled[:5]}"


def test_confidence_of_exactly_one_lands_in_the_closed_final_bin():
    _, detail = compute_ece(["a"], np.array([[1.0, 0.0]]), CLASSES, n_bins=10)
    assert detail["bin_counts"][9] == 1
    assert sum(detail["bin_counts"]) == 1


def test_a_perfectly_calibrated_sample_gives_zero_ece():
    # 10 samples at confidence 0.5, 5 correct -> accuracy 0.5 == confidence 0.5
    probs = np.array([[0.5, 0.5]] * 10)
    y = ["a"] * 5 + ["b"] * 5
    assert compute_ece(y, probs, CLASSES)[0] == pytest.approx(0.0, abs=1e-12)


def test_degenerate_inputs_do_not_crash():
    assert expected_calibration_error([], np.zeros((0, 2)), CLASSES)[0] == pytest.approx(0.0)
    # one sample, confident and correct -> |1.0 - 0.9| = 0.1
    assert expected_calibration_error(["a"], np.array([[0.9, 0.1]]),
                                      CLASSES)[0] == pytest.approx(0.1)
    # one sample, confident and wrong -> |0.0 - 0.9| = 0.9
    assert expected_calibration_error(["b"], np.array([[0.9, 0.1]]),
                                      CLASSES)[0] == pytest.approx(0.9)


def test_entropy_and_cross_entropy_report_their_units():
    """Entropy defaults to bits; cross entropy to nats. They are not comparable."""
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)          # 1 bit
    assert shannon_entropy([1.0]) == pytest.approx(0.0)
    # -log_e(0.5) = ln 2
    assert cross_entropy_loss("a", np.array([[0.5, 0.5]]), CLASSES) == pytest.approx(math.log(2.0))
    assert cross_entropy_loss("a", np.array([[0.5, 0.5]]), CLASSES,
                              base=2.0) == pytest.approx(1.0)        # 1 bit
    assert cross_entropy_loss("a", np.array([[0.99, 0.01]]), CLASSES) == pytest.approx(-math.log(0.99))
    assert cross_entropy_loss("a", np.array([[0.99, 0.01]]), CLASSES,
                              base=2.0) == pytest.approx(-math.log2(0.99))
    assert confidence_margin(0.9, 0.1) == pytest.approx(0.8)
