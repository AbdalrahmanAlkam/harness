"""Tests for EquationStrategy, polynomial interpolation, and substitution verification."""

import pytest
from adaptive_harness.models.domain import Result, Task
from adaptive_harness.strategies.equations import EquationStrategy


@pytest.fixture
def strategy():
    return EquationStrategy()


def test_linear_equation_solver(strategy):
    t = Task(text="solve 2*x + 7 = 19")
    r = strategy.run_with_timing(t)
    assert r.success
    assert r.value["roots"] == [6.0]
    assert r.value["variable"] == "x"
    v = strategy.verify(t, r)
    assert v.success


def test_linear_equation_negative_coefficients(strategy):
    t = Task(text="solve 5*y - 25 = 0")
    r = strategy.run_with_timing(t)
    assert r.success
    assert r.value["roots"] == [5.0]
    assert strategy.verify(t, r).success


def test_quadratic_equation_two_roots(strategy):
    t = Task(text="roots of x^2 - 5*x + 6 = 0")
    r = strategy.run_with_timing(t)
    assert r.success
    roots = r.value["roots"]
    assert len(roots) == 2
    assert sorted(roots) == [2.0, 3.0]
    assert strategy.verify(t, r).success


def test_quadratic_equation_difference_of_squares(strategy):
    t = Task(text="solve x^2 - 16 = 0")
    r = strategy.run_with_timing(t)
    assert r.success
    assert sorted(r.value["roots"]) == [-4.0, 4.0]
    assert strategy.verify(t, r).success


def test_equation_with_natural_language_preamble(strategy):
    t = Task(text="calculate the value of x in 3*x + 9 = 24")
    r = strategy.run_with_timing(t)
    assert r.success
    assert r.value["roots"] == [5.0]
    assert strategy.verify(t, r).success


def test_equation_verification_rejects_wrong_root(strategy):
    t = Task(text="solve 2*x + 7 = 19")
    # Wrong root: x = 10 (2*10 + 7 = 27 != 19)
    bad_result = Result(
        value={"roots": [10.0], "variable": "x"},
        strategy_name="equations",
        success=True,
        metadata={"lhs": "2*x + 7", "rhs": "19", "var": "x"},
    )
    v = strategy.verify(t, bad_result)
    assert not v.success
    assert "does not satisfy" in v.reason
