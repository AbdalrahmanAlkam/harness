"""Regressions for silent wrong answers in the math strategies.

Each test here corresponds to a defect that returned a *wrong* number and then
marked it verified. A strategy that reports a wrong answer as correct is worse
than one that fails, because the caller has no way to tell.
"""

from __future__ import annotations

import math

import pytest

from adaptive_harness.models.domain import Task
from adaptive_harness.strategies.algorithms import AlgorithmStrategy
from adaptive_harness.strategies.arithmetic import SafeArithmeticEvaluator
from adaptive_harness.strategies.equations import EquationStrategy


# --- division by zero is undefined, not infinity ---------------------------


@pytest.mark.parametrize("expr", ["5/0", "0/0", "1/(2-2)", "1/0"])
def test_division_by_zero_is_refused_not_infinity(expr: str):
    """`5/0` used to evaluate to inf and then verify as a numeric answer."""
    with pytest.raises(ValueError):
        SafeArithmeticEvaluator.evaluate(expr)


def test_ordinary_division_still_works():
    assert SafeArithmeticEvaluator.evaluate("10/4") == 2.5
    assert SafeArithmeticEvaluator.evaluate("2+2") == 4


def test_a_non_finite_result_is_never_verified():
    """A result of inf or nan must not pass the arithmetic verifier."""
    from adaptive_harness.models.domain import Result, VerificationResult
    from adaptive_harness.strategies.arithmetic import ArithmeticStrategy

    strategy = ArithmeticStrategy()
    infinite = Result(value=float("inf"), strategy_name=strategy.name, success=True,
                      metadata={"operation": "expression"})
    assert not strategy.verify(Task(text="x"), infinite).success
    not_a_number = Result(value=float("nan"), strategy_name=strategy.name, success=True,
                          metadata={"operation": "expression"})
    assert not strategy.verify(Task(text="x"), not_a_number).success


# --- the verifier must actually re-evaluate ---------------------------------


def test_the_verifier_catches_a_doctored_value():
    """A verifier that only checks `isinstance(x, (int, float))` passes anything.

    The check is now a genuine re-evaluation of the recorded expression, so a
    value that does not match its expression is rejected.
    """
    from adaptive_harness.models.domain import Result
    from adaptive_harness.strategies.arithmetic import ArithmeticStrategy

    strategy = ArithmeticStrategy()
    doctored = Result(value=999.0, strategy_name=strategy.name, success=True,
                      metadata={"operation": "expression", "expr": "2+2"})
    assert not strategy.verify(Task(text="x"), doctored).success

    honest = Result(value=4.0, strategy_name=strategy.name, success=True,
                    metadata={"operation": "expression", "expr": "2+2"})
    assert strategy.verify(Task(text="x"), honest).success


# --- small roots must not be truncated to zero -----------------------------


@pytest.mark.parametrize("equation,expected", [
    ("solve 123456789*x = 1", 1 / 123456789),
    ("solve 2*x = 8", 4.0),
    ("solve 3*x - 9 = 0", 3.0),
])
def test_a_linear_solution_is_exact(equation: str, expected: float):
    strategy = EquationStrategy()
    task = Task(text=equation)
    result = strategy.execute(task)
    assert result.success, result.error
    root = result.value["roots"][0]
    assert math.isclose(root, expected, rel_tol=1e-9, abs_tol=1e-15), (
        f"{equation}: got {root}, expected {expected}")
    assert strategy.verify(task, result).success


def test_a_tiny_root_is_not_reported_as_zero():
    """An absolute 6-decimal truncation turned 8.1e-9 into 0.0, and the absolute
    1e-5 matcher then accepted the wrong answer as a verified root."""
    strategy = EquationStrategy()
    task = Task(text="solve 123456789*x = 1")
    result = strategy.execute(task)
    root = result.value["roots"][0]
    assert root != 0.0
    assert math.isclose(123456789 * root, 1.0, rel_tol=1e-6)


def test_an_equation_with_a_tiny_coefficient_is_not_called_inconsistent():
    """b was recovered by differencing two samples near -c, so it came out 0.0
    and the solver reported an ordinary equation as having no solution."""
    strategy = EquationStrategy()
    task = Task(text="solve 0.0000001*x = 1000000")
    result = strategy.execute(task)
    assert result.success, result.error
    assert math.isclose(result.value["roots"][0], 1e13, rel_tol=1e-6)


def test_a_quadratic_still_works():
    strategy = EquationStrategy()
    result = strategy.execute(Task(text="solve x^2 - 5*x + 6 = 0"))
    assert result.success, result.error
    assert sorted(result.value["roots"]) == [2.0, 3.0]


# --- graph edge weights ----------------------------------------------------


def _shortest(edges: str):
    return AlgorithmStrategy().execute(
        Task(text=f"shortest-path: start=A end=D edges={edges}")).value


@pytest.mark.parametrize("edges,cost", [
    ("A-B:1e3,B-D:2", 1002.0),      # exponent
    ("A-B:1,000,B-D:2", 1002.0),   # thousands separator
    ("A-B:1.5,B-D:2", 3.5),        # decimal
    ("A-B:1,B-D:2", 3.0),          # plain
])
def test_edge_weights_parse(edges: str, cost: float):
    """A weight the regex could not read silently defaulted to 1.0, and the
    verifier rebuilt its map from the same mangled list, so it agreed."""
    assert _shortest(edges)["cost"] == pytest.approx(cost)


def test_a_negative_weight_is_refused_rather_than_hanging():
    """Dijkstra re-relaxes forever on a negative edge. Now that a signed weight
    parses, this must fail loudly instead of spinning."""
    result = AlgorithmStrategy().execute(
        Task(text="shortest-path: start=A end=D edges=A-B:-5,B-D:2"))
    assert not result.success
    assert "non-negative" in (result.error or "")


def test_the_task_marker_is_not_parsed_as_an_edge():
    """`shortest-path:` used to inject a phantom SHORTEST--PATH edge of weight 1."""
    result = _shortest("A-B:1,B-D:2")
    assert result["path"] == ["A", "B", "D"]
    assert result["cost"] == pytest.approx(3.0)
