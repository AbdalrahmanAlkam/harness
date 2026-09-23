"""Tests for ArithmeticStrategy and safe evaluation."""

import pytest
from adaptive_harness.models.domain import Result, Task
from adaptive_harness.strategies.arithmetic import (
    ArithmeticStrategy,
    SafeArithmeticEvaluator,
    is_prime,
    prime_factors,
)


@pytest.fixture
def strategy():
    return ArithmeticStrategy()


def test_safe_arithmetic_evaluator_basic():
    assert SafeArithmeticEvaluator.evaluate("2 + 3 * 4") == 14
    assert SafeArithmeticEvaluator.evaluate("(10 - 4) / 2") == 3.0
    assert SafeArithmeticEvaluator.evaluate("2 ** 5") == 32
    assert SafeArithmeticEvaluator.evaluate("17 % 5") == 2
    assert SafeArithmeticEvaluator.evaluate("-15 + 20") == 5


def test_safe_arithmetic_evaluator_safety():
    # Disallowed AST nodes must raise ValueError
    with pytest.raises(ValueError):
        SafeArithmeticEvaluator.evaluate("__import__('os').system('ls')")

    with pytest.raises(ValueError):
        SafeArithmeticEvaluator.evaluate("open('/etc/passwd')")

    with pytest.raises(ValueError):
        SafeArithmeticEvaluator.evaluate("[x for x in range(10)]")


def test_prime_factors_and_is_prime():
    assert is_prime(2)
    assert is_prime(17)
    assert not is_prime(1)
    assert not is_prime(4)
    assert not is_prime(15)

    assert prime_factors(12) == [2, 2, 3]
    assert prime_factors(81) == [3, 3, 3, 3]
    assert prime_factors(17) == [17]


def test_arithmetic_strategy_execution(strategy):
    # Expressions
    t1 = Task(text="calculate 17 + 29")
    r1 = strategy.run_with_timing(t1)
    assert r1.success
    assert r1.value == 46
    v1 = strategy.verify(t1, r1)
    assert v1.success

    # Powers
    t2 = Task(text="what is 15 squared")
    r2 = strategy.run_with_timing(t2)
    assert r2.success
    assert r2.value == 225
    assert strategy.verify(t2, r2).success

    # GCD
    t3 = Task(text="gcd of 81 and 27")
    r3 = strategy.run_with_timing(t3)
    assert r3.success
    assert r3.value == 27
    assert strategy.verify(t3, r3).success

    # LCM
    t4 = Task(text="lcm of 12 and 18")
    r4 = strategy.run_with_timing(t4)
    assert r4.success
    assert r4.value == 36
    assert strategy.verify(t4, r4).success

    # Modulo
    t5 = Task(text="17 mod 5")
    r5 = strategy.run_with_timing(t5)
    assert r5.success
    assert r5.value == 2
    assert strategy.verify(t5, r5).success

    # Factor
    t6 = Task(text="factor 120")
    r6 = strategy.run_with_timing(t6)
    assert r6.success
    assert r6.value == [2, 2, 2, 3, 5]
    assert strategy.verify(t6, r6).success


def test_arithmetic_strategy_verification_failure(strategy):
    t = Task(text="gcd of 81 and 27")
    # Corrupt result
    bad_result = Result(
        value=15,  # 15 is not a common divisor of 81
        strategy_name="arithmetic",
        success=True,
        metadata={"operation": "gcd", "a": 81, "b": 27},
    )
    v = strategy.verify(t, bad_result)
    assert not v.success


def test_arithmetic_strategy_malformed_input(strategy):
    t = Task(text="calculate abc + def ???")
    r = strategy.run_with_timing(t)
    assert not r.success
    assert r.error is not None
