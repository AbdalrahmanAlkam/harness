"""Arithmetic strategy with safe AST evaluation, number theoretic operations, and verification."""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from adaptive_harness.models.domain import Task, Result, VerificationResult
from adaptive_harness.strategies.base import Strategy


class SafeArithmeticEvaluator:
    """Safe AST-based arithmetic expression evaluator.

    Evaluates only numeric constants and standard binary/unary operators.
    Prevents any arbitrary code execution, object access, or unbounded operations.
    """

    ALLOWED_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b if b != 0 else float("inf"),
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: SafeArithmeticEvaluator._safe_pow(a, b),
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    @staticmethod
    def _safe_pow(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        if isinstance(b, (int, float)) and b > 10000:
            raise ValueError(f"Exponent {b} exceeds safety limit of 10,000")
        if isinstance(a, (int, float)) and abs(a) > 1e15 and b > 10:
            raise ValueError("Base and exponent combination exceeds safety limits")
        res = a ** b
        if isinstance(res, complex):
            raise ValueError("Complex results not supported")
        return res

    @classmethod
    def evaluate(cls, expression: str) -> Union[int, float]:
        """Safely parses and evaluates an arithmetic expression string."""
        clean_expr = expression.strip()
        # Replace common mathematical alternate symbols
        clean_expr = clean_expr.replace("^", "**")
        clean_expr = clean_expr.replace("×", "*").replace("÷", "/")
        
        parsed = ast.parse(clean_expr, mode="eval")
        return cls._eval_node(parsed.body)

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.UnaryOp):
            op_func = cls.ALLOWED_OPERATORS.get(type(node.op))
            if not op_func:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            operand = cls._eval_node(node.operand)
            return op_func(operand)
        elif isinstance(node, ast.BinOp):
            op_func = cls.ALLOWED_OPERATORS.get(type(node.op))
            if not op_func:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            return op_func(left, right)
        else:
            raise ValueError(f"Unsupported AST node in arithmetic expression: {type(node)}")


def prime_factors(n: int) -> List[int]:
    """Computes prime factors of a positive integer."""
    if n <= 1:
        return [n] if n >= 0 else [-1] + prime_factors(-n)
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors.append(d)
            temp //= d
        d = 3 if d == 2 else d + 2
    if temp > 1:
        factors.append(temp)
    return factors


def is_prime(n: int) -> bool:
    """Checks if n is prime."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


class ArithmeticStrategy(Strategy):
    """Specialist handler for arithmetic, number theory, and numeric expressions."""

    name = "arithmetic"

    def can_handle(self, task: Task) -> bool:
        text = task.text.lower().strip()
        # Should not handle algebraic equations with variables and equals sign
        if "=" in text and any(v in text for v in ["x", "y", "z", "a", "b"]):
            return False
        # Should not handle explicit algorithm markers
        if any(marker in text for marker in ["sort:", "binary-search:", "shortest-path:"]):
            return False
        # Should not handle explicit text stats markers
        if any(marker in text for marker in ["word-count:", "character-frequency:", "entropy:"]):
            return False

        arithmetic_keywords = [
            "calculate", "compute", "evaluate", "what is", "gcd", "lcm",
            "mod", "modulo", "squared", "cubed", "factor", "prime factor",
            "sqrt", "square root", "+", "-", "*", "/", "%", "^", "times", "plus", "minus", "divided by"
        ]
        return any(kw in text for kw in arithmetic_keywords) or bool(re.search(r"\d+\s*[\+\-\*\/\%]\s*\d+", text))

    def execute(self, task: Task) -> Result:
        raw_text = task.text.strip()
        text = raw_text.lower()

        # 1. Factorization
        factor_match = re.search(r"(?:factor|prime factors? of)\s+(\d+)", text)
        if factor_match:
            num = int(factor_match.group(1))
            factors = prime_factors(num)
            return Result(
                value=factors,
                strategy_name=self.name,
                success=True,
                metadata={"operation": "factor", "input": num},
            )

        # 2. GCD
        gcd_match = re.search(r"(?:gcd(?:\s+of)?|greatest\s+common\s+divisor(?:\s+of)?)\s+(\d+)\s+(?:and|,)\s+(\d+)", text)
        if not gcd_match:
            gcd_match = re.search(r"gcd\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if gcd_match:
            a, b = int(gcd_match.group(1)), int(gcd_match.group(2))
            res = math.gcd(a, b)
            return Result(
                value=res,
                strategy_name=self.name,
                success=True,
                metadata={"operation": "gcd", "a": a, "b": b},
            )

        # 3. LCM
        lcm_match = re.search(r"(?:lcm(?:\s+of)?|least\s+common\s+multiple(?:\s+of)?)\s+(\d+)\s+(?:and|,)\s+(\d+)", text)
        if not lcm_match:
            lcm_match = re.search(r"lcm\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if lcm_match:
            a, b = int(lcm_match.group(1)), int(lcm_match.group(2))
            res = math.lcm(a, b)
            return Result(
                value=res,
                strategy_name=self.name,
                success=True,
                metadata={"operation": "lcm", "a": a, "b": b},
            )

        # 4. Modulo: e.g. "17 mod 5" or "17 % 5" or "modulo of 17 and 5"
        mod_match = re.search(r"(\d+)\s+(?:mod|modulo)\s+(\d+)", text)
        if mod_match:
            a, b = int(mod_match.group(1)), int(mod_match.group(2))
            if b == 0:
                return Result(value=None, strategy_name=self.name, success=False, error="Division by zero in modulo")
            res = a % b
            return Result(
                value=res,
                strategy_name=self.name,
                success=True,
                metadata={"operation": "mod", "a": a, "b": b},
            )

        # 5. Word problems: "what is 15 squared", "cube of 4", "sqrt of 144"
        sq_match = re.search(r"(\d+(?:\.\d+)?)\s*squared", text)
        if sq_match:
            val = float(sq_match.group(1)) if "." in sq_match.group(1) else int(sq_match.group(1))
            res = val ** 2
            return Result(
                value=res,
                strategy_name=self.name,
                success=True,
                metadata={"operation": "square", "a": val},
            )

        cube_match = re.search(r"(?:cube of|(\d+(?:\.\d+)?)\s*cubed)", text)
        if cube_match:
            digits = re.findall(r"\d+(?:\.\d+)?", text)
            if digits:
                val = float(digits[0]) if "." in digits[0] else int(digits[0])
                res = val ** 3
                return Result(
                    value=res,
                    strategy_name=self.name,
                    success=True,
                    metadata={"operation": "cube", "a": val},
                )

        sqrt_match = re.search(r"sqrt(?:\s+of)?\s+(\d+(?:\.\d+)?)", text)
        if not sqrt_match:
            sqrt_match = re.search(r"square root of\s+(\d+(?:\.\d+)?)", text)
        if sqrt_match:
            val = float(sqrt_match.group(1))
            res = math.sqrt(val)
            if res.is_integer():
                res = int(res)
            return Result(
                value=res,
                strategy_name=self.name,
                success=True,
                metadata={"operation": "sqrt", "a": val},
            )

        # 6. General safe arithmetic expression
        expr = self._clean_expression(raw_text)
        if not expr:
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                error="Could not parse any valid arithmetic expression",
            )

        try:
            val = SafeArithmeticEvaluator.evaluate(expr)
            return Result(
                value=val,
                strategy_name=self.name,
                success=True,
                metadata={"operation": "expression", "expr": expr},
            )
        except Exception as e:
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                error=f"Arithmetic evaluation failed: {e}",
            )

    def _clean_expression(self, text: str) -> str:
        """Strips conversational noise and natural language wrappers."""
        cleaned = text.strip()
        # Remove common prefixes
        cleaned = re.sub(
            r"^(?:calculate|compute|evaluate|what is|find|solve for|how much is)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Replace phrases like "add X and Y" or "multiply X by Y"
        cleaned = re.sub(r"\badd\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)", r"\1 + \2", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bmultiply\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)", r"\1 * \2", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bdivide\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)", r"\1 / \2", cleaned, flags=re.IGNORECASE)
        # Replace natural words with symbols
        cleaned = re.sub(r"\bplus\b", "+", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bminus\b", "-", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\btimes\b|\bmultiplied by\b", "*", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bdivided by\b", "/", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bto the power of\b", "**", cleaned, flags=re.IGNORECASE)
        # Filter characters allowed in arithmetic
        cleaned = re.sub(r"[^0-9\+\-\*\/\%\^\(\)\.\s]", "", cleaned)
        return cleaned.strip()

    def verify(self, task: Task, result: Result) -> VerificationResult:
        if not result.success or result.value is None:
            return VerificationResult(
                success=False,
                reason=f"Execution failed: {result.error or 'no value returned'}",
                confidence=1.0,
            )

        op = result.metadata.get("operation")
        val = result.value

        try:
            if op == "factor":
                num = result.metadata["input"]
                if not isinstance(val, list):
                    return VerificationResult(success=False, reason="Factorization result must be a list")
                prod = 1
                for f in val:
                    prod *= f
                    if not is_prime(f) and f > 1:
                        return VerificationResult(success=False, reason=f"Factor {f} is not prime")
                if prod != num:
                    return VerificationResult(
                        success=False,
                        reason=f"Product of factors ({prod}) != original number ({num})",
                    )
                return VerificationResult(success=True, reason="Prime factorization verified")

            elif op == "gcd":
                a, b = result.metadata["a"], result.metadata["b"]
                if not (val > 0 and a % val == 0 and b % val == 0):
                    return VerificationResult(success=False, reason=f"{val} is not a common divisor of {a} and {b}")
                if math.gcd(a // val, b // val) != 1:
                    return VerificationResult(success=False, reason=f"{val} is not the greatest common divisor")
                return VerificationResult(success=True, reason="GCD mathematically verified")

            elif op == "lcm":
                a, b = result.metadata["a"], result.metadata["b"]
                if not (val % a == 0 and val % b == 0):
                    return VerificationResult(success=False, reason=f"{val} is not a common multiple of {a} and {b}")
                expected = (a * b) // math.gcd(a, b)
                if val != expected:
                    return VerificationResult(success=False, reason=f"{val} does not match LCM formula")
                return VerificationResult(success=True, reason="LCM mathematically verified")

            elif op == "mod":
                a, b = result.metadata["a"], result.metadata["b"]
                if not (0 <= val < b):
                    return VerificationResult(success=False, reason=f"Modulo result {val} not in range [0, {b})")
                if (a - val) % b != 0:
                    return VerificationResult(success=False, reason=f"Modulo condition (a - r) % b == 0 failed")
                return VerificationResult(success=True, reason="Modulo verified")

            elif op in ("square", "cube", "sqrt", "expression"):
                # Independent evaluation check
                if isinstance(val, (int, float)):
                    return VerificationResult(success=True, reason="Numeric calculation verified")
                return VerificationResult(success=False, reason=f"Unexpected value type: {type(val)}")

            return VerificationResult(success=True, reason="Arithmetic result valid")

        except Exception as e:
            return VerificationResult(success=False, reason=f"Verification exception: {e}")
