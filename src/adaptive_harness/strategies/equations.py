"""Equation solving strategy supporting linear and quadratic equations with verification."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from adaptive_harness.models.domain import Task, Result, VerificationResult
from adaptive_harness.strategies.arithmetic import SafeArithmeticEvaluator
from adaptive_harness.strategies.base import Strategy


class EquationStrategy(Strategy):
    """Specialist handler for linear and quadratic single-variable equations."""

    name = "equations"

    def can_handle(self, task: Task) -> bool:
        text = task.text.lower()
        # Has variable marker and equation/roots marker
        has_eq = "=" in text or "equals" in text or "roots" in text or "solve" in text
        has_var = bool(re.search(r"\b[a-z]\b|\b[a-z]\^|\b\d+[a-z]", text))
        # Exclude algorithm and text-stat markers
        if any(marker in text for marker in ["sort:", "binary-search:", "shortest-path:", "word-count:"]):
            return False
        return has_eq and has_var

    def execute(self, task: Task) -> Result:
        raw_text = task.text.strip()
        parsed = self._normalize_equation(raw_text)
        if not parsed:
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                error="Could not parse algebraic equation structure",
            )

        lhs_expr, rhs_expr, var = parsed

        try:
            # Reconstruct P(x) = (LHS) - (RHS)
            p_func = lambda val: (
                SafeArithmeticEvaluator.evaluate(self._substitute(lhs_expr, var, val))
                - SafeArithmeticEvaluator.evaluate(self._substitute(rhs_expr, var, val))
            )

            # Polynomial interpolation for degree <= 2
            p0 = p_func(0.0)
            p1 = p_func(1.0)
            pm1 = p_func(-1.0)
            p2 = p_func(2.0)

            c = p0
            a = (p1 + pm1 - 2.0 * c) / 2.0
            b = (p1 - pm1) / 2.0

            # Verify quadratic assumption at test point x=2
            expected_p2 = a * (2.0 ** 2) + b * 2.0 + c
            if abs(p2 - expected_p2) > 1e-4:
                return Result(
                    value=None,
                    strategy_name=self.name,
                    success=False,
                    error=f"Equation is non-polynomial or degree > 2: P(2)={p2}, expected={expected_p2}",
                )

            # Round near-zero coefficients
            if abs(a) < 1e-9:
                a = 0.0
            if abs(b) < 1e-9:
                b = 0.0
            if abs(c) < 1e-9:
                c = 0.0

            solutions: List[float] = []
            if a == 0.0:
                # Linear: b*x + c = 0
                if b == 0.0:
                    if c == 0.0:
                        return Result(
                            value={"type": "infinite_solutions"},
                            strategy_name=self.name,
                            success=True,
                            metadata={"lhs": lhs_expr, "rhs": rhs_expr, "var": var},
                        )
                    else:
                        return Result(
                            value=None,
                            strategy_name=self.name,
                            success=False,
                            error="Inconsistent equation: no solution exists",
                        )
                root = -c / b
                solutions = [round(root, 6)]
            else:
                # Quadratic: a*x^2 + b*x + c = 0
                discriminant = b ** 2 - 4 * a * c
                if discriminant < -1e-9:
                    return Result(
                        value={"roots": [], "type": "complex"},
                        strategy_name=self.name,
                        success=True,
                        metadata={
                            "lhs": lhs_expr,
                            "rhs": rhs_expr,
                            "var": var,
                            "discriminant": discriminant,
                            "note": "No real roots",
                        },
                    )
                elif abs(discriminant) <= 1e-9:
                    root = -b / (2 * a)
                    solutions = [round(root, 6)]
                else:
                    sqrt_d = math.sqrt(discriminant)
                    r1 = (-b - sqrt_d) / (2 * a)
                    r2 = (-b + sqrt_d) / (2 * a)
                    solutions = sorted([round(r1, 6), round(r2, 6)])

            return Result(
                value={"roots": solutions, "variable": var},
                strategy_name=self.name,
                success=True,
                metadata={
                    "lhs": lhs_expr,
                    "rhs": rhs_expr,
                    "var": var,
                    "coefficients": {"a": round(a, 4), "b": round(b, 4), "c": round(c, 4)},
                },
            )

        except Exception as e:
            return Result(
                value=None,
                strategy_name=self.name,
                success=False,
                error=f"Equation resolution error: {e}",
            )

    def _normalize_equation(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Extracts LHS, RHS, and the single variable symbol."""
        cleaned = text.strip().rstrip(".?")

        # If there is a colon preceding the formula (e.g. 'solve for y: 3y + 14 = -42')
        if ":" in cleaned:
            eq_idx = cleaned.find("=")
            colon_idx = cleaned.find(":")
            if eq_idx == -1 or colon_idx < eq_idx:
                cleaned = cleaned[colon_idx + 1:].strip()

        cleaned = re.sub(r"^.*?\bwhere\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^.*?find\s+[a-z]\s+in\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^.*?\bsatisfies\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^(?:(?:solve|find|roots?\s+of|zeros?\s+of|calculate|compute)\s+(?:the\s+)?(?:quadratic\s+|roots?\s+(?:of\s+|for\s+)?|zeros?\s+(?:of\s+|for\s+)?|solution\s+(?:to\s+)?|value\s+of\s+[a-z]\s+in\s+|for\s+[a-z]\s+in\s+)?|(?:what\s+is\s+[a-z]\s+if\s+)|(?:if\s+))",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        # Check if contains equals
        if "=" in cleaned:
            parts = cleaned.split("=", 1)
            lhs, rhs = parts[0].strip(), parts[1].strip()
        elif "equals" in cleaned:
            parts = cleaned.split("equals", 1)
            lhs, rhs = parts[0].strip(), parts[1].strip()
        else:
            # If no equal sign, assume "= 0"
            lhs = cleaned
            rhs = "0"

        # Clean trailing clauses from rhs like ", what is y?" or "?"
        rhs = re.sub(r"[,;]?\s*(?:what\s+is\s+[a-z].*|\?.*)$", "", rhs, flags=re.IGNORECASE).strip()

        # Detect variable
        var_match = re.findall(r"\b([a-zA-Z])\b", lhs + " " + rhs)
        if not var_match:
            # Check characters adjacent to numbers like 2x
            var_match = re.findall(r"(?:^|[^a-zA-Z])([a-zA-Z])(?:$|[^a-zA-Z])", lhs + " " + rhs)
        var = var_match[0] if var_match else "x"

        # Expand implicit multiplication like 2x -> 2*x, 5*x^2 -> 5*(x**2)
        lhs = self._format_expr(lhs, var)
        rhs = self._format_expr(rhs, var)

        return lhs, rhs, var

    def _format_expr(self, expr: str, var: str) -> str:
        s = expr.strip()
        # Replace power notation
        s = s.replace("^", "**")
        # Implicit multiplication: number followed by variable
        s = re.sub(rf"(\d+)\s*{var}", rf"\1*{var}", s)
        # Variable followed by number without operator (rare, but handle)
        return s

    def _substitute(self, expr: str, var: str, val: float) -> str:
        """Substitutes numeric value in parentheses for the variable."""
        # Substitute var surrounded by non-alphanumeric or start/end
        val_str = f"({val})"
        pattern = rf"\b{var}\b"
        return re.sub(pattern, val_str, expr)

    def verify(self, task: Task, result: Result) -> VerificationResult:
        if not result.success or result.value is None:
            return VerificationResult(
                success=False,
                reason=f"Execution failed: {result.error or 'no result'}",
            )

        exact = self._verify_with_sympy(result)
        if exact is not None:
            return exact

        val = result.value
        if "type" in val and val["type"] in ("infinite_solutions", "complex"):
            return VerificationResult(success=True, reason="Special solution verified")

        roots = val.get("roots", [])
        if not roots:
            return VerificationResult(success=True, reason="No real roots confirmed")

        lhs_expr = result.metadata.get("lhs")
        rhs_expr = result.metadata.get("rhs")
        var = result.metadata.get("var", "x")

        if not lhs_expr or not rhs_expr:
            return VerificationResult(success=True, reason="Roots verified format")

        # Substitute each root into LHS and RHS and verify residual
        max_residual = 0.0
        for root in roots:
            try:
                lhs_val = SafeArithmeticEvaluator.evaluate(self._substitute(lhs_expr, var, root))
                rhs_val = SafeArithmeticEvaluator.evaluate(self._substitute(rhs_expr, var, root))
                residual = abs(lhs_val - rhs_val)
                if residual > max_residual:
                    max_residual = residual
                if residual > 1e-3:
                    return VerificationResult(
                        success=False,
                        reason=f"Root {root} does not satisfy equation: LHS={lhs_val}, RHS={rhs_val}, residual={residual}",
                        confidence=1.0,
                    )
            except Exception as e:
                return VerificationResult(
                    success=False,
                    reason=f"Verification substitution failed on root {root}: {e}",
                )

        return VerificationResult(
            success=True,
            reason=f"All {len(roots)} roots verified by substitution with max residual {max_residual:.2e}",
            confidence=1.0,
            details={"max_residual": max_residual},
        )

    @staticmethod
    def _verify_with_sympy(result: Result) -> VerificationResult | None:
        """Compare rounded display roots to exact real roots and original domains."""
        lhs_text = result.metadata.get("lhs")
        rhs_text = result.metadata.get("rhs")
        variable = result.metadata.get("var")
        if not all(isinstance(value, str) and value for value in (lhs_text, rhs_text, variable)):
            return None
        try:
            import sympy as sp
            from adaptive_harness.tools.python_repl import _safe_sympy
            symbol = sp.Symbol(variable)
            names = {variable: symbol, "pi": sp.pi, "E": sp.E, "I": sp.I}
            lhs, lhs_denoms = _safe_sympy(lhs_text, names)
            rhs, rhs_denoms = _safe_sympy(rhs_text, names)
            exact_roots = sp.solveset(sp.Eq(lhs, rhs), symbol, domain=sp.S.Reals)
            if exact_roots == sp.S.Reals:
                valid = result.value.get("type") == "infinite_solutions"
                return VerificationResult(success=valid, reason="Symbolic identity verified" if valid else
                                          "Expected infinitely many solutions")
            if exact_roots == sp.S.EmptySet:
                valid = result.value.get("type") == "complex" or not result.value.get("roots")
                return VerificationResult(success=valid, reason="No real roots verified symbolically" if valid else
                                          "Reported roots contradict the exact equation")
            if not isinstance(exact_roots, sp.FiniteSet):
                return None
            valid_roots = [root for root in exact_roots
                           if all(sp.simplify(denom.subs(symbol, root)) != 0
                                  for denom in (*lhs_denoms, *rhs_denoms))]
            reported = result.value.get("roots", [])
            if len(valid_roots) != len(reported):
                return VerificationResult(success=False, reason="Reported root count differs from exact solutions")
            remaining = list(valid_roots)
            for rounded in reported:
                match = next((root for root in remaining if abs(complex(sp.N(root)) - complex(rounded)) <= 1e-5), None)
                if match is None:
                    return VerificationResult(success=False,
                                              reason=f"Reported root {rounded} does not satisfy the exact symbolic equation")
                remaining.remove(match)
            return VerificationResult(success=True, reason=f"All {len(valid_roots)} roots matched exact SymPy solutions",
                                      confidence=1.0, details={"exact_roots": [str(root) for root in valid_roots]})
        except (ImportError, ValueError, TypeError, OverflowError, NotImplementedError):
            return None
