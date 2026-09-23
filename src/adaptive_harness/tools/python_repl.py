"""Isolated scientific Python execution and exact symbolic substitution checks."""

from __future__ import annotations

import ast
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import sysconfig

from adaptive_harness.tools.base import Tool, ToolResult


RUNNER = '''import ast, math, numpy as np, scipy, sympy as sp, sys
scope = {"math": math, "np": np, "numpy": np, "scipy": scipy, "sp": sp, "sympy": sp}
source = sys.stdin.read()
tree = ast.parse(source, filename="<repl>")
last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
exec(compile(tree, "<repl>", "exec"), scope)
if last is not None:
    value = eval(compile(ast.Expression(last.value), "<repl>", "eval"), scope)
    if value is not None:
        print(repr(value))
'''


class RunPythonReplTool(Tool):
    name = "run_python_repl"
    description = ("Run short scientific Python in a network-isolated, read-only Bubblewrap sandbox. "
                   "math, numpy as np, scipy, and sympy as sp are preloaded. State resets on each call.")
    parameters = {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python code; final expression is printed."},
    }, "required": ["code"]}

    def execute(self, code: str, **kwargs) -> ToolResult:
        if not isinstance(code, str) or not code.strip() or len(code) > 12_000:
            return ToolResult(success=False, output="", error="Code must be non-empty and at most 12,000 characters")
        try:
            ast.parse(code, filename="<repl>")
        except SyntaxError as exc:
            return ToolResult(success=False, output="", error=f"SyntaxError: {exc.msg} at line {exc.lineno}")
        bwrap = shutil.which("bwrap")
        if not bwrap or sys.platform != "linux":
            return ToolResult(success=False, output="", error="Python sandbox unavailable: Bubblewrap on Linux is required")
        base = Path(sys.base_prefix).resolve()
        site = Path(sysconfig.get_paths()["purelib"]).resolve()
        if not (base / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}").exists() or not site.is_dir():
            return ToolResult(success=False, output="", error="Python sandbox runtime is unavailable")
        mounts = []
        for system_path in ("/usr", "/lib", "/lib64"):
            if Path(system_path).exists():
                mounts.extend(["--ro-bind", system_path, system_path])
        command = [bwrap, "--unshare-net", "--unshare-pid", "--die-with-parent", "--clearenv",
                   "--setenv", "PATH", "/usr/bin", "--setenv", "PYTHONPATH", "/venv/site",
                   "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1",
                   *mounts, "--ro-bind", str(base), "/python", "--ro-bind", str(site), "/venv/site",
                   "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev", "--chdir", "/tmp", "--",
                   f"/python/bin/python{sys.version_info.major}.{sys.version_info.minor}", "-S", "-c", RUNNER]

        def limits() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            resource.setrlimit(resource.RLIMIT_FSIZE, (8_000_000, 8_000_000))
            resource.setrlimit(resource.RLIMIT_AS, (2_000_000_000, 2_000_000_000))

        try:
            result = subprocess.run(command, input=code, text=True, capture_output=True,
                                    timeout=8, env={"PATH": "/usr/bin"}, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Python sandbox timed out after 8 seconds")
        except OSError as exc:
            return ToolResult(success=False, output="", error=f"Python sandbox failed: {exc}")
        output = result.stdout[:8000].strip()
        error = result.stderr[-4000:].strip()
        return ToolResult(success=result.returncode == 0, output=output,
                          error=error if result.returncode else None,
                          metadata={"exit_code": result.returncode, "sandbox": "bubblewrap"})


class VerifyEquationTool(Tool):
    name = "verify_equation"
    description = "Exactly verify a proposed solution by SymPy substitution, including denominator-domain checks."
    parameters = {"type": "object", "properties": {
        "equation": {"type": "string", "description": "Equation such as x**2 - 2 = 0."},
        "variable": {"type": "string", "description": "Variable to solve for, such as x."},
        "solution": {"type": "string", "description": "Exact candidate, such as sqrt(2)."},
    }, "required": ["equation", "variable", "solution"]}

    def execute(self, equation: str, variable: str, solution: str, **kwargs) -> ToolResult:
        try:
            import sympy as sp
            if not variable.isidentifier() or variable.startswith("_") or len(variable) > 24:
                raise ValueError("Variable must be a short identifier")
            if any(len(value) > 500 for value in (equation, solution)):
                raise ValueError("Expression is too long")
            symbols = {variable: sp.Symbol(variable), "pi": sp.pi, "E": sp.E, "I": sp.I}
            left_text, separator, right_text = equation.partition("=")
            left, left_denoms = _safe_sympy(left_text.strip(), symbols)
            right, right_denoms = _safe_sympy(right_text.strip(), symbols) if separator else (sp.Integer(0), [])
            candidate, candidate_denoms = _safe_sympy(solution.strip(), symbols)
            if candidate.free_symbols:
                raise ValueError("Solution must not contain free symbols")
            if any(sp.simplify(denom.subs(symbols[variable], candidate)) == 0
                   for denom in (*left_denoms, *right_denoms, *candidate_denoms)):
                return ToolResult(success=False, output="", error="Candidate makes an original denominator zero")
            residual = sp.simplify((left - right).subs(symbols[variable], candidate))
            valid = residual == 0
            return ToolResult(success=bool(valid), output=f"Substitution residual: {residual}",
                              error=None if valid else "Candidate does not satisfy the equation exactly",
                              metadata={"substitution_verified": bool(valid), "residual": str(residual)})
        except ImportError:
            return ToolResult(success=False, output="", error="SymPy is required for exact equation verification")
        except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
            return ToolResult(success=False, output="", error=f"Equation verification failed: {exc}")


def _safe_sympy(expression: str, symbols: dict):
    """Build a SymPy expression from a deliberately small arithmetic AST grammar."""
    import sympy as sp
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 100:
        raise ValueError("Expression is too complex")
    denominators = []
    functions = {"sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log}

    def build(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in symbols:
            return symbols[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = build(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = build(node.left), build(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div):
                denominators.append(right)
                return left / right
            if isinstance(node.op, ast.Pow):
                if not right.is_number or abs(float(right)) > 100:
                    raise ValueError("Exponent is too large or symbolic")
                return left ** right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and len(node.args) == 1 and not node.keywords):
            return functions[node.func.id](build(node.args[0]))
        raise ValueError("Only arithmetic, named constants, and approved functions are allowed")

    return build(tree.body), denominators
