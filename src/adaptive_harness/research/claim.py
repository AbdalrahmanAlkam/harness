"""The claim under investigation and the verdict reached on it.

A research run has to answer one question: *what happened to the claim?* The
verdict is deliberately three-valued. Binary "solved / not solved" would force a
run to either overclaim (``PROVEN`` when nothing was shown) or underclaim
(``INCONCLUSIVE`` when a counterexample was in fact produced), and both are
worse than naming the third state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from adaptive_harness.research.roles import CONJECTURE_KINDS


class Verdict(str, Enum):
    """Outcome of attempting to settle a claim."""

    PROVEN = "PROVEN"
    DISPROVEN = "DISPROVEN"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNTESTED = "UNTESTED"

    @property
    def decided(self) -> bool:
        """Whether this verdict settles the question either way."""
        return self in (Verdict.PROVEN, Verdict.DISPROVEN)


# Exit codes a self-adjudicating derivation script uses. The proof runner
# reports the raw code, so the verdict is recoverable from the receipt alone.
EXIT_HOLDS = 0
EXIT_COUNTEREXAMPLE = 3

_VERDICT_FROM_EXIT = {
    EXIT_HOLDS: Verdict.PROVEN,
    EXIT_COUNTEREXAMPLE: Verdict.DISPROVEN,
}


@dataclass(frozen=True)
class ClaimAdjudication:
    """A decided (or explicitly undecided) outcome for one proposition."""

    prop_id: str
    statement: str
    verdict: Verdict
    exit_code: int | None = None
    finding: str = ""
    script: str = ""
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"prop_id": self.prop_id, "statement": self.statement,
                "verdict": self.verdict.value, "exit_code": self.exit_code,
                "finding": self.finding, "script": self.script, "hash": self.sha256}


class DeciderError(RuntimeError):
    """The exact-decider guard itself failed, as opposed to unusable input.

    Kept distinct so that a harness defect cannot masquerade as "this claim is
    undecidable" — the two demand opposite responses: one is the caller's problem,
    the other is the model's.
    """


def verdict_from_exit(exit_code: int | None) -> Verdict:
    """Map a derivation script's exit code onto a verdict.

    Any other non-zero code means the script could not decide: a syntax error,
    a timeout, or an exception. That is ``INCONCLUSIVE``, never ``DISPROVEN`` —
    a broken proof attempt is not evidence against a claim.
    """
    return _VERDICT_FROM_EXIT.get(exit_code if exit_code is not None else -1, Verdict.INCONCLUSIVE)


@dataclass(frozen=True)
class TopicPlan:
    """What the kernel decided to attempt for a topic, and why."""

    topic: str
    propositions: tuple["Proposition", ...] = ()
    strategy: str = "none"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "strategy": self.strategy, "notes": self.notes,
                "propositions": [item.to_dict() for item in self.propositions]}


@dataclass(frozen=True)
class Proposition:
    """A theorem with hypotheses, a formal statement, a proof, and its decider.

    ``script`` must be self-adjudicating: exit 0 when the proposition holds,
    exit 3 when a counterexample is found, and anything else when the attempt
    could not decide. That contract is what lets the swarm reach a verdict
    without a human reading the output.

    The mathematical fields exist so the generated paper can typeset real
    mathematics instead of restating a script log. ``hypotheses``,
    ``statement``, ``proof_sketch`` and ``display`` may embed Typst inline math
    delimited by ``$``; ``display`` holds block equations.
    """

    prop_id: str
    kind: str
    name: str
    hypotheses: tuple[str, ...] = ()
    statement: str = ""
    proof_sketch: str = ""
    consequence: str = ""
    display: tuple[str, ...] = ()
    notation: tuple[tuple[str, str], ...] = ()
    script: str = ""
    rationale: str = ""
    lean_statement: str = ""
    # The exactly-decidable form of the claim, as SymPy source: "2 + 2 == 4".
    #
    # This field is what makes a claim decidable by computation *without*
    # inventing anything. A claim whose natural language describes how a proof
    # assistant reduces terms — "Nat.succ n + m reduces definitionally" — has no
    # exact computational content, and a worker asked to decide it in SymPy can
    # only build a Lean evaluator in Python. That is a category error, and it
    # produced 15,000-line scripts that failed every time. When the Director
    # supplies this field the kernel writes the decider itself, so the
    # computational tier is decided rather than attempted.
    sympy_expression: str = ""
    symbols: tuple[str, ...] = ()

    @property
    def exactly_decidable(self) -> bool:
        """Whether the claim carries a form exact computation can settle."""
        return bool(self.sympy_expression.strip())

    @property
    def is_goal(self) -> bool:
        """Whether this states the problem rather than a decidable result.

        Goals are carried through the run as the target — printed in the paper as
        open, and never adjudicated — while the decidable claims beneath them are
        what a run can actually settle.
        """
        return self.kind.strip().lower() in CONJECTURE_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {"prop_id": self.prop_id, "kind": self.kind, "name": self.name,
                "hypotheses": list(self.hypotheses), "statement": self.statement,
                "lean_statement": self.lean_statement,
                "sympy_expression": self.sympy_expression,
                "symbols": list(self.symbols),
                "consequence": self.consequence,
                "notation": [list(item) for item in self.notation],
                "script_lines": len(self.script.splitlines())}


@dataclass
class ClaimLedger:
    """Aggregate the adjudications of one run into a single reached result."""

    adjudications: list[ClaimAdjudication] = field(default_factory=list)

    def add(self, adjudication: ClaimAdjudication) -> ClaimAdjudication:
        self.adjudications.append(adjudication)
        return adjudication

    @property
    def proven(self) -> list[ClaimAdjudication]:
        return [item for item in self.adjudications if item.verdict is Verdict.PROVEN]

    @property
    def disproven(self) -> list[ClaimAdjudication]:
        return [item for item in self.adjudications if item.verdict is Verdict.DISPROVEN]

    @property
    def inconclusive(self) -> list[ClaimAdjudication]:
        return [item for item in self.adjudications if item.verdict is Verdict.INCONCLUSIVE]

    @property
    def headline(self) -> Verdict:
        """The run's single reached result.

        A single counterexample refutes the claim, so ``DISPROVEN`` outranks
        ``PROVEN``: one broken sub-claim is enough to sink the whole claim,
        whereas proving every sub-claim is the only way to earn ``PROVEN``.
        """
        if self.disproven:
            return Verdict.DISPROVEN
        if self.proven and not self.inconclusive:
            return Verdict.PROVEN
        if self.proven:
            return Verdict.INCONCLUSIVE
        return Verdict.UNTESTED if not self.adjudications else Verdict.INCONCLUSIVE

    @property
    def decided(self) -> bool:
        return self.headline.decided

    def summary(self) -> str:
        headline = self.headline
        if not self.adjudications:
            return "No proposition was derived, so the claim was not tested."
        parts = [f"{len(self.proven)} proven", f"{len(self.disproven)} disproven",
                 f"{len(self.inconclusive)} inconclusive"]
        tail = ""
        if self.disproven:
            first = self.disproven[0]
            tail = f" Refuted by {first.prop_id}: {first.finding[:200]}"
        return f"Headline {headline.value} of {len(self.adjudications)} proposition(s) ({', '.join(parts)}).{tail}"

    def to_dict(self) -> dict[str, Any]:
        return {"headline": self.headline.value, "decided": self.decided,
                "counts": {"proven": len(self.proven), "disproven": len(self.disproven),
                           "inconclusive": len(self.inconclusive)},
                "summary": self.summary(),
                "adjudications": [item.to_dict() for item in self.adjudications]}


# --- claim extraction -----------------------------------------------------

# A claim written in a checkable form: "<sympy expression> == <sympy expression>".
_EQUATION = re.compile(r"^\s*(?P<lhs>.+?)\s*==\s*(?P<rhs>.+?)\s*$", re.S)
# An explicitly symbolic claim passed on the command line, e.g.
# "symbols: x w" plus "x**2 - x**2 == 0".
_SYMBOLS = re.compile(r"^[A-Za-z_][A-Za-z_0-9, ]*$")


@dataclass(frozen=True)
class ParsedClaim:
    """A user-supplied claim, normalized for the kernel."""

    statement: str
    lhs: str | None = None
    rhs: str | None = None
    symbols: tuple[str, ...] = ()

    @property
    def machine_checkable(self) -> bool:
        return bool(self.lhs and self.rhs and self.symbols)


def parse_claim(statement: str, symbols: Sequence[str] = ()) -> ParsedClaim:
    """Extract a checkable identity from a claim, if one is present.

    Only a literal ``lhs == rhs`` with declared free symbols is treated as
    machine-checkable. Anything else stays prose, and the kernel will say so
    rather than inventing a formalisation the user did not intend.
    """
    text = (statement or "").strip()
    names = tuple(name.strip() for name in symbols if name.strip())
    match = _EQUATION.match(text) if "==" in text else None
    if match and names:
        return ParsedClaim(statement=text, lhs=match.group("lhs").strip(),
                           rhs=match.group("rhs").strip(), symbols=names)
    if names and _SYMBOLS.match(text):
        return ParsedClaim(statement=text, symbols=names)
    return ParsedClaim(statement=text)


EXACT_DECIDER_HEADER = '''"""{statement}

Exact-computation decider generated by the Adaptive Agent Harness from the
claim's declared expression. The claim text is quoted; the *decision* is made by
SymPy on the expression below, not by a language model.

Decided expression: {expression}
Exit-code contract: {holds} = the identity holds exactly; {counter} = a
counterexample was exhibited; any other code = the attempt could not decide.
"""
import sys

import sympy
from sympy import Eq, simplify, together, cancel, nsimplify, S, zoo, nan, oo

HOLD = {holds}
COUNTEREXAMPLE = {counter}
UNDECIDED = 4

{body}
'''


def free_symbol_names(expression: str) -> tuple[str, ...]:
    """The free symbols of an expression, determined semantically.

    Scanning the text for identifiers is wrong: it cannot tell ``sqrt`` (a SymPy
    function) from ``x`` (a variable), and binding ``sqrt`` to a Symbol turns a
    valid expression into ``'Symbol' object is not callable``. Letting SymPy parse
    the expression is the only reliable way to know which names are variables.
    """
    import sympy

    names: set[str] = set()
    for side in expression.split("=="):
        try:
            names |= {str(item) for item in sympy.sympify(side).free_symbols}
        except (sympy.SympifyError, SyntaxError, TypeError, AttributeError, ValueError):
            return ()
    return tuple(sorted(names))


def _decidable_pair(expression: str) -> tuple[str, str] | None:
    """Split and validate ``lhs == rhs`` for exact decision, or return None.

    Everything here is defensive on purpose. This input is authored by a language
    model, and two ordinary-looking mistakes used to take the whole run down:

    * a second ``==`` in the text (``a == b == c``, ``x == 1 and y == 2``) leaves
      a comparison on the right-hand side, which ``sympify`` evaluates to a Python
      ``bool`` — and a bool has no ``free_symbols``;
    * text that is not SymPy at all raises ``SympifyError`` from deep inside the
      parser.

    A function that raises on model output is a single point of failure for the
    entire run, so every rejection path returns None and the caller falls back to
    the model-authored path.
    """
    try:
        import sympy
        from sympy.core.relational import Relational
        from sympy.logic.boolalg import BooleanFunction

        text = (expression or "").strip()
        if "==" not in text:
            return None
        match = _EQUATION.match(text)
        if not match:
            return None
        lhs, rhs = match.group("lhs").strip(), match.group("rhs").strip()
        if not lhs or not rhs or "==" in lhs or "==" in rhs:
            # A chained or compound comparison is not a single identity.
            return None
        left, right = sympy.sympify(lhs), sympy.sympify(rhs)
        # sympify returns a plain bool for a relational expression; only real
        # SymPy expressions have free symbols.
        if not isinstance(left, sympy.Expr) or not isinstance(right, sympy.Expr):
            return None
        # A lambda, piecewise, or relational term is a function or a proposition,
        # not the value of a claim. Deciding one would manufacture a verdict from
        # something that is not a mathematical assertion.
        forbidden = (sympy.Lambda, sympy.Piecewise, Relational, BooleanFunction)
        if any(left.has(kind) or right.has(kind) for kind in forbidden):
            return None
        if left.has(sympy.Float) or right.has(sympy.Float):
            # A Float is approximate arithmetic. The claim is not exact, and a
            # decider built on one would settle a statement that was never posed.
            return None
        # An undefined or infinite residual means the expression could not be
        # evaluated, not that the claim is false. Reporting that as a refutation
        # would be a false accusation, which is the one outcome worse than silence.
        infinite = (sympy.zoo, sympy.nan, sympy.oo, -sympy.oo)
        if any(left.has(term) or right.has(term) for term in infinite):
            return None
        if left.free_symbols != right.free_symbols:
            # Different free variables on the two sides is a malformed claim, and
            # treating it as an identity would be a category error.
            return None
        return lhs, rhs
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError, AttributeError) as exc:
        # Expected: the text is not SymPy, so there is nothing to decide.
        return None
    except RecursionError:
        return None
    except Exception as exc:  # noqa: BLE001
        # Unexpected. A broad catch here once hid a broken guard that silently
        # disabled *every* decider, so the failure is surfaced rather than
        # swallowed into "this claim is undecidable".
        raise DeciderError(
            f"the exact-decider guard failed unexpectedly on {expression!r}: "
            f"{type(exc).__name__}: {exc}") from exc


def exact_decider(expression: str, *, statement: str = "",
                  symbols: Sequence[str] = ()) -> str:
    """Build a self-adjudicating decider for ``lhs == rhs`` in exact arithmetic.

    The script reduces the difference to zero with SymPy and, for a claim with
    free symbols, searches a small exact grid for a counterexample. Three
    properties matter and are all deliberate:

    * **The decision is the kernel's, not the model's.** The Director supplies the
      expression; SymPy settles it. Nothing is asserted that a language model did
      not compute.
    * **It refuses to guess.** A residual that is not provably zero yields a
      refutation (for a closed claim) or exit 4 (undecided, for one with free
      symbols), never a pass.
    * **It never raises.** The expression is model-authored, so anything that
      cannot be parsed yields an empty string and the caller falls back to the
      model-authored path. A parser crash here would abort the entire run.
    """
    import sympy

    pair = _decidable_pair(expression)
    if pair is None:
        return ""
    lhs, rhs = pair
    try:
        text = (expression or "").strip()
        names = free_symbol_names(text)
    except Exception:  # noqa: BLE001 - defensive; free_symbol_names is already guarded
        names = ()
    declared = tuple(name for name in symbols if name.strip() and name in names)
    if not declared:
        declared = names
    declarations = "\n".join(
        f"{name} = sympy.Symbol({name!r}, integer=True)" for name in declared)
    body = [
        declarations or "# the claim is closed: it has no free symbols",
        "",
        f"LEFT = sympy.sympify({lhs!r})",
        f"RIGHT = sympy.sympify({rhs!r})",
        "",
        "",
        "def _difference():",
        "    return simplify(together(LEFT - RIGHT))",
        "",
        "",
        "def _decide():",
        "    difference = _difference()",
        "    if difference == 0:",
        "        print('[HOLD] the identity holds exactly; the residual is 0')",
        "        return HOLD",
        "    # An undefined or infinite residual is a failure to evaluate, not a",
        "    # refutation. Deciding the claim here would be a false accusation.",
        "    if difference.has(zoo, nan, oo, -oo):",
        "        print('[UNDECIDED] the residual is not a finite value: '",
        "              + str(difference))",
        "        return UNDECIDED",
        "    if not difference.free_symbols:",
        "        # A closed expression with a provably nonzero residual is a",
        "        # refutation, not an absence of proof: there is nothing left to vary.",
        "        print('[COUNTEREXAMPLE] the closed claim is false; the residual is '",
        "              + str(difference))",
        "        return COUNTEREXAMPLE",
        "    return _search(difference)",
    ]
    if declared:
        body += [
            "",
            "",
            "def _search(difference):",
            "    # A small exact grid: enough to exhibit a counterexample, never enough",
            "    # to prove a universal statement. Finding nothing here is not evidence",
            "    # of truth, which is why the identity must still reduce to 0 above.",
            "    import itertools",
            "    names = " + repr(list(declared)),
            "    for combo in itertools.product(range(-3, 4), repeat=len(names)):",
            "        values = dict(zip(names, combo))",
            "        try:",
            "            probe = simplify(together(difference.subs(values)))",
            "        except (TypeError, ValueError):",
            "            continue",
            "        if probe != 0:",
            "            print('[COUNTEREXAMPLE] the residual is '",
            "                  + str(difference.subs(values)) + ' at ' + str(values))",
            "            return COUNTEREXAMPLE",
            "    print('[UNDECIDED] no counterexample on the sampled grid, but the'",
            "          ' residual is not identically zero, so the claim is not decided')",
            "    return UNDECIDED",
        ]
    body += [
        "",
        "",
        "if __name__ == '__main__':",
        "    sys.exit(_decide())",
    ]
    return EXACT_DECIDER_HEADER.format(
        statement=statement or text, expression=text, holds=EXIT_HOLDS,
        counter=EXIT_COUNTEREXAMPLE, body="\n".join(body))


def collect_symbols(*sources: str) -> tuple[str, ...]:
    """Union of declared symbol names across sources, order preserved."""
    seen: list[str] = []
    for source in sources:
        for name in (source or "").replace("(", " ").replace(")", " ").replace(",", " ").split():
            cleaned = name.strip()
            if _SYMBOLS.match(cleaned) and cleaned not in seen:
                seen.append(cleaned)
    return tuple(seen)


def format_verdict_table(adjudications: Sequence[Mapping[str, Any]]) -> str:
    """Plain-text verdict table for the CLI and the run log."""
    if not adjudications:
        return "no proposition was adjudicated"
    width = max(len(str(item.get("prop_id", ""))) for item in adjudications)
    lines = []
    for item in adjudications:
        lines.append(f"  {str(item.get('prop_id')):<{width}}  {str(item.get('verdict')):<13} "
                     f"{str(item.get('statement', ''))[:70]}")
    return "\n".join(lines)
