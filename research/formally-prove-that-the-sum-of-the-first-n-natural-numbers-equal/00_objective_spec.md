# Objective Specification — formally prove that the sum of the first n natural numbers equals n*(n+1)/2 and verify both in sympy and lean 4

## Objective

formally prove that the sum of the first n natural numbers equals n*(n+1)/2 and verify both
in sympy and lean 4

## Derivation Plan

Strategy selected: `gauss`

The kernel will attempt the following propositions (derived by the kernel):

- `PROP-GAUSS` (theorem) — Gauss's sum in division-free form
  - $2 S(n) = n (n + 1)$. The division-free form is used so that both proof tiers check identical content without a divisibility argument.
- `PROP-GAUSS-STEP` (lemma) — The induction step of Gauss's sum
  - $2 (S(k) + (k + 1)) = (k + 1) (k + 2)$, so the doubling identity is preserved by one step of the recursion.

## Definition of Solved

A result is settled only when all five invariants hold:

- **mathematical_soundness** — every derivation in `proofs/` is free of approximation and reached a clean verdict.
- **empirical_replication** — every experiment in `experiments/` reproduces its prediction at a pinned seed within 95%.
- **adversarial_clearance** — no red-team worker holds an open counterexample.
- **claim_adjudication** — the headline claim reached a decided verdict: PROVEN or DISPROVEN.
- **document_integrity** — `paper.typ` compiles to `paper.pdf` with zero Typst warnings.

## Divisions

- **Literature Lead** — Establish which prior results constrain the problem, and record every external claim as a citable evidence record rather than an assertion.
- **Theoretical Lead** — Convert the objective into exact, self-contained derivations whose scripts execute to exit code 0 with no floating-point approximation.
- **Empirical Lead** — Reproduce every theoretical prediction under a pinned seed and report the 95% interval, including the runs that contradict the theory.
- **Adversarial Lead** — Attempt to falsify every claim by constructing counterexamples, boundary cases, and unstated assumptions; grant clearance only when attempts fail.
- **Formal Proof Lead** — Formalise each proposition in Lean 4 and have the kernel machine-check it, so the published claim rests on a verified derivation and not only on symbolic computation.
