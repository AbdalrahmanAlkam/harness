# Objective Specification — pareto heavy tailed network delay: critical index and variance-optimal balanced routing

## Objective

pareto heavy tailed network delay: critical index and variance-optimal balanced routing

## Derivation Plan

Strategy selected: `power-law-moments+balanced-allocation+quantile`

Multiple strategies matched (power-law-moments, balanced-allocation, quantile); all were
attempted so that one refutation is enough to sink the claim.

The kernel derived the following propositions:

- `PROP-01` (theorem) — Existence and value of the first moment
- `PROP-02` (theorem) — The critical index and the second moment
- `PROP-03` (theorem) — Closed form for the variance
- `PROP-04` (theorem) — No finite-variance objective below the critical index
- `PROP-05` (lemma) — Load imbalance contributes a non-negative excess
- `PROP-06` (theorem) — The balanced split minimises batch variance
- `PROP-07` (corollary) — Exact inversion of the Pareto distribution function

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
