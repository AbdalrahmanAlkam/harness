# Changelog

All notable changes to Adaptive Agent Harness. The five entries below cover the
work in this push; earlier history is summarized in `git log`.

## [Unreleased]

### Added
- **Autonomous hierarchical research swarm** (`adaptive-harness research "<topic>"`,
  `src/adaptive_harness/research/`). An Executive Director owns the objective and
  the Relentless Convergence Loop; four Division Leaders (Literature, Theory,
  Empirical, Adversarial) each spawn an unbounded worker pool on demand via
  `spawn_subagent(parent_id, role_name, directive, allowed_tools, budget_tokens)`.
  A topic is reported `SOLVED` only when all four invariants hold simultaneously:
  exact proofs exiting 0, seeded replications inside a 95% interval, red-team
  clearance, and a warning-free `paper.typ` → `paper.pdf` build.
- **Proof receipts with mechanical exactness enforcement.** Every script in
  `proofs/` is scanned by AST for floating-point literals and approximating calls
  (`float`, `evalf`, `N`, …) *before* execution, then run in a subprocess with
  networking disabled; only exit code 0 admits it as evidence. This caught a real
  error in the harness's own first draft of the Pareto second moment.
- **Empirical replication receipts.** Experiments run under an explicitly pinned
  seed, their data artifacts are hashed, and declared predictions are adjudicated
  against a Student-t 95% interval with a small-sample correction.
- **Hash-chained `comm_ledger.jsonl`.** Append-only inter-agent ledger where each
  entry carries the previous entry's SHA-256. `verify()` recomputes the whole
  chain and detects edits, reordering, and truncation. Opening a tampered ledger
  reports through `verify()` rather than raising, so an auditor can always
  inspect the evidence of tampering.
- **Relentless Convergence Loop that is provably terminating.** The loop is
  stagnation-limited rather than turn-limited. Progress is a new *minimum* in
  outstanding gaps (a high-water mark), which makes a flapping invariant
  oscillating 4→3→4→3 impossible to mistake for progress; stagnation escalates
  the worker budget, and when escalation can no longer grow the run concedes with
  an honest `STAGNATION_ABORT` / UNSOLVED rather than a false success.
- **`compile_typst` tool and generated papers.** Typst is resolved from `PATH`,
  then the `typst` Python wrapper, then reported as unavailable. Compilation
  treats *any* Typst diagnostic as a build break, because a paper with an
  unresolved reference or a bad figure path is a defect, not a warning.
  `paper.typ` is generated from receipts, so nothing can be typeset that was not
  proven or measured.
- **Deterministic vector figures** (`research/figures.py`) rendered as SVG, so
  the same receipts always yield byte-identical figures.
- **Research division prompts** (`research.division.*`, `research.director`) in
  the central prompt registry, editable without touching code.
- **Model-facing research tools** (`spawn_subagent`, `scale_division`,
  `verify_proofs`, `run_experiments`) plus `DeveloperAgent.enable_research()` and
  `adaptive-harness dev --research TOPIC`, so a live agent can recruit and retire
  workers mid-task instead of being locked to a fixed roster. The tools and the
  Director system-prompt framing are gated by the same predicate and appear only
  in the investigative modes — never in `security`/audit.

### Added
- **Autonomous derivation kernel** (`research/synthesis.py`). This is what makes
  research mode *research* rather than verification: given a topic, the kernel
  constructs machine-checkable propositions from definitions, emits a
  self-adjudicating SymPy script for each, and lets the exit code decide the
  verdict (`0` = PROVEN, `3` = DISPROVEN, anything else = INCONCLUSIVE). It also
  generates the corroborating simulations. A bare topic now yields proofs,
  experiments, a verdict, and a paper with no manual preparation.
- **Three-valued claim adjudication** (`research/claim.py`). `PROVEN` /
  `DISPROVEN` / `INCONCLUSIVE` / `UNTESTED`, with the headline verdict prioritising
  a single refutation over any number of proofs, since one broken sub-claim sinks
  the claim. Refutation requires an *exhibited* exact rational witness, so
  `exp(x) == 1+x` falls at `x=1` and `sqrt(x^2) == x` at `x=-1`; a script that
  merely fails to simplify is never counted as evidence against a claim.
- **`claim_adjudication` invariant and `--claim` / `--symbols` CLI options.** A
  supplied identity in SymPy syntax is decided directly instead of being matched
  against the topic library.
- **The paper is now a mathematical paper** (`research/paper.py`): title,
  abstract, introduction, notation table, numbered theorems with explicit
  hypotheses, statements typeset in 2D math, proofs ending in a qed box, a
  consequence and a verdict line per theorem, then empirical corroboration, the
  adversarial audit, a conclusion matching the verdict, references, and appendices
  carrying the ledger and receipts. Section numbers are generated rather than
  delegated to a Typst counter, which prefixed every heading with a spurious "0.".
- **Self-adjudicating experiment generator** with pinned seeds, Student-t 95%
  intervals, and a rank-order prediction that a wrong theory would break.

### Added
- **Per-file Lean receipts** at `proofs/<theorem>.lean.receipt.json`, carrying the
  digest, verdict, exit code, located diagnostics, unsolved goals, placeholders,
  axiom set, compiler output, and verification time, so a later audit reads a
  durable record rather than re-running the compiler.
- **Lean 4 as the machine-checked epistemic proof engine.** `RunLeanProofTool`
  (`run_lean_proof`) compiles and adjudicates Lean sources, and a sixth
  invariant, `formal_verification`, requires that any theorem the paper asserts
  is machine-checked. The paper gains a *Formal Foundations* section with a
  certification box per proof and an appendix listing the full sources.
- **The zero-sorry invariant, enforced three independent ways.** Established
  empirically before implementation: **Lean exits 0 on a proof containing
  `sorry`** — it emits a warning, not an error — so a verifier trusting the exit
  code would certify a false theorem. The tool therefore requires (1) no
  `sorry`/`admit` token in the source, after comments and string literals are
  stripped; (2) no "declaration uses 'sorry'" diagnostic; and (3) no dependency
  on `sorryAx`, reported by a kernel-emitted `#print axioms` audit. The third is
  the strongest, being vouched for by Lean rather than inferred from text.
  `test_tampered_lean_file_blocks_the_run` pins this end to end.
- **A two-tier proof standard** (`research/lean_gate.py`). Tier 1 is SymPy
  computation, Tier 2 is Lean-checked logic. The requirement tracks the verdict:
  a PROVEN theorem must be certified; a DISPROVEN result is certified by its
  exact witness, since a Lean proof of a falsehood is neither expected nor
  meaningful; and an INCONCLUSIVE verdict publishes no theorem, so the tier does
  not apply. An empty formal tier on an asserted theorem blocks publication.
- **A Formal Proof Lead division** with a Lean Formaliser, Tactic Specialist, and
  Axiom Auditor, plus `LEAN_PROOF_SUBMISSION` / `_VERIFIED` / `_REJECTED` ledger
  actions and `proofs/lean/` beside `proofs/` so the tiers cannot contaminate
  each other.
- **A `gauss` strategy in the derivation kernel**, so the acceptance example is
  covered by both tiers. SymPy discharges the algebra of the induction step;
  Lean discharges the induction itself, which SymPy cannot do.

### Added
- **Dynamic topic formulation** (`research/formulate.py`). A topic no library
  template covers is no longer abandoned: the Director is asked to formulate
  candidate claims, which are then decided by the kernel. The division of
  labour is the point — the model states the claim, the kernel writes the script
  that judges it, so no model-authored code is ever executed and a model
  proposing a false identity gets it *refuted* rather than believed. Claims
  that are not exact symbolic identities (float literals, `N()`, prose) are
  dropped rather than admitted, and a declined or unusable reply is reported as
  such rather than passed off as coverage.
- **Research worker autonomy**: workers are now multi-step, tool-using
  subagents rather than one-shot completions (see below).

### Fixed
- **`MockLLMClient` was unusable by the agent.** Its `complete` did not accept
  `tier`, `reasoning_effort`, or `reasoning_budget_tokens`, which the agent always
  passes, so every offline call raised `TypeError` and degraded to "Model request
  failed". The documented offline engine therefore never worked end to end. It
  now accepts and records them, exposed as `last_call` for assertions.
- **A provider fault reported only the exception type**, collapsing every
  failure into the same `provider_error` and leaving an operator unable to
  distinguish a bad credential from a network outage. The message is now
  carried through.
- **A declining red team blocked publication of a kernel-certified refutation.**
  The clearance requirement was evaluated before the DISPROVEN branch, so an
  inconclusive red team prevented a result the kernel had already certified with
  an exact witness. The refutation is now judged first.
- **`compile_typst` and `run_lean_proof` were unreachable from a task.** Both were
  absent from `domain_tool_names[DomainMode.RESEARCH]`, and `run_lean_proof` was
  not on the toolbelt at all, so a research agent could not build the paper it was
  writing nor machine-check a proof even though the swarm used the tool
  internally. Both are now registered and exposed in RESEARCH and SCIENCE, and
  both are carried by the `research_active` set.
- **A bare `axiom` declaration was accepted.** `sorry` and `admit` were refused,
  but an assumed constant was not, so a theorem could lean on an unproved premise.
  `axiom` is now a placeholder token, matched as a whole word so `#print axioms`
  is not a false positive. The axiom-dependency audit would have caught it, but
  only after the fact and with a far less obvious message.
- **Live mode could never converge.** Any non-empty falsification response was read
  as a counterexample, so with `--author` the adversarial gate never cleared. The
  red team now requires a structured verdict
  (`{"falsified": true|false, "finding": "..."}`); an unparseable reply is recorded
  as inconclusive and grants no clearance, so a chatty model cannot rubber-stamp a
  claim. `FalsificationVerdict.parse` covers the ambiguous cases explicitly.
- **Vacuous red-team clearance.** Adversarial clearance passed even when no proof
  existed, letting a run clear a claim nobody had made. It now requires a claim to
  exist first.
- **Runaway audit-log growth in the convergence loop.** A gate that oscillated
  between gap counts reset the stagnation counter on every "improvement",
  producing an unbounded loop that appended ~15,000 ledger entries per second
  and grew `comm_ledger.jsonl` past 2 GB. Progress is now measured against a
  monotone high-water mark, receipts are recorded once per distinct verdict
  rather than once per evaluation, and an absolute cycle ceiling guarantees
  termination.
- **Quadratic ledger reads.** `read_raw()` re-parsed the entire ledger on every
  read, and the audit log renders the message tree each cycle. Parses are now
  cached on `(size, mtime)`, invalidated on append or external edit.
- **Typst 0.15 compatibility.** `raw()` takes a string rather than content or
  `text=`; `width` belongs to `image()` rather than `figure()`; and escaping `_`
  inside a string literal injected a backslash that made figure paths unloadable.
  Added a separate literal escaper for string contexts.

### Notes
- Formal proofs are **core-Lean only**: no imports beyond the prelude. Mathlib
  could not be installed on the development machine (its cache needs roughly
  5 GB, and the fetch exhausted the tmpfs), and a proof that needs a
  multi-gigabyte dependency is not reproducible anyway. Core Lean has no `ring`,
  `linarith`, or `ring_nf`, so polynomial identities are distributed by hand.
  The tool still prefers `lake env lean` when a `lakefile` is present, so a
  Mathlib environment is used automatically where one exists.
- The research swarm defaults to **mechanical mode**: no language model is called
  unless `--author` is passed, so a default run makes zero network requests and
  costs nothing. All verification is local and deterministic.

### Added (earlier)
- **Classifier-driven tool-step policy** (`--step-policy classifier|fixed|unbounded`,
  `--max-steps N`, TUI `/steps`). The default `classifier` policy runs without a
  fixed step cap: the runtime overseer injects a visible stop-circling system
  prompt when the agent repeats unnecessary tool calls and stops the run with a
  termination verdict when interventions fail. `fixed` restores the previous
  hardcoded budgets (4–20 steps by thinking level, cap 32); `unbounded` removes
  the cap and all interventions. Explicit `max_steps` always wins, and swarm
  subagents inherit the session policy.
- **Prompt-injection and system-prompt visibility.** Every classifier or harness
  prompt injection is emitted as a `prompt_injection` event and rendered verbatim
  in the CLI and TUI; the fully assembled ingested system prompt (workspace, mode
  guidance, preferences, skills, memory exemplars) is shown before each task via
  the `system_prompt` event.
- **Central editable prompt registry** (`src/adaptive_harness/prompts.py`). All
  model-facing text lives in one place as named templates: system prompts and
  assembly fragments, domain guidance, overseer interventions, the stop-circling
  and termination prompts, harness nudges, swarm role prompts and instructions,
  and the LLM classifier prompts. Override any prompt via
  `.harness/prompts.json` (project), `~/.config/adaptive-harness/prompts.json`
  (user), or `ADAPTIVE_PROMPTS_FILE`. Inspect with `adaptive-harness prompts
  list|show|export|path` or the TUI `/prompts` command.
- **Swarm/subagent remediation** (`docs/swarm-subagent-analysis.md` documents the
  full design comparison):
  - Subagent observability: `DeveloperAgentWorker` forwards all run events to the
    TUI (`⇢ subagent` prefix), summaries are never empty, and failures always
    report their error and stop reason.
  - Structural refusal detection (explicit decline phrasing with zero completed
    tool work) replaces the phrasing regex that failed benign read-only roles.
  - Mutation evidence from observable file state: files created through shell
    commands count as edits regardless of tool names.
  - QA verification requires real evidence (passing tests or any successful
    inspection tool call), so test-less repositories stay verifiable without
    claim-only passes.
  - Bounded QA/security → coder repair loop (`max_repair_cycles`, default 2)
    that routes review findings back to the implementer.
  - Feedback retries: transient role failures retry once with the prior failure
    cause attached; deterministic failures (refusals, boundary violations) do
    not retry.
  - Unified delegation (`run_assignment`) shared by the pipeline and
    `delegate_subagent`; a reachable security reviewer role with its own prompt;
    security and QA reviews run in parallel while writes stay serialized.
  - Subagents inherit the experience repository and the session safety profile.
  - Model-driven delegation policy: auto swarm mode pipelines only explicit
    multi-agent requests; complex edits use `delegate_subagent` and the model
    decides. Long prompts are never hijacked by text heuristics.

### Changed
- `DeveloperAgentWorker` no longer hardcodes an 8-step subagent cap; it follows
  the session step policy (`max_steps` remains available as an explicit override).
- `delegate_subagent` accepts `architect`, `coder`, `reviewer`, and `security`
  roles and returns structured results with verification status.
- The TUI swarm telemetry shows a fourth role state (Security).

### Removed
- The temporary unbounded-only step limit and the hardcoded 32-step ceiling for
  default runs (both remain reachable via `--step-policy fixed`).
