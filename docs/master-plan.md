# MASTER PLAN — Adaptive Agent Harness 3.0

> **Objective:** make this the most capable, most trusted, most ergonomic agent harness in
> existence — without growing the core.
>
> **Single governing law:** *anything not everyone needs is a plugin.* The core contains only what
> every single invocation requires. Everything else is loaded, on demand, through a manifest.
>
> **Standing instruction:** the implementing model does not stop until every acceptance criterion in
> this document is met and every test passes. Ambiguity is resolved in favour of the governing law.

---

## 0. Orientation — what already exists (do not rebuild it)

Before writing code, read these and understand them. They are the foundation; the plan is an
extension, not a replacement.

| Area | Module | Status |
|---|---|---|
| Agent loop | `agent/agent.py` (1,456 lines) | fixed/classifier/unbounded step policies, operator steering, cancellation, prompt-injection events, workspace-mutation signatures |
| Context | `agent/context_window.py`, `agent/tool_filter.py`, `agent/compaction.py` | threshold compaction w/ protected regions, `ToolOutputArchive` LRU + `read_full_output`, classifier noise filter |
| Classifiers | `classifiers/` (10 modules) | skill, domain, thinking, risk, complexity, ambiguity, verification, runtime overseer, SEMIF engine, backend abstraction (sklearn/Ollama/OpenAI-compat/ONNX/OpenRouter) |
| Skills | `skills/` (3 modules) | registry, local TF-IDF router, **evidence-only verifier** |
| Plugins | `plugins/` | 3-tier discovery, 9 permissions, 6 contribution types, untrusted-code discipline |
| Verification | `harness/verifier.py`, `tools/lean.py`, `tools/plotting.py`, `tools/python_repl.py` | active verification; Lean/Typst/SymPy exact gates with SHA-256 receipts |
| Research | `research/` (17 modules) | formulation, formalisation, ledger, gate, synthesis, multi-agent swarm |
| Eval | `evaluation/`, `tests/` (46 files) | benchmarks, metrics, plots, integrity + prompt-hygiene regressions |

**The five sacred invariants** (regression-tested; touching one is a failed task):

1. Local-first classification — no foundation-model tokens spent deciding what enters context.
2. Advisory classifiers fail **open**; security classifiers fail **closed**. This split is explicit.
3. The agent loop never branches on provenance. No `if is_plugin`.
4. Unknown manifest keys are hard errors; ungranted permissions are refused, not ignored.
5. Errors are never compressed, filtered, or summarized.

---

## PHASE 0 — Foundations (blocking; nothing else may start until this lands)

Everything in Phases 1+ depends on Phase 0. Do it first, do it well.

### 0.1 First-class plugin contribution types

`plugins/host.py` currently accepts `tools, skills, settings, prompts, commands, hooks`. Add:

- **`mcp`** — declare MCP server definitions; the host supervises the subprocess lifecycle, parses
  the capability list at startup, and exposes each remote tool through the *existing*
  `PluginToolAdapter`. Zero new agent-loop code. Remote tools are `risk="net"` by default and
  cannot be downgraded below `net` by the plugin.
- **`subagent`** — declare a specialist agent (system prompt + tool allowlist + budget). Surfaces as
  a *mode* for the existing `SpawnSubagentTool`, never as a new tool.
- **`context`** — declare *context contributions*: file globs, project-instruction files, or
  snippets that must enter context when their trigger matches. **This is the mechanism that makes
  §1 possible.**
- **`commands` → extend** — allow a command to declare a `model:`/`tier:` override and an
  `argument_schema`, so operators can pin a cheap model for `build` and an expensive one for `review`
  without touching config.

Acceptance: a test plugin in `plugins/bundled/` uses every one of the seven types; the agent loop
diff contains no new branches.

### 0.2 Hooks that can deny

Today `hooks` observe. Elevate to a two-phase hook protocol:

- `PRE_TOOL` — may return `{allow}` or `{deny, reason}` or `{rewrite, arguments}`. A denial is
  surfaced to the model as a tool error, not swallowed.
- `POST_TOOL` — may attach metadata or redact output before it reaches context.
- `ON_FINAL` — see Phase 4; this is the classifier's mount point.

Deny must be enforceable *inside* the safety-profile gate, so a hook cannot be bypassed by a
high-privilege profile. A denial is always reported in the transcript.

### 0.3 `harness plugin` lifecycle CLI

```
harness plugin init <name>        # scaffold plugin.py + manifest.json + README + tests
harness plugin dev <name>         # load from ./.harness/plugins with reload on change
harness plugin pack <name>        # → dist/<name>-<version>.tar.gz, manifest validated
harness plugin validate <path>    # manifest + permissions + import check, no execution
harness plugin list|info|enable|disable
```

`scaffold` output must be a plugin that passes `validate` on first run with zero edits — that is the
test.

### 0.4 Token accounting that is actually right

`len(json)//4` under-counts code and CJK by 10–15%, which silently defeats `threshold=0.75`.
Introduce `TokenEstimator` with pluggable backends: `chars4` (default, zero-dep), `tiktoken`,
`transformers`, `provider` (delegate to the API's usage). Keep the estimator behind the existing
`estimate_tokens` signature so no call site changes. **The estimator itself is selected by a
plugin setting** — the core only knows the interface.

---

## PHASE 1 — The Context Plane (the heart of the design)

> The main model's context must stay as close to empty as possible while it remains sufficient.

### 1.1 Everything context-bearing goes through the Context Plane

One rule, enforced by a test that walks the source: **no module may write into the message list
except through `context_plane.py`.** Prompts, skill text, memory, tool output, file contents, MCP
descriptions, project instructions — all of it arrives as a `ContextFragment`.

```python
@dataclass(frozen=True)
class ContextFragment:
    source: str          # "skill:refactor", "mcp:github", "file:src/app.py"
    content: str
    tokens: int
    priority: int        # eviction order; lower survives longer
    trigger: Trigger     # always | regex | classifier:<label> | command | on_tool:<name>
    pinned: bool = False # never evicted, never compacted
    provenance: str = "" # which plugin contributed it
```

### 1.2 Two-stage gating: cheap-and-local first, classifier second

Stage 1 — the **trigger matcher**: regex/keyword/local-model, microseconds, no context spent.
Stage 2 — the **relevance classifier**: the existing `classifiers/` machinery, local backend, scores
the surviving fragments and keeps only those above a calibrated threshold.

A fragment never enters the message list because *a plugin said it might be useful*. It enters
because a classifier scored it relevant **for this turn**.

### 1.3 Context budget allocator

Replace the single `threshold` with an explicit budget, and make it legible:

- reserve for system+developer, current turn, pinned fragments, and a reserved reserve for the
  *final summary*;
- distribute the remainder across `guaranteed` fragments by priority, then admit `opportunistic`
  fragments by classifier score until full;
- emit a `context_budget` event every turn: admitted / rejected / evicted, with reasons.

`/context` (Phase 2) renders this event stream. It is the feature users will screenshot.

### 1.4 Hierarchical compaction

Single-pass summarization loses detail irreversibly. Implement map→reduce: summarize the oldest
cohort, then summarize summaries with a generation counter, down to a fixed budget. Pinned
fragments are exempt. Preserve the existing protected-region logic exactly.

### 1.5 Durable memory (self-authored, reviewed)

Three tiers, each with a different trust level:

- **session** — already present.
- **project** — `AGENTS.md` / `.harness/instructions/*.md`, auto-loaded with an include chain and
  a precedence rule; a `harness init` scaffolds it. This closes gap 3.1.
- **agent-authored** — the model may *propose* a memory. A proposal never takes effect silently: the
  classifier scores whether it is project-general or task-specific, and the operator is shown a
  one-line diff to accept. Accepted memories are namespaced, revocable, and listed by `/memory`.

---

## PHASE 2 — Interfaces

### 2.1 Plan mode (gap 5.3)

A mode, not a personality. In plan mode the agent may read, search, and delegate, but every mutating
tool is intercepted by the safety gate. On completion the plan is presented with an explicit
diff-preview, and the operator approves, edits, or rejects. Rejection returns the plan as context,
not as a failure.

### 2.2 Live task list (gap 5.4) — plugin, first

`todo-scan` scans source markers; what is missing is *live harness task state*. Ship
`plugins/bundled/tasklist/` exposing `task_update` / `task_list`, rendered as a persistent TUI
panel. Requirements: monotonic state transitions, an item may not be marked done without tool
evidence, and the classifier may challenge a `done` that lacks it. **Ship it as the second bundled
plugin specifically to prove the pattern: a feature users expect in the core living outside it.**

### 2.3 Permission prompts and rule files (gaps 4.1, 4.2)

- Interactive `allow once / always / never` at the moment of a gated call.
- `.harness/rules.json`: ordered allow/deny rules, regex over `tool + normalized arguments`,
  first-match-wins, deny wins ties. Hot-reloadable.
- Enforcement point is the safety gate itself, so profiles cannot bypass rules.

### 2.4 Real sandboxing (gap 4.3)

Profiles restrict *tools*; they do not restrict what a subprocess can reach. Add an optional
`SeatbeltBackend` (macOS) / `LandlockBackend` (Linux) / `container` backend behind the sandbox
interface, defaulting to the current enforcement profile where unavailable. **The core ships only
the interface and the detection logic**; each backend is a plugin.

### 2.5 TUI additions

- `/context` budget waterfall (renders §1.3 events).
- `/doctor` — environment, plugin validity, classifier backends, toolchain availability, latency
  probes. Cheaper to build than it looks, and immediately useful.
- `/usage` — tokens, cost, latency per turn, top tools by tokens.
- `/rewind`, `/clear`, session picker.
- `/export` → deterministic, diffable Markdown + JSON transcript.
- Fix first-run onboarding: the palette must be discoverable without reading the README.

### 2.6 Non-interactive contract

Define and test exit codes (`0` ok, `1` agent failure, `2` gate/verification failure, `3` budget
exhausted, `4` classifier unavailable), `--json` event stream, and `--max-cost`/`--max-turns`
ceilings that halt deterministically. This is what makes the harness CI-usable.

---

## PHASE 3 — Interop & capability

All plugins. The core gains no tool.

| Plugin | Contents |
|---|---|
| `mcp` host support (§0.1) | lifecycle + adapter; nothing core |
| `web` | `web_fetch` (page → markdown), search + fetch composed, SSRF-guarded, per-host rate limits |
| `lsp` | go-to-definition, references, hover, diagnostics; drives a "fix the type error" loop |
| `ide` | open file, read selection, insert at cursor |
| `notebook` | cell-level edit/execute for `.ipynb` |
| `sandbox-*` | the §2.4 backends |
| `agents-md` | the §1.5 project-memory loader |

`web_fetch` is the highest-value single addition in the entire plan: research mode currently has
`web_search` with nothing to read.

---

## PHASE 4 — The Quality Controller (explicitly requested; the crown jewel)

> *The classifier is always fed the model's final summary and checks whether what the model did
> fulfils what the user asked for.*

### 4.1 Requirement extraction

At turn start, before the model acts, extract from the user's request a `RequirementSet`:
numbered, atomic, checkable obligations. "Add retry logic and update the docs" ⇒
`R1 implement retry`, `R2 docs mention retry`. Store it in session state; it is the ground truth
for all later stages. Persist it so a resumed session does not lose it.

### 4.2 Evidence ledger (observe, don't ask)

Every tool result is recorded as a typed `Evidence` record: tool, normalized target, success,
salient output digest, workspace-signature delta. The ledger is built *during* the run — asking the
model to narrate what it did is exactly the hallucination surface we are trying to remove. This
mirrors the `SkillVerifier` principle ("an LLM assertion alone never satisfies a check") at
turn scope.

### 4.3 The final gate

On the model's proposed final summary:

1. **Feed it** the `RequirementSet`, the `EvidenceLedger`, and the workspace signature delta.
2. **Classify per requirement**: `satisfied` (evidence present) / `partially_satisfied` /
   `unsupported_claim` (asserted, no evidence) / `missing`.
3. **Detect contradictions** between summary claims and the ledger — e.g. "all tests pass" with a
   recorded `tests failed` evidence record. This is deterministic and runs before any classifier.
4. **Emit a verdict** — `accept`, `revise` (with an itemised list of what is unsupported), or
   `reject`.

### 4.4 What happens on a non-accept verdict

- `revise` → the specific deficiencies are returned to the model as a *targeted* instruction, not
  a generic "try again". The model continues; it does not start over.
- `reject` → the run is marked failed in the transcript and the CLI exit code reflects it.
- The gate **must never** fail open on a hallucination-class verdict, and must never rewrite the
  model's text itself. It reports; the model authors.
- Reuse `RuntimeOverseer.HALLUCINATION_DETECTED` rather than inventing a second mechanism.

### 4.5 Calibration

`unsupported_claim` must have a measured false-positive rate. Build a labelled set of
(historical summary, ledger) pairs — real ones, harvested from `ExperienceRepository` with the
distill pipeline — and report precision/recall in `evaluation/`. A quality gate that cries wolf
gets switched off, which is worse than having none.

### 4.6 Cost discipline

Local backend by default. The final gate is the single highest-value place to spend an FM token, so
it gets the `reasoning` tier and everything else stays on the local/sklearn tier.

---

## PHASE 5 — Self-improvement (the harness improves itself, verifiably)

### 5.1 Experience → skill distillation

Already partly present (`learning/distill.py`, `harvester.py`). Extend: harvested skills are
*proposed*, must pass `SkillVerifier` evidence checks, and land in a staging area awaiting
acceptance. Self-modification is only ever additive and reviewable.

### 5.2 Classifier self-calibration

Track per-classifier accuracy and confidence against outcomes; auto-adjust decision thresholds
within configured bounds; expose drift as an event. A classifier that silently degrades is a
harness that silently rots.

### 5.3 Prompt registry with A/B evaluation

`prompts.py` is already a registry. Add: variants, an evaluator, and offline comparison so a prompt
change cannot land without a measured delta on the benchmark suite.

---

## PHASE 6 — Ecosystem

- **Registry & marketplace**: signed index, `harness plugin add <name>`, version resolution,
  `min_harness` enforcement (the field already exists — use it).
- **SKILL.md compatibility**: read Anthropic-standard `SKILL.md` with frontmatter so skills are
  portable *into* the harness; keep the native format as a superset.
- **Shareable transcripts**: the §2.5 export format, with the ledger and classifier verdicts
  included so a transcript is itself evidence.
- **Docs generated from the source**: plugin authoring, contribution points, and the classifier
  contract auto-documented, so a new contribution type cannot ship undocumented.

---

## PHASE 7 — Verification, performance, polish

- Property tests for the safety gate, path confinement, and manifest validation.
- Fuzz the context plane: no fragment may escape the budget; no eviction may remove a `pinned`
  fragment; compaction must never touch an error string.
- Latency budget per turn, asserted in tests; a `test_workspace_signature_performance.py`-style
  guard for every new hot path.
- Every existing regression file keeps passing. Tests run **serially** (`-n auto` is unsupported —
  `test_a_worker_that_exceeds_its_budget_is_timed_out_and_retryable` depends on same-file state).
- New: `test_plugin_contribution_types.py`, `test_context_plane_budget.py`,
  `test_hook_denial.py`, `test_final_gate_verdicts.py`, `test_claim_gate_calibration.py`,
  `test_sandbox_escape.py`.

---

## Execution order and definition of done

```
Phase 0 (foundations)  ──► blocking for everything
   ├── Phase 1 (context plane)      ──► the core differentiator
   ├── Phase 4 (quality controller) ──► runs on §1 + §0.2
   ├── Phase 2 (interfaces)         ──► mostly plugins
   ├── Phase 3 (interop)            ──► pure plugins
   ├── Phase 5 (self-improvement)   ──► needs §4.5 calibration first
   ├── Phase 6 (ecosystem)          ──► needs a stable manifest schema
   └── Phase 7 (verification)       ──► continuous, never "done"
```

### Non-negotiable definition of done

A phase is complete only when **all** hold:

1. Every acceptance criterion above is met and demonstrably exercised.
2. **Zero new lines of core logic** attributable to any optional capability. Diff review must show
   the feature living entirely in `plugins/`.
3. `pytest` green, serially. `pytest -m "not slow"` green.
4. The five sacred invariants still hold, each covered by a named regression test.
5. `docs/plugins.md` and the auto-generated docs updated for any new contribution type.
6. `CHANGELOG.md` updated.
7. A failure mode was tested, not just the happy path.

### How to work

- **Use subagents freely** — parallel independent plugins, independent test authoring, independent
  doc updates. This harness already has `SpawnSubagentTool` and worktree isolation; dogfood it.
- **Verify, do not assume.** Run the suite. Inspect failures. Never report success from intent.
- **Never edit a file you have not read.** This codebase's comments encode non-obvious rationale;
  several "obvious simplifications" would silently destroy correctness.
- **When in doubt, it is a plugin.**

### Anti-goals — explicitly forbidden

- ❌ Adding a tool to the core because "everyone might want it". The bar is *everyone needs it*.
- ❌ An `if plugin` / `if is_plugin` branch anywhere in the agent loop.
- ❌ Making any classifier silently swallow a hallucination verdict.
- ❌ Replacing a local classifier with an FM call to "improve quality" without a measured win.
- ❌ Breaking serial test execution for convenience.
- ❌ Shipping a self-modification path that is not additive, staged, and reviewable.
- ❌ Declaring a phase done while its tests are failing.
