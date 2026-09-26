# Developer agent walkthrough

Run `adaptive-harness tui --workspace /path/to/project` for the interactive agent. Enter a task at the prompt. Type `/` for a command palette above the input; type more to filter, use Up/Down to select, Tab or Enter to complete without submitting, and Escape to close. The right panel shows full skill names and probability bars, entropy and margin, classifier latency, domain mode, thinking budget, and model selection. Press F6 to toggle it; F2 opens the theme picker with live hover and keyboard preview, Enter to save, and Escape to cancel. Tool calls and results appear in the chat log; diffs use syntax coloring, inline LaTeX uses readable Unicode math, and display fractions are stacked in monospaced blocks. Code fences stay literal. `/output` opens a selectable full-text viewer for the latest answer with a Copy All action. Scrolling upward pins the log while new output arrives. When a risky operation needs clarification, select an option with 1–9 or type an instruction. Escape cancels the task. Searchable popups consume Enter so model searches cannot accidentally be submitted as tasks. The question and choices scroll on small terminals while the answer field stays visible.

The TUI uses the directory where it was launched and edits there directly by default. `/isolation on|off|auto` can enable Git worktree review when desired. `/swarm on|off|auto` defaults to auto: explicit multi-agent requests and complex editing tasks run an Architect, a Coder, and a Reviewer. The Architect's plan passes to the Coder, which has write, edit, shell, and test tools; a planning warning does not leave Coder queued. The Reviewer inspects created files and checks them. For science charts, install `pip install 'adaptive-harness[plotting]'`; `plot_terminal` draws line, scatter, and bar charts, and the sandboxed Python REPL exposes optional plotext as `plt`.

For a headless run, use `adaptive-harness dev "inspect the tests"`. The default OpenRouter model is `stealth/space-bunny-alpha` in manual mode. Add `--model anthropic/claude-sonnet-4` or `--tier fast` to choose another fixed model; `--model auto` opts into complexity routing. In the TUI, F4 or `/model` opens a searchable picker for the active provider, and `/model auto` enables automatic routing. OpenRouter choices come from its catalog when available; direct providers show their configured tier models. Repeat `--backup-provider NAME` to opt into failover on 429 or server errors.

For a guaranteed no-network walkthrough, run `adaptive-harness dev --offline "check status"`. This uses the mock engine even if a live key is configured. The mock reports implementation requests it cannot satisfy as errors and `dev` exits with status 1 for incomplete tasks. For shell or test tools, timeouts terminate the full child process group, and captured stdout or stderr is capped at 200 KB.

Copy an agent reply with `/copy` or Ctrl+Shift+C. Drag-select a passage in the chat log first to copy only that selection. If the provider rejects a thinking setting, the harness retries once with the same model and its default reasoning behavior.

To use a local OpenAI-compatible server for the main task model, pass `--base-url http://localhost:8080/v1 --model YOUR_LOCAL_MODEL`. This is independent of `--classifier-backend`, which controls the smaller decision engine.

For local semantic routing, install `pip install '.[semif]'` and download the default Qwen3.5 4B checkpoint into the Hugging Face cache with `hf download Qwen/Qwen3.5-4B`. Then use `adaptive-harness tui --classifier-engine semif --semif-model Qwen/Qwen3.5-4B --semif-device auto`. SemIf evaluates its typed candidate branches directly from one next-token logit pass and does not generate classification text. The automatic engine choice selects SemIf only when its optional packages and checkpoint are already cached; otherwise it uses sklearn. Explicit SemIf selection also falls back to sklearn with a visible notice if weights cannot load. `--semif-4bit` enables CUDA quantization when bitsandbytes is installed, and `--semif-temperature` adjusts probability sharpness. In the TUI, `/classifier semif [model_path]` selects it and `/classifier sklearn` switches back.

To route skill classification through Ollama locally, install Ollama, run `ollama pull qwen2.5:1.5b`, and launch `adaptive-harness tui --classifier-backend ollama --classifier-model qwen2.5:1.5b`. The default endpoint is `http://localhost:11434/api/generate`; use `--classifier-endpoint` for another server. `/classifier sklearn` switches back instantly. Other supported engines are `openrouter` (requires `OPENROUTER_API_KEY`) and `onnx` (requires a local ONNX model and optional Python packages `onnxruntime` and `transformers`).

Domain modes adapt the system instruction and available tools to coding, research, scientific analysis, or security audit. Use `--mode security` or `/mode security` to force the audit mode, and `/mode auto` to resume classification. Coding validates edited Python syntax; research requires source URLs from web search; science provides `calculate` and `check_convergence` for finite arithmetic and observed tail stability; security limits shell access to read-only Git inspection. The science check does not establish a mathematical proof. The telemetry badge displays the active mode, including whether it was forced.

Use `/thinking` to choose among the active model's known reasoning efforts. Space Bunny Alpha exposes `low`, `medium`, `high`, `xhigh`, and `max`; `/thinking deep` remains an alias for `high`, and `/thinking auto` returns to classifier selection. `--thinking` also works at startup and in headless runs. The status changes from thinking to visible-response generation when the provider returns; the provider call itself is non-streaming. OpenRouter receives supported effort settings; Claude receives a reasoning token cap with a larger completion limit. Responses cut off by the model's output limit continue automatically within the step budget. Use `/usage` to review session input, output, reasoning, cache, total tokens, and reported cost.

Tool steps default to classifier control: there is no fixed step cap, and the runtime overseer injects a visible stop-circling system prompt when the agent starts making unnecessary repeated tool calls, then stops the run when interventions fail to restore progress. `/steps fixed` restores the old per-thinking-level budgets (4–20 steps, capped at 32), `/steps unbounded` disables both the cap and interventions, and `/steps N` sets an explicit cap; `--step-policy` and `--max-steps` do the same at launch, and swarm subagents inherit the session policy. Injected classifier prompts and the ingested system prompt appear verbatim in the chat log so you can audit exactly what the model was told.

All model prompts are editable in one place. `adaptive-harness prompts list` (or `/prompts` in the TUI) shows every system prompt, intervention, and injection by name; `prompts show NAME` prints one and `prompts export` writes a JSON copy for editing. Override any of them by mapping names to new text in `.harness/prompts.json` in the project or `~/.config/adaptive-harness/prompts.json` globally.

Use `/workspace /path/to/project` to switch projects between tasks. F5 or `/sessions` opens a searchable saved-session picker; `/sessions list` prints IDs, `/session new [title]` starts one, and `/session load ID` resumes one. A session saves automatically after a task, including classifier backend, model, endpoint, and interaction profile. `adaptive-harness tui --session ID` resumes it on launch while using the launch directory as its workspace. The 22 built-in craft skills are selected automatically: SemIf probes their descriptions locally in one decision; the fallback router scores task wording without an API call. `/skill list` or `/skills` displays them. `/skill NAME` forces a skill and `/skill off` returns to automatic routing; headless and TUI startup also support `--skill NAME`. Selected skill instructions are sent to the task model, while core coding tools remain available. The sidebar shows the active skill and bound tools, and skill verification reports missing checks after execution.

Add a custom skill with `./skills/NAME/SKILL.md`, `.harness/skills/NAME/SKILL.md`, or `~/.config/adaptive-harness/skills/NAME/SKILL.md`. Optionally add `skill.json` beside it with a title, category, trigger description, tool names, and invariant names. A plain `SKILL.md` uses inspection-only tools until you declare a toolset. Custom packages are discovered on each task and can be chosen manually or by automatic routing.

The default `turbo` interaction profile runs ordinary edits, shell commands, and tests without questions; catastrophic commands such as disk formatting or `sudo` still require approval. `/safety balanced` adds high-risk command approval, and `/safety strict` asks before every edit or command. Legacy `cautious` remains available for semantic uncertainty prompts. The status line prominently shows the selected safety mode. The pulsing status line shows classification, model request, tool execution, and verification activity. “Thinking / generating” is an in-flight request indicator; the task API currently returns full responses rather than token-by-token output.

If you ask to create an app, the agent must write files with tools. A fenced code block with an explicit `path:` header is recovered as a `write_file` call, and missing parent folders are created. Plain code in chat cannot complete an edit task. Explicit requests for multiple agents route to the swarm pipeline (architect, coder, parallel security and QA reviewers, with bounded QA→coder repair cycles and feedback retries); headless `dev --swarm` also exposes `delegate_subagent` to the coordinator model. In auto mode only explicit multi-agent requests use the pipeline — otherwise the main agent keeps `delegate_subagent` and decides when to spawn architect, coder, reviewer, or security roles. Subagent tool calls, injected prompts, and stop reasons stream into the main log with a `⇢ subagent` prefix, and failed roles always report their error and stop reason. Run `adaptive-harness distill` after verified tasks to export a local JSONL training set and LoRA scaffold from the experience database.

For focused code inspection, ask the agent to call `read_file` with `symbol="Class.method"`; it returns that Python definition and module imports. Large default reads are limited to 120 lines. Full conversation and tool results remain available in the saved session. A request copy is compacted only when its estimated context use exceeds 75% of the selected model's window; older tool output is reduced first, then older complete turns can be summarized. System instructions, runtime overseer corrections, recent turns, and active diffs remain intact. A successful read-only file answer can be reused at zero API tokens while its source hash remains unchanged. The telemetry panel reports estimated context tokens saved, actual provider-reported cached tokens, and memory hits. These counters are observations, not a promise of a fixed savings percentage.

In science mode, `run_python_repl` executes NumPy, SciPy, and SymPy code inside a network-disabled Bubblewrap sandbox; install Bubblewrap on Linux to enable it. `verify_equation` checks exact symbolic substitutions and catches solutions that make denominators zero. The Python tool resets state on every call and cannot read the project workspace. Clarification choices for stable, non-destructive preferences are stored at `~/.config/adaptive-harness/preferences.json` with private permissions; approvals and credentials are excluded.

Use `/new` or Ctrl+N to start a fresh session, or `/reset` to clear the current session's conversation, telemetry, and tool output. Up and Down recall task prompts, including across launches; `/history` lists recent prompts. Slash commands, including `/key`, are excluded from this history. `/key <provider> <key>` saves each provider's credential in `~/.config/adaptive-harness/credentials.json` with private `0600` permissions; `/key status` shows masked keys and `/key clear <provider>` removes one. `/provider <name>` switches among OpenRouter, Anthropic, OpenAI, DeepSeek, Google, Groq, and local OpenAI-compatible endpoints. On launch, `--key` takes precedence over the active provider's environment key, which takes precedence over saved credentials. Existing OpenRouter keys in `config.json` are still read. `/theme` saves a preferred terminal palette in `config.json`. `/export markdown` or `/export json` writes the complete session conversation and tool traces to `output/sessions/` under the current workspace. The header displays live provider, context gauge, and cumulative token usage.

The runtime overseer examines each tool result for loops, unsupported claims, stalled failures, and obvious scope drift. It injects a concrete corrective instruction only when an abnormal state is detected; two unresolved loop or stall interventions lead to an interactive choice. The gate is deterministic and local. Pass `--overseer-model PATH` to use an optional local ONNX embedding classifier as a second tier for uncertain trajectories. Model catalog context lengths are used when available; otherwise the gauge uses conservative provider defaults. Direct Anthropic and Gemini connections currently use the providers' OpenAI-compatible chat layers, which support the harness's basic text and tool workflow but not every native API feature.

The small-terminal layout hides chat and status panels below 12 rows so the prompt remains above the Footer. Expand the terminal to restore the full view. A synthetic pytest log was compacted from 78,071 to 1,460 characters in a local check, and a 500-function Python source was reduced to a 78-character AST slice; these are measured examples, while actual model token savings vary. This checkout has no cached SemIf checkpoint, so CPU/GPU neural inference latency must be measured on the target machine after weights are installed. The full test suite passed with warnings treated as errors during the Phase 5 audit.

The tool suite has local file search, bash, pytest, and restricted arithmetic. Set `BRAVE_SEARCH_API_KEY` to enable web search with source URLs in research mode. The domain mode changes prompts and available tools, while verification reports checks that actually ran. The arithmetic tool checks expressions but does not replace symbolic proof or numerical convergence testing. See [the implementation brief](docs/implementation-prompt.md) for the broader product requirements and acceptance criteria.

## Autonomous research swarm

`adaptive-harness research "<topic>"` runs a self-scaling investigation that will not report a result until every claim is backed by an executable receipt. An Executive Director owns the objective and the convergence loop; Literature, Theory, Empirical, and Adversarial Leads each own a methodologically distinct line of attack and spawn as many workers as the outstanding gaps justify.

The default is **mechanical mode**, which calls no language model: the Director, the divisions, the gates, and the paper are all local and deterministic, so a default run costs nothing and makes no network requests. Pass `--author` to have a live model draft the artifacts on top; the mechanical gates still adjudicate them either way. This is why a default run shows no OpenRouter traffic.

```bash
# Zero-cost, zero-network run
adaptive-harness research "optimal routing under heavy-tailed network delay"

# Pinned seed, explicit objective, live authoring
adaptive-harness research "variance-minimising routing policy" \
    --objective "Derive the policy and the regime where it is well-posed" \
    --seed 20260926 --author

# Stop the loop after N cycles; the verdict stays honest either way
adaptive-harness research "some topic" --max-cycles 3
```

A topic is `SOLVED` only when all four invariants hold at once. `mathematical_soundness` requires every script in `proofs/` to be free of floating-point literals and approximating calls (`float`, `evalf`, `N`) and to exit 0 — the scan happens before execution, so an approximate "proof" is rejected without running. `empirical_replication` requires every script in `experiments/` to reproduce its prediction at a pinned seed and emit a hashed data artifact. `adversarial_clearance` requires no open red-team counterexample. `document_integrity` requires `paper.typ` to compile to `paper.pdf` with zero Typst warnings. Typst is resolved from `PATH`, then the `typst` Python wrapper, then reported unavailable; the `compile_typst` tool does the same from inside a task.

The loop is not turn-limited. It is stagnation-limited: it keeps working until it converges or until it can prove that more identical work cannot help. Progress means reaching a new *minimum* in outstanding gaps, tracked as a high-water mark, so a flapping invariant that oscillates cannot masquerade as progress. After `--patience` no-progress cycles the worker budget doubles; when escalation can no longer grow, the run reports `STAGNATION_ABORT` and an honest **UNSOLVED** verdict. `--max-cycles` and the absolute ceiling likewise report `BUDGET_EXHAUSTED` and UNSOLVED — never a false success.

Artifacts land in `research/<topic-slug>/`: the hash-chained `comm_ledger.jsonl` message tree, `00_objective_spec.md`, evidence records, `proofs/`, `experiments/` with their raw CSV, vector `figures/`, `03_adversarial_audit.md`, `bibliography.bib` generated from evidence records only, `convergence_history.json`, the receipt indexes, and the generated `paper.typ` plus its `paper.pdf`. Because the paper is generated from receipts, nothing can be typeset that was not proven or measured, and a withdrawn receipt disappears on the next build. Opening a tampered ledger reports through `verify()` rather than raising, so an auditor can always inspect the evidence of tampering.

The worked example in `research/routing-policy-under-heavy-tailed-delay/` derives a variance-minimising routing split for heavy-tailed delays and, more usefully, establishes the regime where that objective is even well-posed: the variance of a Pareto delay is infinite for tail index `alpha <= 2`, so a variance-minimising policy does not exist on part of the empirically observed range and must be replaced by a robust objective. Three exact SymPy derivations and two seeded experiments back that, and the run converges in two cycles.

## What a research run actually does

`adaptive-harness research "<topic>"` does not audit research you already did; it does the research. The kernel reads the topic, constructs machine-checkable propositions from definitions, writes a self-adjudicating SymPy script for each, executes it, and reads the verdict off the exit code: `0` is PROVEN, `3` is DISPROVEN, anything else is INCONCLUSIVE. It then generates the seeded simulations that corroborate those propositions, runs a red team over the result, and publishes a paper.

Try it on a topic you already know the answer to:

```bash
adaptive-harness research "pareto heavy tailed network delay: critical index and variance-optimal balanced routing"
```

Seven propositions are derived, all seven hold, and the run reports `PROVEN`. The interesting one is Theorem 4: for a Pareto delay of index at most two the second moment diverges, so the variance is infinite under *every* allocation and no variance-minimising routing policy exists at all. The paper says that, with the logarithmic divergence at the critical index shown exactly.

To watch a claim fall, hand it a false identity:

```bash
adaptive-harness research "quadratic expansion" --claim "(x+y)**2 == x**2 + y**2" --symbols x,y
```

This reports `DISPROVEN` with the witness `counterexample at x=1, y=1: lhs - rhs = 2`, and that is a *successful* run: settling the question either way is the goal, so the red team certifies the refutation rather than the claim. The witness matters — refutation requires an exhibited counterexample, not a failure to simplify. The probe set deliberately includes negative points, which is why `sqrt(x**2) == x` is caught at `x = -1` rather than surviving on positive probes.

Every derived script is scanned for floating-point literals and approximating calls before it runs, so an "exact-looking" proof that secretly calls `N()` is rejected without executing. That check earned its place immediately: it caught a real error in the first draft of the Pareto second moment, where the asserted closed form was missing a factor that SymPy got right.

A topic the kernel cannot formalise is not a failure to hide. `adaptive-harness research "zzz qqq unmatchable"` produces a paper that explicitly reports INCONCLUSIVE, states that no derivation strategy matched, and names what input would let it proceed. Prefer that to a confident paper about nothing.

Artifacts land in `research/<topic-slug>/` — the hash-chained ledger, the derivation plan, the generated proof and experiment scripts with their raw data, the figures, the audit log, the bibliography, the convergence history, the receipts, and the paper. Because the paper is generated from receipts, nothing can be typeset that was not derived and checked.

## The two proof tiers, and why `sorry` is fatal

SymPy can tell you two expressions are equal. It cannot tell you a *deduction* is valid. That is what Lean 4 is for, and the harness now runs both as a single standard.

Try the acceptance example:

```bash
adaptive-harness research "Formally prove that the sum of the first N natural numbers equals N*(N+1)/2 and verify both in SymPy and Lean 4"
```

The kernel derives two SymPy propositions (the closed form is algebraically well formed; the induction step is polynomial and therefore exactly decidable) and pairs them with a Lean file that discharges the induction itself. The run reports `PROVEN`, and `paper.pdf` carries a *Formal Foundations* section with a green certification box per proof.

The part worth knowing is what Lean does *not* tell you. Lean will happily accept this:

```lean
theorem gauss_two_mul (n : Nat) : 2 * sumFirst n = n * (n + 1) := by
  sorry
```

and **exit 0**, emitting a `warning: declaration uses 'sorry'` rather than an error. Any verifier that reads the exit status would certify a false theorem. So three checks must all pass: no `sorry` token in the source (with comments and string literals stripped first, so prose cannot trip it), no "declaration uses 'sorry'" diagnostic, and — the strongest — no dependency on `sorryAx`, which the kernel asks Lean about directly via `#print axioms`. You can watch the difference yourself: replace the induction step in `proofs/lean/LEAN-GAUSS.lean` with `sorry`, run `lean` on it, note the exit code is still 0, then re-run the research command. `formal_verification` fails, the run reports `UNSOLVED`, and the paper is not published. That test is pinned as `test_tampered_lean_file_blocks_the_run`.

What must be certified depends on the verdict, because the standard applies to asserted theorems. A `PROVEN` theorem needs its Lean file. A `DISPROVEN` result does not: the refutation is the result, and it is certified by an exact witness the kernel already checked — a Lean proof of a falsehood would be neither expected nor meaningful. An `INCONCLUSIVE` verdict publishes no theorem, so the tier does not apply and the paper says so rather than implying formal backing it lacks.

The shipped proofs use core Lean only, with no imports, because Mathlib cannot be assumed present — its cache is roughly 5 GB and the install failed here for want of space. Core Lean has no `ring`, so the Gauss proof expands `(k+1)(k+2)` by hand. Every library proof is compiled by a test, so if a future Lean release breaks a tactic the suite fails rather than shipping a theorem that no longer verifies. Where a `lakefile` exists the tool uses `lake env lean`, so a Mathlib environment is picked up automatically.
