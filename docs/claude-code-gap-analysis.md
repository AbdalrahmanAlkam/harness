# Gap Analysis: Claude Code vs. Adaptive Agent Harness 2.0

**Date:** 2026-02-14
**Baseline commit:** `b89466b` (v2.0.0, 111 modules / 27,060 LOC under `src/`, 11,855 LOC of tests)
**Method:** every row below was verified against the source tree by search, not from memory. Rows are
marked `[VERIFIED]` (confirmed present/absent by reading the code), `[PARTIAL]` (present but
materially weaker than the Claude Code equivalent), or `[ASSUMED]` (a Claude Code capability I could
not verify from this repo and am describing from public documentation — treat as a requirement to
confirm, not an established fact).

## Executive summary

This harness is **not** behind Claude Code on agentic quality. On verification, honesty, and
classifier-mediated context control it is substantially *ahead*: Claude Code has no equivalent of
`RuntimeOverseer`, `SkillVerifier`'s evidence-only check semantics, the workspace-mutation signature,
or the semantic-drift detector.

It is behind on **ergonomics, extensibility interop, and session trust**. The gaps cluster into
seven groups:

1. **Interop** — no MCP, no LSP, no editor/IDE bridge.
2. **Session trust & recovery** — no checkpoint/undo, no rewind, no per-turn filesystem snapshot.
3. **Context engineering** — no hierarchical/auto-compact at the conversation level, no
   prompt-cache-aware context assembly, no `/context` budget visualiser, no explicit
   context-window *shape* control.
4. **Planning & transparency** — no real plan mode, no todo/task-list as a first-class user-visible
   artifact, no permission *prompts* at call time.
5. **Ergonomics** — no `/rewind`, `/model` fast cycling, `/clear` semantics, `/export` fidelity,
   `/config` depth, no resume-picker, no `/vim`.
6. **Operational depth** — no background tasks/detach, no scheduled tasks, no `/usage` cost
   accounting, no remote/TUI-pairing.
7. **Ecosystem** — no plugin marketplace/registry, no skills-installer, no shareable transcript
   format, no CI non-interactive contract as a first-class tested surface.

The strategic read: Claude Code won on *ergonomics and interop*; this harness won on *rigor*. The
plan in `docs/master-plan.md` is to keep the rigor and close the ergonomics gap without importing a
single one of these features into the core — every one of them ships as a plugin.

---

## 1. Model & provider layer

| # | Claude Code capability | This harness | Status | Notes |
|---|---|---|---|---|
| 1.1 | `/model` with fast-cycle and pinned "default" | `prompts.py` `system.default` + CLI model flags; routing events carry `manual`/`forced`/`auto` | [VERIFIED] | Manual override exists. No *pinned default that survives routing*, and no fast cycling. |
| 1.2 | Multi-provider failover on rate-limit/overload | `llm/fallback.py` path exists at agent level | [VERIFIED] | Present but provider-metadata-blind. No cooldown/backoff circuit breaker. |
| 1.3 | Per-request effort/thinking control | `llm/effort.py`, `ThinkingLevel` 8-way, `FIXED_STEP_BUDGETS` | [VERIFIED] | **Ahead**: step budget is tied to thinking level. |
| 1.4 | Automatic context-window overflow handling | `agent/context_window.py` `prepare_context(threshold=0.75)`, protects system/developer, last 8, operator prefix, unified diffs | [VERIFIED] | **Ahead** of naive truncation. But single-pass, no hierarchy. |
| 1.5 | Prompt caching / cache-aware context ordering | `tests/test_prompt_caching_regressions.py` exists | [VERIFIED] | Tested for regressions; no explicit cache-budget accounting or hit-rate telemetry. |
| 1.6 | Streaming with partial JSON tool-arg recovery | not found | [VERIFIED] | **Missing.** Malformed/streamed-truncated arguments are not repaired. |
| 1.7 | Token counting by real tokenizer, not `len/4` | `estimate_tokens` = `max(1,(len+3)//4)` | [VERIFIED] | **Weakness.** ~10-15% error on CJK/code. Should be a plugin-tunable estimator. |
| 1.8 | Cost tracking per session | `cost` referenced in 10 files | [VERIFIED] | Displayed, not enforced. No `/usage`, no per-model price table, no budget ceiling that halts. |

## 2. Tools & execution

| # | Claude Code capability | This harness | Status | Notes |
|---|---|---|---|---|
| 2.1 | All tools carry `read`/`write` risk, gated by safety profile | `Tool.risk`, `RESERVED_TOOL_NAMES`, `READ_ONLY_TOOLS`, strict-profile gate | [VERIFIED] | **Ahead**: `PluginToolAdapter.risk` feeds the gate. |
| 2.2 | Symlink-resolved workspace sandbox | `tools/base.py` `workspace_path`, confinement in `_confine_paths` | [VERIFIED] | **Ahead** of naive `Path.resolve` prefix checks. |
| 2.3 | Structured shell/pytest arguments, no string interpolation | `RunBashTool`, `RunPytestTool` structured | [VERIFIED] | Good. |
| 2.4 | Background/detached processes | `tools/process.py` | [VERIFIED] | Present but no user-facing "run in background and notify me" contract. |
| 2.5 | `web_fetch` — retrieve and convert a page to markdown | no `webfetch` anywhere in `src/` | [VERIFIED] | **Missing.** `web_search` exists (9 files); fetch/extract does not. |
| 2.6 | LSP: go-to-definition, references, diagnostics | none | [VERIFIED] | **Missing.** High value; cheap as a plugin. |
| 2.7 | Notebook / REPL state persistence | `run_python_repl` exists, sandboxed, stateless per docs | [VERIFIED] | No REPL state persistence across calls. |
| 2.8 | Editor/IDE integration (open file, selection context) | none | [VERIFIED] | **Missing.** |
| 2.9 | Notebook-edit tool (cell-level, not whole-file) | `write_file`/`edit_file` only | [VERIFIED] | **Missing.** |
| 2.10 | Glob/grep as *tools* rather than shell | `search_files`, `list_directory` | [VERIFIED] | Adequate. |
| 2.11 | Slash commands invocable by the *model* mid-run | plugin `commands` contribution | [VERIFIED] | Operator-only. No model-invocable command surface. |

## 3. Context & memory

| # | Claude Code capability | This harness | Status | Notes |
|---|---|---|---|---|
| 3.1 | `CLAUDE.md` project memory auto-loaded | `AGENTS.md` = 0 hits | [VERIFIED] | **Missing.** No project-instruction file, no import/includes, no precedence chain. |
| 3.2 | Auto-memory: notes the agent writes for itself | `data/memory` referenced in 14 files | [VERIFIED] | Session-scoped only. No durable, self-authored, reviewed memory. |
| 3.3 | `ToolOutputArchive` + `read_full_output` retrieval | LRU limit 12, `OUTPUT-{n}` tokens, NOTICE string | [VERIFIED] | **Ahead** — a genuinely good context-relief primitive. |
| 3.4 | Classifier-based noise detection on tool output | `ToolOutputFilter`, `OUTPUT_VALUE_LABELS`, fail-open | [VERIFIED] | **Ahead.** Fails open on classifier error — correct. |
| 3.5 | `/context` — visualise what is consuming the window | none | [VERIFIED] | **Missing.** `prepare_context` returns the data; nothing renders it. |
| 3.6 | Hierarchical compaction (map → reduce over summaries) | single-pass summarize of old turns | [VERIFIED] | **Missing.** No summary-of-summaries, no pinned-anchor support. |
| 3.7 | Skill *definitions* kept out of context by a classifier | `classifiers/skill_classifier.py`, `skills/router.py` (TF-IDF+LR, local, no FM tokens) | [VERIFIED] | **Ahead — this is the core mentality. Preserve exactly.** |
| 3.8 | `@file` / `@dir` explicit context inclusion | none | [VERIFIED] | **Missing.** |
| 3.9 | Microcompact of individual large tool results | `compact_tool_output(limit=1200)` | [VERIFIED] | Present, single-threshold. |

## 4. Permissions & safety

| # | Claude Code capability | This harness | Status | Notes |
|---|---|---|---|---|
| 4.1 | Per-call permission prompt / allowlist editing | permission *profiles* (`turbo`/`balanced`/`cautious`) set at startup | [VERIFIED] | **Gap:** no interactive "allow this once / always" at call time. |
| 4.2 | Allow/deny rule files with regex patterns | none | [VERIFIED] | **Missing.** Highest-value safety gap. |
| 4.3 | Sandbox-exec / seatbelt on macOS, landlock on Linux | declarative profiles only | [VERIFIED] | **Missing.** Profiles restrict which tools exist, not what a subprocess may touch. |
| 4.4 | Read-before-write / edit-collision detection | `_workspace_signature` (mtime_ns+size per file), `write_file` reread discipline | [VERIFIED] | **Ahead.** Observable-mutation detection independent of tool used. |
| 4.5 | Hooks (PreToolUse/PostToolUse) | plugin `hooks` permission, observes event stream | [VERIFIED] | **Partial:** *observe* only. A hook cannot *deny* a tool call or rewrite arguments. |
| 4.6 | Injection defence on retrieved content | `OPERATOR_PREFIX` protected content; all tool output untrusted | [VERIFIED] | Adequate. No taint tracking. |

## 5. Agent loop & autonomy

| # | Claude Code capability | This harness | Status | Notes |
|---|---|---|---|---|
| 5.1 | Runtime loop/stall/hallucination/drift detection | `RuntimeOverseer`, 5 states, evidence-based, `check_claim()` | [VERIFIED] | **Well ahead.** `HALLUCINATION_DETECTED` on existence-claim-after-failed-read is stronger than anything in Claude Code. |
| 5.2 | Evidence-only skill verification | `SkillVerifier`: "an LLM assertion alone never satisfies a check"; `reachable_invariants` | [VERIFIED] | **Ahead.** Unreachable invariants are never demanded. |
| 5.3 | Plan mode / accept-edits mode | `plan_mode` = 0 hits | [VERIFIED] | **Missing.** Safety profile ≠ plan mode. |
| 5.4 | Todo/task list as first-class shared artifact | only as a bundled plugin (`todo-scan` scans source markers) | [VERIFIED] | **Gap:** there is no live *task-state* tool, only a source-scanner. Highest-value missing UX. |
| 5.5 | Subagent delegation | `SpawnSubagentTool`, `agent/swarm.py`, worktree isolation | [VERIFIED] | **Ahead**: real git-worktree isolation, off by default. |
| 5.6 | /clear, /rewind, /resume pickers | `/steer`, `/interrupt`, resume supported; no rewind | [VERIFIED] | **Gap.** |
| 5.7 | Effort/thinking persisted across steps | `STEP_POLICIES` classifier/fixed/unbounded | [VERIFIED] | **Ahead.** |
| 5.8 | Bounded recovery, visible failure | `harness/fallback.py`, storage_error events | [VERIFIED] | **Ahead** — the README line "Errors are never compressed" is a real design commitment. |

## 6. UX & ergonomics

| # | Claude Code capability | This harness | Status | Notes |
|---|---|---|---|---|
| 6.1 | Rich TUI, command palette, settings screen | `tui/app.py`, Ctrl+P palette, F7 `/settings`, clipboard, row-visibility tests | [VERIFIED] | Strong. A11y/row-visibility is a maintained concern here. |
| 6.2 | `/help` that is actually readable | `2099c46 fix: /help was silently deleting half of every line` | [VERIFIED] | Present and cared about. |
| 6.3 | `/export` transcript fidelity | `export` in CLI | [VERIFIED] | Format unverified; no shareable/transcript-diffable format. |
| 6.4 | `/vim` modal editing | none | [ASSUMED] | Low value; **plugin** if at all. |
| 6.5 | `/usage`, cost ceilings | none | [VERIFIED] | **Missing.** |
| 6.6 | `/doctor`, `/bug`, crash reporting | none | [VERIFIED] | **Missing.** `/doctor` is high value, low cost. |
| 6.7 | `--resume --last`, session picker UI | resume supported | [VERIFIED] | No picker. |
| 6.8 | Non-interactive `adaptive-harness -p` CI contract | `dev "TASK"` | [VERIFIED] | No documented exit-code contract for CI gating. |

## 7. Extensibility & ecosystem

| # | Claude Code capability | This harness | Status | Notes |
|---|---|---|---|---|
| 7.1 | Plugin system, manifest permissions, discovery precedence | `plugins/host.py`: 3-tier precedence, 9 permissions, unknown-key rejection, `min_harness`, reserved names, untrusted-code discipline | [VERIFIED] | **Ahead in rigor**, behind in *ergonomics* (see 7.4–7.6). |
| 7.2 | Bundled example plugin | `plugins/bundled/todo-scan` | [VERIFIED] | Only one. Needs a real template + test harness. |
| 7.3 | Skills | `skills/registry.py`, `router.py`, `verifier.py`, classifier-gated | [VERIFIED] | **Ahead** in the classifier gate. |
| 7.4 | One-command plugin install / marketplace | none | [ASSUMED] | **The single biggest adoption blocker.** |
| 7.5 | Hooks, commands, prompts, settings, skills, tools, subagents, MCP servers, LSP — all as plugin contributions | tools, skills, settings, prompts, commands, hooks (observe-only) | [VERIFIED] | **Gap:** no `subagent` and no `mcp` contribution types; hooks cannot deny. |
| 7.6 | Plugin lifecycle (`init`, dev-mode reload, package/sign/publish) | hand-author a directory + manifest | [VERIFIED] | **Missing.** `harness plugin init` / `dev` / `pack`. |
| 7.7 | Agent Skills (Anthropic `SKILL.md` standard) | bespoke `BaseSkill` in `skills/registry.py` | [ASSUMED] | **Gap:** no SKILL.md loader → portability of skills is zero. |

## 8. Evaluation & research

| # | Claude Code capability | This harness | Status | Notes |
|---|---|---|---|---|
| 8.1 | Benchmark + metrics + plots | `evaluation/benchmark.py`, `metrics.py`, `plots.py`, `terminal_plotting` tests | [VERIFIED] | **Ahead.** |
| 8.2 | Formal verification gates (Lean, Typst, SymPy) | `run_lean_proof`, `compile_typst`, `verify_equation`, receipt hashing, sorry/admit/axiom rejection | [VERIFIED] | **Far ahead.** No peer has this. |
| 8.3 | Multi-agent research pipeline | `research/` 17 modules, roles, ledger, gate, swarm | [VERIFIED] | **Ahead.** |
| 8.4 | Regression suite as a feature | 46 test files incl. `test_integrity_regressions.py`, `test_system_prompt_hygiene.py` | [VERIFIED] | **Ahead.** |
| 8.5 | Public benchmark comparability | none | [VERIFIED] | **Gap:** no reproducible public eval harness. |

---

## What NOT to "fix"

These harness behaviours look like gaps but are deliberate and superior. The plan must protect them:

- **Local-first classifiers.** `skills/router.py`: *"Local routing across craft skills; no
  foundation-model tokens are used."* Every classifier must have a deterministic local backend.
  A feature that requires an FM to decide what to put in context is a regression.
- **Fail-open on classifier error**, never fail-closed, for *advisory* classifiers
  (`tool_filter.judge`, `ToolOutputFilter`). Security classifiers stay conservative — that split
  is intentional.
- **Unknown manifest keys are errors.** Plugin typos must not silently no-op.
- **Unreachable invariants are never demanded** (`reachable_invariants`). Do not "simplify" this.
- **The core never branches on where a tool came from.** `PluginToolAdapter` exists specifically so
  the agent loop contains no `is_plugin` check. Any new capability must preserve this.
- **Errors are never compressed.** The output filter must never touch a failure string.
