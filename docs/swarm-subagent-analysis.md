# Swarm/subagent design vs. modern harness practice

Analysis of `SwarmCoordinator` + `DeveloperAgentWorker` + `DelegateSubagentTool`
against how contemporary agent harnesses (Claude Code subagents, OpenAI Agents
SDK, LangGraph/AutoGen, aider architect mode, Devin/Cursor-style composers)
handle subagents.

## Executive summary

The swarm layer implements a **fixed, one-shot pipeline** (architect → coder →
qa) with **lossy JSON hand-offs**, while modern harnesses use **model-driven
decomposition with shared, observable traces and repair loops**. Five gap
classes explain the observed failures: orchestration rigidity, context loss,
zero observability, brittle success heuristics, and missing budget/feedback
mechanisms.

## 1. Orchestration: fixed DAG vs. model-driven decomposition

| This harness | Modern practice |
|---|---|
| `SwarmCoordinator.run` hardcodes three single-assignment waves (`swarm.py:173-187`) | The lead model decomposes the task at run time and spawns exactly the subagents needed |
| Two disconnected paths: auto-pipeline (TUI `_should_swarm`) and `delegate_subagent` (one role per call, synchronous) | One delegation primitive (Task tool / handoff) used by the orchestrator model, chainable and parallel |
| `ThreadPoolExecutor(max_workers=2)` exists but every wave is a 1-element tuple — nothing ever runs in parallel | Read-only work (plans, reviews, research) fans out concurrently; only writes serialize |
| `SwarmRole.SECURITY` is defined and the worker supports it, but nothing ever schedules it | Roles are data-driven and actually reachable |
| No repair cycle: QA findings are terminal; the coder cannot be re-invoked with review feedback | implement → review → fix loops until acceptance criteria or budget |
| `_should_swarm` regex heuristics silently hijack long prompts into the swarm path | Delegation is explicit or model-decided; never a text heuristic on the user's prompt |

Consequence: the pipeline runs even when the task needs one agent, and cannot
run the shape a task actually needs (research-only, multi-file parallel edits,
review-fix loops).

## 2. Context and artifacts: lossy hand-offs vs. shared traces

| This harness | Modern practice |
|---|---|
| Prior results passed as truncated JSON (`summary[:4000]`, tool evidence names only) appended to the prompt | Subagent transcripts are retrievable; the parent can inspect full tool calls and diffs |
| Subagents are cold-started: `DeveloperAgentWorker` does not pass `repository=`, so no experience/exemplars, no memory | Subagents inherit curated memory, exemplars, and project conventions |
| `context_header` + JSON blob can bloat the prompt with no compaction policy of its own | Hand-off contracts are typed and bounded (plans, diffs, test reports) |
| `_swarm_completed` pastes the concatenated summaries into the parent's message history as one assistant turn | Results enter the parent trace as structured artifacts, not a synthetic assistant message |
| Role identity is incoherent: the coder gets `DEFAULT_SYSTEM_PROMPT` ("You are Adaptive Agent… ask the user…") plus a role suffix | Subagent system prompts are role-pure and scoped |

## 3. Observability: "I can't see anything"

| This harness | Modern practice |
|---|---|
| `DeveloperAgentWorker` consumes `run_stream` and **discards every event except `response` and tool name/success/error** (`swarm.py:235-241`) | Subagent events stream into the main timeline as nested spans |
| User sees only status flips (queued/running/done/failed) and a final line | Full nested tool traces, token/cost attribution per subagent |
| On failure the summary is frequently empty (failure paths clear `final_answer`), so the UI prints `Architect (plan):` with no diagnostic; the captured `stop_reason` is never rendered | Failures carry machine-readable reasons surfaced in the UI |
| No cost/usage aggregation for subagent runs | Per-subagent budgets and spend are first-class |

This class is the direct cause of "it keeps failing and I can't really see
anything".

## 4. Success semantics: brittle heuristics vs. acceptance criteria

| This harness | Modern practice |
|---|---|
| `refused = re.search(r"\b(i cannot\|i can't\|unable to\|…)", summary)` marks benign role language as "Model refused the assigned role" and flips success to False | Refusal detection is structural (tool-denial + explicit decline), not phrasing |
| `verified = successful_tests or inspected_files` — a repo without tests can never satisfy QA | Acceptance criteria come from the task (files exist, tests run, checks pass) and are explicit |
| `successful_mutations` counts only `write_file`/`edit_file`, though the system prompt explicitly allows creating files via `run_bash` | Mutation evidence is defined by observable file state (e.g., diff/hash before-after), not by which tool wrote |
| Retry once, blindly, with the same prompt (`swarm.py:157-167`) | Retries carry the failure evidence as feedback ("your attempt failed with X") |

## 5. Budgets and interaction model

| This harness | Modern practice |
|---|---|
| Was hard-capped at 8 steps/subagent (now inherits session step policy — fixed in `ad122fe`) | Per-subagent budgets (steps/tokens/cost) with graceful truncation reports |
| `delegate_subagent` blocks the parent's tool call for the entire subagent run | Long subagents run in the background with progress; the parent can interleave |
| No human-in-the-loop gate per subagent; whole-run safety profile only | Approval gates per risky subagent action |
| Swarm results bypass the experience repository entirely | Verified subagent work is distilled into shared memory |

## What modern harnesses share (design checklist)

1. **Model-driven decomposition** — the orchestrator decides roles, count, and
   sequencing; tooling stays generic.
2. **One delegation primitive** — chainable, parallel-capable, with typed
   structured output contracts.
3. **Shared, nested traces** — every subagent tool call visible and attributable
   in the main timeline.
4. **Repair loops** — review findings route back to the implementer until
   acceptance or budget.
5. **Evidence-based success** — file-state diffs and explicit checks, not
   phrasing heuristics.
6. **Memory inheritance** — subagents receive curated context; verified results
   are distilled back.
7. **Per-subagent budgets and scoped permissions** — small tool sets (present
   here already) plus step/token/cost caps and sandboxing.
8. **Graceful degradation** — if orchestration fails, fall back to a single
   agent run instead of returning nothing.

## Prioritized recommendations

1. **P0 – Observability**: forward `run_stream` events from `DeveloperAgentWorker`
   through a callback (e.g. `on_event`) into `_render_event`, and always render
   `stop_reason`/`error` in failure summaries (fixes "can't see anything").
2. **P0 – Success semantics**: replace the refusal regex with structural refusal
   detection; count shell-created files as mutations via before/after file-state
   hashes; make QA `verified` fall back to task-defined checks when no tests exist.
3. **P1 – Repair loop**: when QA fails, re-run the coder with the QA artifacts as
   feedback (bounded, e.g. 2 cycles) instead of terminating.
4. **P1 – Retry with feedback**: include the failure error in the retry prompt;
   don't retry deterministic failures.
5. **P2 – Unify delegation**: make `delegate_subagent` the single primitive and
   implement the auto-swarm pipeline on top of it; schedule or delete the
   SECURITY role; run read-only waves in parallel (the executor is already there).
6. **P2 – Memory**: pass `repository`/exemplars to subagents and distill verified
   subagent traces back into the experience store.
7. **P3 – Delegation policy**: replace the `_should_swarm` regex heuristic with
   an explicit flag or a classifier decision.
