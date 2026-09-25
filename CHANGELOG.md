# Changelog

All notable changes to Adaptive Agent Harness. The five entries below cover the
work in this push; earlier history is summarized in `git log`.

## [Unreleased]

### Added
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
