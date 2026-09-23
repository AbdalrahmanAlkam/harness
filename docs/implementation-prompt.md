# Adaptive Agent Harness: implementation brief

## Concept and purpose

Adaptive Agent Harness is a developer and research agent that combines a capable task model with fast classifiers and a verification loop. The task model writes code, reasons, searches, and uses tools. The classifier layer decides which skill, domain, model tier, and thinking budget fit the request; it measures uncertainty and flags high-impact actions. The verifier checks tool outcomes and sends concrete failures back for recovery. The system exists to improve answer quality and reduce latency and cost without interrupting the user for routine choices. A classifier prediction is guidance, not proof of success. The harness must verify results independently.

The product should help with software engineering, literature research, scientific analysis, mathematics, and security review. It must remain useful without cloud classifier calls: the default decision engine is local statistical ML, while users may choose an Ollama SLM, a local OpenAI-compatible server, local ONNX embeddings, or a cloud classifier. A local classifier model is distinct from the main task LLM. Show both choices clearly.

## Operating principles

1. Finish the user's task with minimal questions. Ask only for a genuinely missing task target or a destructive action that needs authorization. If the user asks “which approach is better?”, evaluate and choose. Entropy and confidence margin are telemetry and recovery signals, not automatic reasons to interrupt.
2. Respect user-selected models and workspaces. Precedence is explicit model or tier, then automatic complexity routing. `/model auto` returns to automatic routing. Never overwrite a manual selection inside the agent loop.
3. Keep the TUI usable at 60×20 as well as wide terminals. Every question, option, error, command, and model identifier must remain readable and keyboard accessible. Long text must wrap or scroll; it must never silently truncate.
4. Treat model output and tool output as untrusted text. Escape Rich markup or render it with safe `Text`, `Markdown`, and `Syntax` objects. Keep shell and pytest arguments structured. Keep file tools inside the selected workspace.
5. Persist conversations deliberately. A session stores its workspace, messages, active model choice, and selected skills. The user can list, resume, create, and save sessions from the TUI and resume one at launch.
6. Make failure visible. An API error, failed verification, canceled operation, or exhausted step budget must never be reported as a successful task. Keep enough context for retry or a truthful final report.

## Execution order

### Phase 1 — repository and behavior audit

Read the agent loop, LLM client, classifiers, tools, CLI, TUI, storage, tests, README, and walkthrough. Record the current call graph and identify tool and model state owners. Reproduce reported bugs with tests or a headless Textual pilot. Check tracked generated files and preserve unrelated user changes. Run a baseline test suite before edits.

### Phase 2 — model and classifier routing

Implement explicit model precedence in one place. Emit a routing event containing selected model, selected tier or `manual`, and `forced` or `auto`. Test both CLI and TUI changes across multiple tool steps. A custom model ID must be passed unchanged to every completion.

Provide a `BaseClassifierBackend` and implementations for sklearn, Ollama `/api/generate`, local OpenAI-compatible `/chat/completions`, ONNX embeddings, and OpenRouter. The default sklearn engine should be a local TF-IDF and logistic-regression classifier with normalized probabilities. External engines must return structured labels and confidence, validate their JSON, and have short timeouts. Record per-head and total observed latency. If an engine fails, emit a visible classifier error and fall back to local classification for that decision. Do not silently pretend that a remote SLM succeeded. Avoid cloud API tokens unless the user selects a cloud classifier.

Use the selected engine for skill, domain, and thinking decisions. Keep explicit safety rules and conservative overrides for high-risk operations; do not ask a language model to decide whether a disk-format command is safe. Expose `/classifier <backend> [model]`, CLI backend/model/endpoint options, and the active backend/model/latency in telemetry. Accept arbitrary local model IDs, including small Qwen or SmolLM models, rather than hardcoding a single SLM. Document the exact local endpoint payloads and optional ONNX dependencies.

### Phase 3 — domain modes and result quality

Provide coding, research, science/math, and audit modes. Each mode has a concise system instruction, an appropriate tool set, and verification expectations. Coding should inspect code, check syntax, run targeted tests, and review diffs. Research should gather source URLs and dates, compare claims, maintain notes, distinguish evidence from inference, and cite sources when a search tool is configured. Science/math should state assumptions and units, test boundary cases, and check numerical convergence when applicable. Audit should verify findings, include paths or evidence, and keep the default workflow read oriented.

Do not claim generic verification succeeded merely because a prompt asked for it. Produce explicit verification events for checks actually run. Add focused tools where they materially improve a domain. Keep the tool suite composable so future web search, document indexing, numerical solvers, and security scanners can be added without changing the agent loop. Keep failure recovery bounded by a step limit and report unresolved failures.

### Phase 4 — thinking budget and model requests

Classify none, low, medium, and deep thinking levels. Show the estimated budget in telemetry. Map levels to fast, standard, or reasoning tiers only when model selection is automatic. For supported OpenRouter reasoning models, pass an effort hint; do not claim that the estimated number of reasoning tokens is enforced. Keep unsupported request parameters away from models that reject them. Test routing, effort parameters, and manual override precedence with a fake client.

### Phase 5 — TUI and clarification experience

Create a clear layout with a readable chat area, status line, prompt input, and optional telemetry panel. Hide telemetry automatically on narrow terminals and let F2 toggle it. Distinguish developer text, agent text, tool calls, tool results, verification, and errors with strong contrast. Render Markdown code blocks and unified diffs with syntax coloring. Show a useful preview of large tool output and provide a way to view the full result.

Clarification must use a bounded, vertically scrollable card. Wrap the entire question, reason, and option text at 60 columns. On mount, keep the title and question visible while focusing the first option without scrolling it into view. Number keys 1–9 select options; Up/Down move focus and scroll to the focused option; Enter activates it; Tab reaches the custom input; Escape cancels. Digits typed into the custom input must stay in the input. Empty custom input must never silently approve the first option. Show an “agent waiting” indicator on the main screen. Verify mouse buttons and keyboard behavior with Textual pilot tests at small and normal terminal sizes.

### Phase 6 — workspace, sessions, and skills

Add `--workspace` and `/workspace [path]`; display the active directory in status. Validate that it exists, update every workspace-bound tool, and prevent switching while a task is active. File reading, writing, editing, listing, and test paths must remain within the workspace, including after symlink resolution. Avoid overlapping tasks that mutate the same conversation.

Store sessions in SQLite or another durable local store. Provide `/sessions`, `/session new [title]`, `/session load <id>`, `/session save`, and `--session <id>`. Save at task completion and app exit. Restore model selection, workspace, message history, and active skills. Do not put API keys in session data. Ensure `:memory:` works for tests.

Discover `SKILL.md` files from workspace and user configuration directories. Provide `/skills` and `/skill <name|off>` with clear active status. Read only the user's selected skill into the task system context; cap size, handle missing files, and persist selected names in the session. Keep skill instructions subordinate to direct user instructions and tool safety constraints.

### Phase 7 — systematic verification

Write tests that target observable behavior: manual model ID survives every agent step; local classifier fallback is reported; domain and thinking events use the configured engine; risk is checked again on generated tool calls; canceled actions do not execute; malformed tool arguments become visible failures; API errors never appear as success; sessions round-trip; workspace switching updates tools; path traversal fails; pytest arguments do not invoke a shell; and long clarification text wraps inside a 60×20 terminal. Run `pytest tests/ -v`, CLI help and task smoke checks, a headless TUI pilot, and `git diff --check`. Avoid tests that merely duplicate implementation details.

### Phase 8 — documentation and review

Update README and walkthrough with the actual commands, classifier model choices, local Ollama and ONNX setup, domain behavior, thinking levels, sessions, workspace, skills, clarification keys, and limitations. Remove outdated claims that entropy pauses tasks or that an unavailable tool exists. Review the final diff for accidental database or bytecode changes, stale imports, uncaught exceptions, hidden status, and UI text clipping. Report what was implemented, what was tested, and which external backends could not be exercised locally.

## Acceptance criteria

The TUI can ask a long question on a 60×20 terminal with a visible title and full wrapped text. The user can answer or cancel using keyboard alone. The user can choose a workspace, resume a session, enable a skill, choose a classifier backend/model, and force or unforce the task model. Domain and thinking decisions appear in telemetry. A failed API request, unverified tool result, or destructive command without authorization does not become a successful completion. The full test suite and CLI smoke checks pass.
