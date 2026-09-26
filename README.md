# Adaptive Agent Harness 2.0

## Current developer agent setup

Install the project into its virtual environment, then run `adaptive-harness dev "git status"` or `adaptive-harness tui`. With no API key or custom task-model endpoint, completions use the offline mock client. Set `OPENROUTER_API_KEY` or pass `--key` for live completions. `--base-url` points the task model at an OpenAI-compatible local server; the classifier backend is configured separately.

Use `adaptive-harness dev --offline "check status"` to explicitly run the no-network mock path even when credentials are configured. The mock supports simple demonstration tools, but reports implementation requests it cannot fulfill as errors; `dev` exits with status 1 for incomplete tasks. Live request failures are shown as errors, their response text is redacted for the configured key, and the current request is never presented as completed work.

The default OpenRouter model is `stealth/space-bunny-alpha` in manual selection mode. The agent keeps the active model for every step. `--model MODEL_ID` or `--tier fast|standard|reasoning` selects another fixed model; `--model auto` or `/model auto` enables complexity-based routing explicitly. In the TUI, `/model` or F4 opens a searchable OpenRouter catalog. The catalog falls back to built-in and previously fetched choices when the network is unavailable. A fixed model appears as `MANUAL` in telemetry.

The workspace defaults to the directory where you launch the command. Choose another with `--workspace PATH` or `/workspace PATH`; a resumed TUI session keeps the launch workspace. Conversations and classifier settings save automatically to the configured SQLite database. Use `/sessions` or F5 for a searchable session picker, `/sessions list` to print IDs, `/session new [title]` to start fresh, `/session load ID` to resume, or `--session ID` at launch. The harness includes 22 craft skills across software engineering, testing, mathematics, research, infrastructure, and security. SemIf selects one locally from typed descriptions when it is available; a local lexical router takes over otherwise. `/skills` or `/skill list` shows the full catalog, `/skill NAME` forces a skill, and `/skill off` restores automatic routing. Both `dev` and `tui` accept repeatable `--skill NAME` options. Coding tools remain available even when a selected skill declares fewer tools.

Custom skills can live in `./skills/NAME/`, `.harness/skills/NAME/`, or `~/.config/adaptive-harness/skills/NAME/`. Each package needs a `SKILL.md` instruction file. An optional `skill.json` declares `title`, `category`, `trigger`, `tools`, `invariants`, and `icon`; without it, a custom skill gets read-only inspection tools. For example:

```json
{"title":"Release Review","category":"Custom","trigger":"Review release readiness","tools":["read_file","search_files","run_pytest"],"invariants":["tests_green"]}
```

The telemetry panel shows the selected skill, category, confidence, and actual toolset. Skill verification reports checks still missing from observed tool evidence; a text claim alone does not satisfy an invariant.

The classifier auto-selects local SemIf when PyTorch, Transformers, and a cached checkpoint are present; otherwise it uses TF-IDF and logistic regression. Install the optional runtime with `pip install '.[semif]'`, then download the default Qwen3.5 4B weights into the Hugging Face cache with `hf download Qwen/Qwen3.5-4B`. SemIf runs locally and scores candidate choices from next-token logits in one model pass without generating text. Select it with `--classifier-engine semif --semif-model Qwen/Qwen3.5-4B`; use `--classifier-engine sklearn` for the lightweight fallback. TUI: `/classifier semif [model_path]` or `/classifier sklearn`. If the model, tokenizer, or optional runtime cannot load, the current task falls back to sklearn and reports why.

Control SemIf hardware and calibration with `--semif-device auto|cpu|cuda|mps`, `--semif-4bit` (CUDA plus bitsandbytes), and `--semif-temperature FLOAT`. Checkpoints are loaded from local paths or the Hugging Face cache without network downloads during agent startup. Other engines remain available with `--classifier-backend ollama|local-slm|onnx|openrouter` and `--classifier-model MODEL`. `--classifier-endpoint URL` sets a custom HTTP endpoint for Ollama or OpenRouter classification.

For a local SLM, start Ollama and pull a small model, such as `ollama pull qwen2.5:1.5b`, then run:

```bash
adaptive-harness tui --classifier-backend ollama --classifier-model qwen2.5:1.5b
```

The Ollama backend calls the local `/api/generate` endpoint with JSON output. For llama.cpp or another local OpenAI-compatible server, select `local-slm` and set `--classifier-endpoint` to its `/v1/chat/completions` URL. The ONNX backend requires `onnxruntime` and `transformers` installed separately and a local model directory containing `model.onnx` and tokenizer files. It scores task and label embeddings by cosine similarity. The OpenRouter classifier uses `OPENROUTER_API_KEY` and a separate, configurable classifier model; it consumes API tokens.

Each task also receives domain, thinking, model-tier, skill, and tool-verification decisions from the selected classifier engine. SemIf classifies these using typed choices; sklearn provides statistical fallback. Choose `--mode coding|research|science|security|auto` for either `dev` or `tui`, or change it with `/mode NAME` in the TUI; `security` maps to the internal audit mode. The selected mode changes the system guidance and available tools. Coding validates edited Python syntax, research checks that web results contain source URLs, science exposes a finite-value convergence check, and security restricts shell commands to read-only Git inspection. Set `BRAVE_SEARCH_API_KEY` to enable cited web search. Destructive-operation checks remain separate safety gates.

Use `--thinking NAME` or `/thinking NAME` to choose a model reasoning effort; `auto` restores classifier selection. `/thinking` opens a picker limited to the active model's known efforts. Space Bunny Alpha supports `low`, `medium`, `high`, `xhigh`, and `max`; legacy `deep` maps to `high`. The displayed token budget is a request target, not a measured count. Unsupported efforts are rejected when a model is fixed. The telemetry panel shows domain and thinking badges, classifier latency, entropy, margin, skill probabilities, and model selection. The prompt sits in the normal vertical layout with a blank row above the footer, including when command suggestions appear.

Tool-step limits follow a step policy. The default `--step-policy classifier` runs without a fixed step cap: the runtime overseer watches tool trajectories, injects a visible system prompt when the agent repeats unnecessary tool calls (a stop-circling directive), and stops the run when repeated interventions show no path to progress. `--step-policy fixed` uses bounded budgets from 4 to 24 steps by thinking level, and `--step-policy unbounded` removes both the cap and all classifier interventions. `--max-steps N` overrides any policy with an explicit cap; swarm subagents follow the same policy as the session. In the TUI, `/steps classifier|fixed|unbounded|N` switches at runtime and the telemetry panel shows the active policy and limit. Every classifier or harness prompt injection is displayed verbatim to the user, and the fully assembled ingested system prompt (workspace, mode guidance, user preferences, skills, memory exemplars) is shown before each task so model steering can be audited.

Every prompt the models receive is centralized in `src/adaptive_harness/prompts.py`: base and role system prompts, domain guidance, overseer interventions, the stop-circling and termination prompts, harness nudges, swarm role instructions, and the LLM classifier prompts. `adaptive-harness prompts list` shows all names, `prompts show NAME` prints one, `prompts export` writes an editable JSON copy, and `prompts path` lists the override locations; the TUI offers the same via `/prompts`. To customize, map prompt names to new text in `.harness/prompts.json` inside the project or `~/.config/adaptive-harness/prompts.json` for all projects (an explicit `ADAPTIVE_PROMPTS_FILE` wins over both). Overrides are applied per workspace and are marked in the listings.

The default interaction profile is `turbo`: the agent executes ordinary edits, shell commands, and tests without clarification. It asks before catastrophic actions such as disk formatting or `sudo`. `balanced` adds high-risk command approval; `strict` requests approval before each shell, file edit/write, test, REPL, or web tool call. Set this with `--safety` or `/safety`. The TUI status line displays the selected profile. Legacy `cautious` remains available for semantic uncertainty prompts. The TUI status line shows separate thinking and visible-response phases; provider requests are non-streaming, so it does not expose private reasoning tokens. `/usage` and the session status show input, output, reasoning, cache-read, cache-write, total tokens, and provider-reported cost when available. Usage is saved with the session; unsupported local/offline costs are shown as unavailable rather than estimated. At terminal heights below 12 rows, the chat and status panels hide temporarily to keep the prompt and Footer separated; resizing restores them.

Context reduction is automatic. `read_file` accepts `symbol="Class.method"` to return a Python AST slice and module imports; an unbounded read of a large file returns its first 120 lines and asks the agent to narrow the next read. Long tool outputs are cleaned of terminal controls, repeated lines are collapsed, and pytest failures retain assertions and traceback evidence before the next model call. Search results spanning many files are locally ranked to three representative files, using SemIf when active. The sidebar shows an **estimated** token reduction from compacted tool text and the provider's **reported** cached prompt tokens. Actual savings depend on the task and provider; no fixed percentage is assumed.

Science mode exposes `run_python_repl` and `verify_equation`. The REPL preloads `math`, NumPy, SciPy, and SymPy inside a Bubblewrap sandbox with no network or workspace mount; it fails closed if Bubblewrap is unavailable. State resets per call. `verify_equation` uses a restricted expression grammar and exact SymPy substitution, including checks for invalid denominators. NumPy floating-point operations remain approximate. Python writes and edits are AST-checked *before* the file changes.

Successful read-file tasks can be reused from SQLite without another model request when the prompt and settings match and every read source still has the same SHA-256 hash. Path case is preserved for case-sensitive filesystems. The fast path does not reuse writes, shell commands, tests, or network results. Reusable non-destructive clarification answers are saved privately in `~/.config/adaptive-harness/preferences.json` and applied to the same question on later runs. Destructive approvals, one-time task targets, and credential questions are never remembered. OpenRouter Claude requests opt into automatic prompt caching; the displayed cache counter uses usage data actually returned by the provider. [OpenRouter documents the cache behavior and provider-routing trade-off](https://openrouter.ai/docs/guides/best-practices/prompt-caching).

Shell and pytest tools enforce bounded timeouts, kill child processes when a timeout expires, and retain at most 200 KB of each output stream. The shell filter blocks recursive deletion of filesystem roots, common fork bombs, and destructive disk commands; treat shell access as a developer tool and rely on the domain and tool policy as additional restrictions. Security mode allows read-only Git inspection and the direct `pip-audit` and `safety check` commands.

Clarification opens for detected destructive operations or a genuinely missing task target. Numeric keys 1–9 select options, arrows navigate, Enter confirms, Tab reaches the custom instruction field, and Escape cancels. Long text wraps in a scrollable card, and empty custom input cannot approve an action. Enter in a searchable popup is contained there and cannot submit its search text as a task. Entropy and requests for the agent's design judgment do not pause the task. F2 opens a live theme preview; F3 opens isolated diff review, while F6 toggles telemetry on wide terminals. `/output` opens a selectable full-text viewer for the latest agent response; `/tool-output` opens the latest tool result. Ctrl+Y or Ctrl+Shift+C copies a selection, the latest response, or a pending diff.

The TUI edits the launch workspace directly by default. Use `/isolation on|off|auto` to opt into Git worktree isolation; `auto` selects multi-step editing tasks. When enabled in a Git repository, a task uses `.harness/worktrees/task-ID` from the current `HEAD`; F3 or `/diff` opens patch review before changes reach the main checkout. A dirty or advanced main checkout blocks patch application safely.

In a directory outside Git, in a newly initialized repository with no commits, or when worktree creation fails, the TUI continues directly in the selected workspace and shows a notice. This also applies when `/isolation on` or `/swarm on` is selected. Direct edits have no patch review gate. `write_file` creates missing parent directories, so tasks can write into new subfolders.

Swarm mode defaults to `auto`, which pipelines only explicit multi-agent requests — complex edits are model-driven: the main agent keeps `delegate_subagent` and decides when to spawn roles, so long prompts are never hijacked by heuristics. The pipeline runs Architect, Coder, and then security and QA reviewers **in parallel** (read-only roles run concurrently; only the Coder writes). The Architect plans with read tools; the Coder has all core coding tools and sole write access to the launch workspace or selected worktree; reviewers inspect files and run checks, and QA verification requires real evidence (passing tests or successful inspections). Failed or unverified reviews route back to the Coder with the findings as feedback for up to two repair cycles, and transient role failures are retried once with the failure cause included. The sidebar shows all four role states.

Requests that explicitly mention multiple agents also enter swarm mode automatically. Headless `adaptive-harness dev --swarm "TASK"` exposes `delegate_subagent` for architect, coder, reviewer, and security roles. The Coder can write files and run build commands inside the selected workspace. When a model returns fenced code with an explicit `path:` header, the agent writes it using `write_file`; a plain code answer never counts as a completed edit. Files created through shell commands count as edits based on observable file state, not tool names. Subagent activity (tool calls, injected prompts, stop reasons) streams into the main log with a `⇢ subagent` prefix, summaries are never empty, and failures always report their error and stop reason.

Verified developer trajectories are stored in `output/experience.db`. Similar verified tasks from the same workspace can provide a bounded example before a model request. Run `adaptive-harness distill --db output/experience.db --output output/distillation --model qwen2.5-0.5b` to export an instruction JSONL dataset, LoRA config, and optional local training script. Other model choices are `qwen2.5-1.5b` and `smollm2`. Only traces marked verified success are exported.

Install `pip install 'adaptive-harness[plotting]'` to enable `plot_terminal` line, scatter, and bar charts in science and benchmark tasks. The tool validates bounded finite data and prints an inline terminal chart. The sandboxed `run_python_repl` also exposes optional `plotext` as `plt` for scripts that call `plt.show()`; the optional package must be installed before use.

Use `/copy` or Ctrl+Shift+C to copy selected chat text, or copy the latest agent reply when no text is selected. If OpenRouter rejects a thinking parameter, the request retries once with that parameter removed and keeps the chosen model.

The TUI stores provider keys in `~/.config/adaptive-harness/credentials.json` with mode `0600`; existing OpenRouter keys in `config.json` remain readable for compatibility. Use `/key <provider> <key>`, `/key status`, and `/key clear <provider>`; `/key KEY` and `/key clear` act on the active provider. `/provider <openrouter|anthropic|openai|deepseek|google|groq|local>` switches the task model endpoint. Headless and TUI startup accept `--provider`; `--base-url` can point to a local OpenAI-compatible endpoint. Repeat `--backup-provider NAME` to opt into failover on rate limits or server errors; backups need their own key, except local endpoints. Direct Anthropic and Gemini currently use their documented OpenAI compatibility endpoints, so provider-specific features outside that interface are not available. Typing `/` opens a floating command palette above the prompt; keep typing to filter, use Up/Down to select, Tab or Enter to complete, and Escape to close it. Completion fills the input without submitting a task. The prompt input recalls the last 500 task prompts with Up/Down, restoring an unfinished draft when you return to the bottom; slash commands are excluded from `prompt_history.txt`. `/new` or Ctrl+N starts a clean session, while `/reset` clears the current one. `/theme` previews built-in themes on hover or arrow focus; Enter saves and Escape restores the old theme. Inline `$...$` math renders common LaTeX symbols as Unicode; display `$$...$$` and `\[...\]` fractions render as stacked monospaced equations. Fenced code stays literal. `/export markdown|json` writes the session transcript and tool calls/results to the workspace's `output/sessions/` folder. The header shows provider, context utilization, and token totals; scrolling up pins the chat log until you return to the bottom.

After each tool result, the local runtime overseer checks for repeated actions, unsupported success claims, stalled errors, and clearly off-target edits. A targeted correction enters the next model request when evidence warrants it; repeated unresolved loops escalate to a clarification in the TUI. A fast deterministic gate handles clear cases. For an optional smaller semantic second tier, install `.[onnx]` and supply a local ONNX embedding directory through `--overseer-model PATH`. The existing large SemIf checkpoint is not loaded solely for runtime oversight. Conversation history stays verbatim below 75% of the active model's estimated context window; above that point, a request-only copy compacts older tool outputs and, if needed, summarizes older complete turns. System and overseer instructions, recent turns, and active diffs are preserved. The sidebar shows the overseer state and context gauge. Token counts before provider response are estimates; they do not guarantee a particular latency or savings percentage.

> **An Autonomous AI Developer Agent Harness powered by OpenRouter LLMs and Pervasive ML Classifiers everywhere — featuring real-time intent routing, middle-of-development clarification dialogs, multi-tier complexity routing, active output verification, and an interactive Terminal User Interface (TUI).**

---

## 1. Overview & Core Philosophy

Modern AI developer agents often suffer from two major extremes: either blindly trusting a single monolithic LLM prompt to make every decision at exorbitant token cost, or hardcoding rigid scripts that fail when anything unexpected occurs.

The **Adaptive Agent Harness 2.0** solves this by embedding **pervasive machine learning classifiers everywhere** throughout the agentic lifecycle:

1. **Skill & Intent Classifier (`SkillClassifier`)**: Routes developer tasks into specialized tool pipelines with a probability distribution. The optional SemIf engine reads local model logits directly and emits typed probabilities without generating or parsing text.
2. **Ambiguity & Risk Classifier (`AmbiguityClassifier`)**: Quantifies task uncertainty via Shannon entropy $H(p)$ and confidence margin. In the interactive SemIf path, high uncertainty can open the clarification modal; destructive actions also remain behind a separate risk check.
3. **Cognitive Complexity & Model Tier Router (`ComplexityRouter`)**: Dynamically routes requests across LLM model tiers:
   - **Fast Tier** (e.g. `google/gemini-2.5-flash-lite`): rapid queries, file reads, git status, typo fixes.
   - **Standard Tier** (`stealth/space-bunny-alpha`): core coding, refactoring, unit test authoring.
   - **Reasoning Tier** (e.g. `anthropic/claude-sonnet-4`): complex architectures, concurrency/deadlocks, multi-file algorithms.
4. **Tool Verification & Self-Healing Classifier (`VerificationClassifier`)**: Actively inspects tool outputs (bash stdout/stderr, pytest assertions, compiler syntax errors, missing paths) to classify failure modes (`SYNTAX_ERROR`, `TEST_FAILURE`, `FILE_ERROR`, `RUNTIME_ERROR`) and immediately trigger targeted recovery actions (`AUTO_RETRY_SYNTAX_FIX`, `AUTO_RETRY_TEST_FIX`, etc.).
5. **Interactive Textual TUI (`AdaptiveHarnessApp`)**: A full terminal IDE featuring live classifier telemetry gauges, probability bar charts, streaming agent thought logs, and interactive clarification modals.
6. **OpenRouter Protocol & Offline Fallback**: Direct integration with OpenRouter's API (`https://openrouter.ai/api/v1`) and OpenAI-compatible endpoints, paired with a **Mock LLM engine** for simple offline demonstrations without requiring a paid API key.

---

## 2. Pervasive Classifiers Everywhere Architecture

```text
                                    ┌───────────────────────┐
                                    │    Developer Input    │
                                    └───────────┬───────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     ┌───────────────────────┐                                     ┌───────────────────────┐
     │   Skill Classifier    │                                     │ Complexity Router     │
     │  (Intent Probability) │                                     │ (Fast / Std / Reason) │
     └───────────┬───────────┘                                     └───────────┬───────────┘
                 │ p(skill), H(p), Margin                                      │
                 ▼                                                             │
     ┌───────────────────────┐                                                 │
     │ Ambiguity Classifier  │                                                 │
     │  (Risk & Uncertainty) │                                                 │
     └───────────┬───────────┘                                                 │
                 │                                                             │
        ┌────────┴────────┐                                                    │
   High │ Uncertainty     │ Low Risk / Clear                                   │
        ▼                 ▼                                                    │
 ┌─────────────┐   ┌────────────────────────────────────────────────────────┐  │
 │  INTERACTIVE│   │                     LLM Agent Loop                     │  │
 │CLARIFICATION│──►│            (OpenRouter / Multi-Tier Model)             │◄─┘
 │    MODAL    │   └────────────────────────────┬───────────────────────────┘
 └─────────────┘                                │ Tool Calls
                                                ▼
                   ┌────────────────────────────────────────────────────────┐
                   │                  Developer Tools Suite                 │
                   │  (run_bash, read_file, write_file, edit_file, pytest)  │
                   └────────────────────────────┬───────────────────────────┘
                                                │ Tool Output
                                                ▼
                   ┌────────────────────────────────────────────────────────┐
                   │                 Verification Classifier                │
                   │ (SUCCESS | SYNTAX_ERROR | TEST_FAILURE | FILE_ERROR)   │
                   └────────────────────────────┬───────────────────────────┘
                                                │
                             ┌──────────────────┴──────────────────┐
                             ▼                                     ▼
                      [Needs Recovery]                         [Proceed]
                             │                                     │
                             ▼                                     ▼
                    Feed Invariants /                     Synthesize Final
                    Compiler Fix to LLM                   Developer Response
```

### The 4 Pillars of Classification

| Classifier | What It Analyzes | Key Metrics / Classes | Action Triggered |
| :--- | :--- | :--- | :--- |
| **`SkillClassifier`** | Developer prompt, code context | `code_edit`, `run_command`, `search_explore`, `testing`, `ask_clarification`, `general_reasoning` | Activates optimal tool subset and prompt framing |
| **`AmbiguityClassifier`** | Intent distribution entropy $H(p)$, confidence margin $\Delta p$, destructive tokens (`rm`, `drop`, `reset --hard`) | Shannon entropy $H(p) = -\sum p \log_2 p$, Margin $p_1 - p_2$, Risk: `LOW`, `MEDIUM`, `HIGH` | Opens clarification for destructive actions or explicit decision ambiguity |
| **`ComplexityRouter`** | Task scope, vocabulary depth, architectural cues | `FAST` (Gemini Flash Lite), `STANDARD` (GLM 5.3 Flash), `REASONING` (Claude Sonnet 4) | Dynamic model selection minimizing cost and maximizing reasoning depth |
| **`VerificationClassifier`** | Execution exit codes, Python AST tracebacks, pytest failures, regex patterns | `SUCCESS`, `SYNTAX_ERROR`, `TEST_FAILURE`, `FILE_ERROR`, `RUNTIME_ERROR` | Auto-recovery routing (`AUTO_RETRY_SYNTAX_FIX`, `AUTO_RETRY_TEST_FIX`, etc.) |

---

## 3. Terminal User Interface (TUI)

Launch the interactive Textual terminal app:

```bash
# Launch with default offline mock mode
adaptive-harness tui

# Or launch with your OpenRouter API key
adaptive-harness tui --key sk-or-v1-xxxxxxxxxxxxxxxxx
```

### TUI Features
- **Live Classifier Telemetry Panel**:
  - Live **Model Tier** badge (`[FAST]`, `[STANDARD]`, `[REASONING]`).
  - Active model identifier and top predicted developer skill.
  - Real-time **Shannon Entropy meter** ($H(p)$ in bits) and **Confidence Margin**.
  - Color-coded **Risk Level** badge (`LOW`, `MEDIUM`, `HIGH`).
  - Animated ASCII probability bars for all developer skill classes.
- **Middle-of-Development Clarification Modal**:
  - Destructive operations or requests with no identifiable target open an interactive modal with selectable options or custom input.
- **Rich Streaming Agent Log**:
  - Live streaming of agent thoughts, tool execution invocations with argument inspection, real-time tool results, and verification classifications.
- **Built-in Slash Commands**:
  - `/key <OPENROUTER_API_KEY>`: Save a private OpenRouter key; `/key status` and `/key clear` inspect or remove it.
  - `/model` or F4: Search the OpenRouter catalog; `/model <MODEL_ID>` switches directly.
  - `/tier <fast|standard|reasoning>`: Switch between optimized cost/capability tiers.
  - `/mode <coding|research|science|security|auto>`: Select an operational mode.
  - `/thinking <auto|low|medium|high|xhigh|max>`: Select an effort supported by the active model (`deep` remains a `high` alias).
  - `/safety <turbo|cautious>`: Choose the question frequency profile.
  - `/sessions` or F5: Search and resume a saved session.
  - `/new`, `/reset`: Start a new session or clear the current session state.
  - `/theme`, `/history`, `/export markdown|json`: Preview colors, recall prompts, and export a transcript.
  - `/clear`: Clear only the visible terminal log.
  - `/help`: Display available commands.
  - `/exit`: Terminate application.

---

## 4. Developer Tools Suite

The agent is equipped with a workspace-scoped developer toolbelt (`src/adaptive_harness/tools/`):

- **`RunBashTool` (`run_bash`)**: Executes workspace commands with timeouts, output capture, virtualenv path resolution, and dangerous command safety filtering (blocks `rm -rf /`, fork bombs, etc.).
- **`ReadFileTool` (`read_file`)**: Reads workspace files with line numbering, line bounds, or a Python `symbol` slice.
- **`WriteFileTool` (`write_file`)**: Writes new files or overwrites existing files, checking Python syntax before changing them.
- **`EditFileTool` (`edit_file`)**: Replaces unique code blocks, checks Python syntax before saving, and produces unified diffs.
- **`ListDirectoryTool` (`list_directory`)**: Lists directory contents with sizes and folder indicators, filtering out hidden noise.
- **`SearchFilesTool` (`search_files`)**: Recursive regex grep across workspace files with extension filtering.
- **`RunPytestTool` (`run_pytest`)**: Executes pytest suites, parses structured passed/failed counts, and extracts failure stack traces.
- **`AskUserTool` (`ask_user`)**: Prompts the developer in the middle of development with multiple-choice buttons or free-text answers.
- **`RunPythonReplTool` (`run_python_repl`)**: Runs scientific Python inside a restricted Bubblewrap process with no network or workspace mount.
- **`VerifyEquationTool` (`verify_equation`)**: Checks exact SymPy substitution and denominator validity for a proposed root.
- **`CompileTypstTool` (`compile_typst`)**: Compiles a Typst source to a publication PDF, resolving Typst from `PATH` or the `typst` Python wrapper. Any Typst warning fails the build.

### Autonomous research tools (`src/adaptive_harness/tools/research_swarm.py`)

These operate on a live `ResearchSwarm` rather than the workspace, and back the
research mode described in section 5.1:

- **`SpawnSubagentTool` (`spawn_subagent`)** / **`ScaleDivisionTool` (`scale_division`)**: grow or shrink a division's worker pool on demand.
- **`VerifyProofTool` (`verify_proofs`)** / **`RunExperimentTool` (`run_experiments`)**: adjudicate the exact-derivation and seeded-replication gates.

---

## 5. Command-Line Interface (CLI)

The CLI provides subcommands for both developer agent tasks and empirical routing benchmarks:

### 1. Developer Agent Tasks (`dev`)
Run developer tasks directly from your shell with full classifier diagnostics:

```bash
# Inspection task (routes to FAST tier)
adaptive-harness dev "list files in workspace"

# Code testing task (routes to STANDARD tier, runs pytest, verifies output)
adaptive-harness dev "run pytest on test_agent_and_tools.py"

# Custom OpenRouter model
adaptive-harness dev "refactor the storage layer" --key $OPENROUTER_API_KEY --model anthropic/claude-sonnet-4
```

### 1.1 Autonomous Research Swarm (`research`)

Give it a topic. It derives propositions, decides each one by executing a
self-adjudicating derivation script, and publishes a mathematical paper stating
the result — `PROVEN`, `DISPROVEN`, or `INCONCLUSIVE`. It does not audit research
you did yourself; it does the research.

```bash
# Derive, prove, corroborate, and publish. No network requests, no cost.
adaptive-harness research "pareto heavy tailed network delay: critical index and variance-optimal balanced routing"

# Settle a specific algebraic claim directly
adaptive-harness research "quadratic expansion" \
    --claim "(x+y)**2 == x**2 + 2*x*y + y**2" --symbols x,y

# A claim that is false is refuted, and that is a success
adaptive-harness research "bad claim" --claim "(x+y)**2 == x**2 + y**2" --symbols x,y
```

**How a claim is settled.** The kernel constructs a machine-checkable statement
from a definition, writes a self-adjudicating script for it, and the exit code
*is* the verdict:

| Exit | Verdict | Meaning |
| --- | --- | --- |
| `0` | `PROVEN` | the difference `lhs - rhs` is identically zero |
| `3` | `DISPROVEN` | an exact rational witness makes the difference nonzero |
| other | `INCONCLUSIVE` | the attempt could not decide — never treated as a refutation |

Refutation requires an *exhibited* counterexample, not a failure to simplify:
`exp(x) == 1 + x` is refuted at `x = 1`, and `sqrt(x^2) == x` at `x = -1`, which
is why the probe set includes negative points. A script that errors or times out
is `INCONCLUSIVE`, because a broken proof attempt is not evidence against a claim.

**The second proof tier: Lean 4.** SymPy can confirm that two expressions are
equal; only a theorem prover can confirm that a *deduction* is valid. Every
theorem the swarm publishes can carry a formal Lean 4 counterpart, compiled by the
Lean kernel:

```bash
adaptive-harness research "Formally prove that the sum of the first N natural numbers equals N*(N+1)/2 and verify both in SymPy and Lean 4"
```

A formal proof is admitted only when **all three** of these hold:

| Check | Source of truth |
| --- | --- |
| the source contains no `sorry`/`admit` token | static scan, after comments and string literals are stripped |
| the compiler emitted no "declaration uses 'sorry'" | `lean` diagnostics |
| no declaration depends on `sorryAx` | `#print axioms`, emitted by the kernel |

The third check is why the exit code alone is not trusted. **Lean exits 0 on a
proof containing `sorry`** — it emits a *warning*, not an error. A verifier that
reads only the exit status would certify a false theorem; this one does not, and
`tests/test_lean_verification.py::test_tampered_lean_file_blocks_the_run` pins
that behaviour end to end.

What is required depends on the verdict, because the standard applies to
*asserted theorems*:

- **PROVEN** — the paper asserts a theorem, so every Lean file must be certified.
  An empty proof set blocks publication.
- **DISPROVEN** — nothing is asserted; the refutation *is* the result, certified
  by an exact witness the kernel checked. Requiring a Lean proof to publish "this
  claim is false" would be wrong.
- **INCONCLUSIVE** — no theorem is published, so the tier is not applicable and
  the paper says so.

**Core Lean only.** The shipped proofs use no imports beyond Lean's prelude,
because Mathlib cannot be assumed present: on this machine its cache needs about
5 GB and the fetch failed for want of space. Core Lean has no `ring`, `linarith`,
or `ring_nf`, so polynomial identities are distributed by hand — the Gauss proof
below expands `(k+1)(k+2)` explicitly for that reason. Every library proof is
compiled by a test, so the library cannot rot silently. The tool resolves
`lake env lean` automatically when a `lakefile` is present, so a Mathlib
environment works if one exists.

The paper gains a *Formal Foundations* section with a green certification box per
proof (toolchain, theorem names, axiom set, digest) and an appendix listing the
full Lean sources, so a reader can reproduce the check with
`lean proofs/lean/<name>.lean`.

**No approximation is admissible.** Before execution every script is scanned by
AST for floating-point literals and approximating calls (`float`, `evalf`, `N`),
and rejected without running if it contains any. This is not decoration: it
caught a genuine error in an early draft of the Pareto second moment, where the
asserted `α·x_m²/((α−1)(α−2))` was wrong and SymPy's `α·x_m²/(α−2)` was right.

**Six invariants**, all required simultaneously:

| Invariant | Requirement |
| --- | --- |
| `mathematical_soundness` | every derivation is approximation-free and reached a clean verdict |
| `empirical_replication` | every seeded simulation reproduced its prediction at 95% |
| `adversarial_clearance` | the *reported verdict* is certified — no open objection, and for a refutation an exact witness |
| `claim_adjudication` | the headline claim reached a decided verdict |
| `formal_verification` | every published theorem is backed by a Lean 4 proof the kernel checked, with zero `sorry` |
| `document_integrity` | `paper.typ` compiles to `paper.pdf` with **zero** Typst warnings |

Artifacts land under `research/<topic-slug>/`: the hash-chained
`comm_ledger.jsonl`, `00_objective_spec.md` (including the derivation plan),
`evidence/index.json`, `proofs/` and `experiments/` with their generated scripts
and raw CSV, `figures/`, `03_adversarial_audit.md`, `bibliography.bib`,
`convergence_history.json`, receipt indexes, and the generated `paper.typ` plus
its `paper.pdf`.

**The output is a real mathematical paper**, not a log: title, abstract,
introduction, a notation table, numbered theorems each with explicit hypotheses,
a formal statement typeset in 2D math, a proof ending in □, a consequence, and a
verdict line — followed by empirical corroboration, the adversarial audit, a
conclusion that matches the verdict, references, and appendices carrying the
ledger and every receipt. A topic the kernel cannot formalise produces an
explicit `INCONCLUSIVE` paper saying so, rather than a confident-sounding paper
about nothing.

**The loop is stagnation-limited, not turn-limited.** It runs until it converges
or until it can *prove* more identical work cannot help. Progress means reaching
a new *minimum* in outstanding gaps (a high-water mark), so a flapping invariant
oscillating 4→3→4→3 cannot look like progress. After `--patience` no-progress
cycles the worker budget doubles; when escalation cannot grow, the run concedes
with an honest `STAGNATION_ABORT` and UNSOLVED.

```python
from adaptive_harness.research import ResearchSwarm, SwarmConfig

swarm = ResearchSwarm("pareto heavy tailed network delay", root="research")
swarm.run()
print(swarm.claims.headline.value)   # PROVEN | DISPROVEN | INCONCLUSIVE
print(swarm.claims.summary())
```

Attach the same engine to a normal agent task so a model can drive it mid-task:

```bash
adaptive-harness dev "settle whether balanced routing minimises delay variance" \
    --research "balanced routing under heavy-tailed delay"
```

`--research TOPIC` exposes `spawn_subagent`, `scale_division`, `verify_proofs`,
and `run_experiments`. Like the standalone command it defaults to mechanical
mode and makes no network requests; the tools appear only in the investigative
modes, never in `security`/audit.

### 2. Algorithmic Routing & Recovery Benchmarks
```bash
# Run multi-policy ablation benchmark (Classifier vs Greedy vs Threshold vs Escalation)
adaptive-harness benchmark --n-samples 100 --save-plots

# Run empirical sweeps (confidence threshold trade-offs and data scaling curves)
adaptive-harness experiment

# Inspect SQLite experience store
adaptive-harness history --limit 10

# Summarize overall recovery and entropy metrics
adaptive-harness report

# Retrain routing classifier on historical verified traces
adaptive-harness retrain
```

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.12+
- Linux / macOS / Windows WSL

### Setup
```bash
# 1. Clone repository
git clone https://github.com/adaptive-agent-harness/harness.git
cd harness

# 2. Create virtual environment and activate
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode
pip install -e ".[dev]"
```

### Configure OpenRouter (Optional)
```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx"
```
*(If no API key is provided, the harness automatically runs in intelligent offline simulated mock mode).*

---

## 7. Verification & Automated Test Suite

The project includes unit and integration tests covering algorithmic strategies, mathematical calibration, developer tools, pervasive classifiers, the agent event loop, and the Textual TUI:

```bash
pytest tests/ -v
```

```text
============================= 257 passed in 36.37s =============================
tests/test_agent_and_tools.py::test_llm_client_mock_mode PASSED          [  0%]
tests/test_agent_and_tools.py::test_run_bash_tool PASSED                 [  0%]
tests/test_agent_and_tools.py::test_file_ops_tools PASSED                [  1%]
tests/test_research_swarm.py::test_proof_receipt_requires_exit_zero_and_exactness PASSED
tests/test_research_swarm.py::test_experiment_requires_a_hashed_data_artifact PASSED
tests/test_research_swarm.py::test_ledger_chain_verifies_and_detects_tampering PASSED
tests/test_research_swarm.py::test_loop_aborts_on_a_flapping_invariant_instead_of_spinning PASSED
tests/test_research_swarm.py::test_solved_run_produces_a_verified_ledger_and_a_pdf PASSED
... [strategies, benchmarks, calibration, swarm, TUI, and skills tests]
============================= 257 passed in 36.37s ==============================
```

The research tests are deliberately adversarial about the research machinery
itself: they assert that a flaky invariant cannot be mistaken for progress, that
an unsolved run stays terminated and reports UNSOLVED, and that a solved run's
PDF is byte-for-byte generated from its receipts.

---

## 8. Project Structure

```text
harness/
├── pyproject.toml                         # Project metadata and dependencies
├── README.md                              # Comprehensive architectural documentation
├── output/
│   ├── experience.db                      # SQLite store of verified execution traces
│   ├── models/                            # Serialized classifier models
│   └── plots/                             # Calibration, confusion matrix, & sweep plots
├── src/
│   └── adaptive_harness/
│       ├── agent/                         # Autonomous Developer Agent
│       │   └── agent.py                   # Event-driven streaming agent loop
│       ├── classifiers/                   # Pervasive ML Classifiers
│       │   ├── skill_classifier.py        # 6-class intent routing classifier
│       │   ├── ambiguity_classifier.py    # Shannon entropy & risk classifier
│       │   ├── complexity_router.py       # Multi-tier LLM routing (fast/std/deep)
│       │   └── verification_classifier.py # AST/output error classification & recovery
│       ├── llm/                           # LLM Client Layer
│       │   ├── client.py                  # OpenRouter & OpenAI client
│       │   └── mock_client.py             # Intelligent offline simulation client
│       ├── tools/                         # Developer Tools
│       │   ├── base.py                    # Tool abstract base class & OpenAI schema
│       │   ├── bash.py                    # Safe bash execution tool
│       │   ├── file_ops.py                # Read, write, and surgical edit tools
│       │   ├── workspace.py               # List directory and grep search tools
│       │   ├── testing.py                 # Pytest runner with structured parsing
│       │   └── clarification.py           # Interactive developer clarification tool
│       ├── tui/                           # Textual Terminal User Interface
│       │   ├── app.py                     # Full TUI application & slash commands
│       │   └── widgets.py                 # Telemetry panel & Clarification modal
│       ├── research/                      # Autonomous research swarm
│       │   ├── swarm.py                   # Executive Director, divisions, convergence loop
│       │   ├── ledger.py                  # Hash-chained comm_ledger.jsonl writer
│       │   ├── proof.py                   # Exact SymPy proof receipts (exit 0, no floats)
│       │   ├── experiment.py              # Seeded replication, data hashes, 95% intervals
│       │   ├── gate.py                    # Invariant gate & Relentless Convergence Loop
│       │   ├── paper.py                   # Typst paper generated from receipts
│       │   ├── typst.py                   # Typst resolution & warning-free compilation
│       │   ├── figures.py                 # Deterministic vector SVG figures
│       │   └── roles.py                   # Divisions, leaders, and worker roles
│       ├── cli.py                         # Typer CLI (tui, dev, research, train, benchmark)
│       ├── harness/                       # Algorithmic routing harness & verifier
│       ├── models/                        # Domain models, feature extraction, calibration
│       ├── data/                          # Dataset generator and SQLite repository
│       ├── strategies/                    # Specialized algorithmic problem solvers
│       ├── evaluation/                    # Ablation benchmarks, ECE, & matplotlib plots
│       └── dashboard/                     # Rich console output formatting
├── research/                              # Research artifacts (one dir per topic)
│   └── <topic-slug>/                      # Ledger, proofs, experiments, figures, paper.pdf
└── tests/
    ├── test_agent_and_tools.py            # Phase 2 test suite (TUI, agent, classifiers, tools)
    ├── test_research_swarm.py             # Receipts, ledger, invariant gate, publication
    └── test_*.py                          # Strategy, benchmark, and calibration unit tests
```

---

## 9. License

MIT License. Designed and built as an advanced agentic software engineering harness.
