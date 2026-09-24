# Adaptive Agent Harness 2.0

## Current developer agent setup

Install the project into its virtual environment, then run `adaptive-harness dev "git status"` or `adaptive-harness tui`. With no API key or custom task-model endpoint, completions use the offline mock client. Set `OPENROUTER_API_KEY` or pass `--key` for live completions. `--base-url` points the task model at an OpenAI-compatible local server; the classifier backend is configured separately.

Use `adaptive-harness dev --offline "check status"` to explicitly run the no-network mock path even when credentials are configured. Live request failures are shown as errors, their response text is redacted for the configured key, and the current request is never presented as completed work; the client switches to offline mock mode for the next request.

Model selection is automatic by default. The complexity router selects fast, standard, or reasoning for each task; the standard tier uses `z-ai/glm-5.3-flash`. `--model MODEL_ID` or `--tier fast|standard|reasoning` fixes the model for every step. In the TUI, `/model` or F4 opens a searchable OpenRouter catalog, while `/model MODEL_ID` selects one directly and `/model auto` restores automatic routing. The catalog falls back to built-in and previously fetched choices when the network is unavailable. A forced model appears as `FORCED` in telemetry.

Choose a working directory with `--workspace PATH` or `/workspace PATH`. Conversations and classifier settings save automatically to the configured SQLite database. Use `/sessions` or F5 for a searchable session picker, `/sessions list` to print IDs, `/session new [title]` to start fresh, `/session load ID` to resume, or `--session ID` at launch. The harness includes 22 craft skills across software engineering, testing, mathematics, research, infrastructure, and security. SemIf selects one locally from typed descriptions when it is available; a local lexical router takes over otherwise. `/skills` or `/skill list` shows the full catalog, `/skill NAME` forces a skill, and `/skill off` restores automatic routing. Both `dev` and `tui` accept repeatable `--skill NAME` options. Only the selected skill instructions and tools enter a task request, and domain restrictions still limit available tools.

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

Thinking levels are `none` (0), `low` (about 1k), `medium` (about 4k), and `deep` (16k) reasoning tokens. Use `--thinking NAME` or `/thinking NAME`; `auto` restores classifier routing. A manual thinking choice also selects a corresponding model tier unless a specific model was forced. Supported OpenRouter models receive reasoning effort; Claude requests use a direct token cap and a completion limit above it. Model support varies, so the displayed budget is a request target, not a measured token count. The telemetry panel shows domain and thinking badges, classifier latency, entropy, margin, skill probabilities, and model selection. The prompt sits in the normal vertical layout with a blank row above the footer, including when command suggestions appear.

The default interaction profile is `turbo`: the agent proceeds through ordinary uncertainty without opening a question, while destructive operations and requests with no identifiable target still pause. Set `--safety cautious` or `/safety cautious` to allow semantic uncertainty to trigger a question; `/safety strict` requests approval before each shell, file edit/write, test, REPL, or web tool call. The TUI status line shows separate thinking and visible-response phases; provider requests are non-streaming, so it does not expose private reasoning tokens. `/usage` and the session status show input, output, reasoning, cache-read, cache-write, total tokens, and provider-reported cost when available. Usage is saved with the session; unsupported local/offline costs are shown as unavailable rather than estimated. At terminal heights below 12 rows, the chat and status panels hide temporarily to keep the prompt and Footer separated; resizing restores them.

Context reduction is automatic. `read_file` accepts `symbol="Class.method"` to return a Python AST slice and module imports; an unbounded read of a large file returns its first 120 lines and asks the agent to narrow the next read. Long tool outputs are cleaned of terminal controls, repeated lines are collapsed, and pytest failures retain assertions and traceback evidence before the next model call. Search results spanning many files are locally ranked to three representative files, using SemIf when active. The sidebar shows an **estimated** token reduction from compacted tool text and the provider's **reported** cached prompt tokens. Actual savings depend on the task and provider; no fixed percentage is assumed.

Science mode exposes `run_python_repl` and `verify_equation`. The REPL preloads `math`, NumPy, SciPy, and SymPy inside a Bubblewrap sandbox with no network or workspace mount; it fails closed if Bubblewrap is unavailable. State resets per call. `verify_equation` uses a restricted expression grammar and exact SymPy substitution, including checks for invalid denominators. NumPy floating-point operations remain approximate. Python writes and edits are AST-checked *before* the file changes.

Successful read-file tasks can be reused from SQLite without another model request when the prompt and settings match and every read source still has the same SHA-256 hash. Path case is preserved for case-sensitive filesystems. The fast path does not reuse writes, shell commands, tests, or network results. Reusable non-destructive clarification answers are saved privately in `~/.config/adaptive-harness/preferences.json` and applied to the same question on later runs. Destructive approvals, one-time task targets, and credential questions are never remembered. OpenRouter Claude requests opt into automatic prompt caching; the displayed cache counter uses usage data actually returned by the provider. [OpenRouter documents the cache behavior and provider-routing trade-off](https://openrouter.ai/docs/guides/best-practices/prompt-caching).

Shell and pytest tools enforce bounded timeouts, kill child processes when a timeout expires, and retain at most 200 KB of each output stream. The shell filter blocks recursive deletion of filesystem roots, common fork bombs, and destructive disk commands; treat shell access as a developer tool and rely on the domain and tool policy as additional restrictions. Security mode allows read-only Git inspection and the direct `pip-audit` and `safety check` commands.

Clarification opens for detected destructive operations or a genuinely missing task target. Numeric keys 1–9 select options, arrows navigate, Enter confirms, Tab reaches the custom instruction field, and Escape cancels. Long text wraps in a scrollable card, and empty custom input cannot approve an action. Enter in a searchable popup is contained there and cannot submit its search text as a task. Entropy and requests for the agent's design judgment do not pause the task. F2 opens a live theme preview; F3 toggles telemetry on wide terminals. `/output` opens a selectable full-text viewer for the latest agent response; `/tool-output` opens the latest tool result. Ctrl+Shift+C copies a selection or the full response from the viewer.

Use `/copy` or Ctrl+Shift+C to copy selected chat text, or copy the latest agent reply when no text is selected. If OpenRouter rejects a thinking parameter, the request retries once with that parameter removed and keeps the chosen model.

The TUI stores `/key` credentials in `~/.config/adaptive-harness/config.json` with mode `0600`; key priority is `--key`, `OPENROUTER_API_KEY`, saved key, then offline mock. `/key status` shows the source and a masked value, and `/key clear` removes the saved key and switches the current TUI to mock mode. The prompt input recalls the last 500 task prompts with Up/Down, restoring an unfinished draft when you return to the bottom; slash commands are excluded from `prompt_history.txt`. `/new` or Ctrl+N starts a clean session, while `/reset` clears the current one. `/theme` previews built-in themes on hover or arrow focus; Enter saves and Escape restores the old theme. `/export markdown|json` writes the session transcript and tool calls/results to the workspace's `output/sessions/` folder. The header shows provider and token totals; scrolling up pins the chat log until you return to the bottom.

> **An Autonomous AI Developer Agent Harness powered by OpenRouter LLMs and Pervasive ML Classifiers everywhere — featuring real-time intent routing, middle-of-development clarification dialogs, multi-tier complexity routing, active output verification, and an interactive Terminal User Interface (TUI).**

---

## 1. Overview & Core Philosophy

Modern AI developer agents often suffer from two major extremes: either blindly trusting a single monolithic LLM prompt to make every decision at exorbitant token cost, or hardcoding rigid scripts that fail when anything unexpected occurs.

The **Adaptive Agent Harness 2.0** solves this by embedding **pervasive machine learning classifiers everywhere** throughout the agentic lifecycle:

1. **Skill & Intent Classifier (`SkillClassifier`)**: Routes developer tasks into specialized tool pipelines with a probability distribution. The optional SemIf engine reads local model logits directly and emits typed probabilities without generating or parsing text.
2. **Ambiguity & Risk Classifier (`AmbiguityClassifier`)**: Quantifies task uncertainty via Shannon entropy $H(p)$ and confidence margin. In the interactive SemIf path, high uncertainty can open the clarification modal; destructive actions also remain behind a separate risk check.
3. **Cognitive Complexity & Model Tier Router (`ComplexityRouter`)**: Dynamically routes requests across LLM model tiers:
   - **Fast Tier** (e.g. `google/gemini-2.5-flash-lite`): rapid queries, file reads, git status, typo fixes.
   - **Standard Tier** (`z-ai/glm-5.3-flash`): core coding, refactoring, unit test authoring.
   - **Reasoning Tier** (e.g. `anthropic/claude-sonnet-4`): complex architectures, concurrency/deadlocks, multi-file algorithms.
4. **Tool Verification & Self-Healing Classifier (`VerificationClassifier`)**: Actively inspects tool outputs (bash stdout/stderr, pytest assertions, compiler syntax errors, missing paths) to classify failure modes (`SYNTAX_ERROR`, `TEST_FAILURE`, `FILE_ERROR`, `RUNTIME_ERROR`) and immediately trigger targeted recovery actions (`AUTO_RETRY_SYNTAX_FIX`, `AUTO_RETRY_TEST_FIX`, etc.).
5. **Interactive Textual TUI (`AdaptiveHarnessApp`)**: A full terminal IDE featuring live classifier telemetry gauges, probability bar charts, streaming agent thought logs, and interactive clarification modals.
6. **OpenRouter Protocol & Offline Fallback**: Direct integration with OpenRouter's API (`https://openrouter.ai/api/v1`) and OpenAI-compatible endpoints, paired with an intelligent **Mock LLM engine** for instant offline development without requiring a paid API key.

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
  - `/thinking <none|low|medium|deep|auto>`: Select the reasoning budget.
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
============================= 49 passed in 13.78s ==============================
tests/test_agent_and_tools.py::test_llm_client_mock_mode PASSED          [  2%]
tests/test_agent_and_tools.py::test_run_bash_tool PASSED                 [  4%]
tests/test_agent_and_tools.py::test_file_ops_tools PASSED                [  6%]
tests/test_agent_and_tools.py::test_workspace_tools PASSED               [  8%]
tests/test_agent_and_tools.py::test_clarification_and_schema PASSED      [ 10%]
tests/test_agent_and_tools.py::test_skill_classifier PASSED              [ 12%]
tests/test_agent_and_tools.py::test_ambiguity_classifier PASSED          [ 14%]
tests/test_agent_and_tools.py::test_complexity_router PASSED             [ 16%]
tests/test_agent_and_tools.py::test_verification_classifier PASSED       [ 18%]
tests/test_agent_and_tools.py::test_developer_agent_stream PASSED        [ 20%]
tests/test_agent_and_tools.py::test_tui_app_headless[asyncio] PASSED     [ 22%]
... [algorithmic strategies, benchmarks, calibration, and storage tests]
============================= 49 passed in 13.78s ==============================
```

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
│       ├── cli.py                         # Typer CLI (tui, dev, train, benchmark, etc.)
│       ├── harness/                       # Algorithmic routing harness & verifier
│       ├── models/                        # Domain models, feature extraction, calibration
│       ├── data/                          # Dataset generator and SQLite repository
│       ├── strategies/                    # Specialized algorithmic problem solvers
│       ├── evaluation/                    # Ablation benchmarks, ECE, & matplotlib plots
│       └── dashboard/                     # Rich console output formatting
└── tests/
    ├── test_agent_and_tools.py            # Phase 2 test suite (TUI, agent, classifiers, tools)
    └── test_*.py                          # Strategy, benchmark, and calibration unit tests
```

---

## 9. License

MIT License. Designed and built as an advanced agentic software engineering harness.
