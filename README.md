# Adaptive Agent Harness 2.0

## Current developer agent setup

Install the project into its virtual environment, then run `adaptive-harness dev "git status"` or `adaptive-harness tui`. With no API key or custom task-model endpoint, completions use the offline mock client. Set `OPENROUTER_API_KEY` or pass `--key` for live completions. `--base-url` points the task model at an OpenAI-compatible local server; the classifier backend is configured separately.

Model selection is automatic by default. The complexity router selects fast, standard, or reasoning for each task. `--model MODEL_ID` or `--tier fast|standard|reasoning` fixes the model for every step. In the TUI, `/model MODEL_ID` and `/tier NAME` do the same; `/model auto` restores automatic routing. A forced model appears as `FORCED` in telemetry.

Choose a working directory with `--workspace PATH` or `/workspace PATH`. Conversations and classifier settings save automatically to the configured SQLite database. Use `/sessions` to list them, `/session new [title]` to start fresh, `/session load ID` to resume, or `--session ID` at launch. Put optional `SKILL.md` files under `.harness/skills/NAME/` in the workspace or `~/.config/adaptive-harness/skills/NAME/`; `/skills` lists them and `/skill NAME` toggles one. `dev --skill NAME` enables a skill for a headless task.

The default classifier is local TF-IDF and logistic regression. Select another engine with `--classifier-backend sklearn|ollama|local-slm|onnx|openrouter` and `--classifier-model MODEL`. The TUI equivalent is `/classifier BACKEND [MODEL]`. `--classifier-endpoint URL` sets a custom HTTP endpoint for Ollama or OpenRouter classification. If a classifier request fails, the agent reports the error and uses sklearn for that task.

For a local SLM, start Ollama and pull a small model, such as `ollama pull qwen2.5:1.5b`, then run:

```bash
adaptive-harness tui --classifier-backend ollama --classifier-model qwen2.5:1.5b
```

The Ollama backend calls the local `/api/generate` endpoint with JSON output. For llama.cpp or another local OpenAI-compatible server, select `local-slm` and set `--classifier-endpoint` to its `/v1/chat/completions` URL. The ONNX backend requires `onnxruntime` and `transformers` installed separately and a local model directory containing `model.onnx` and tokenizer files. It scores task and label embeddings by cosine similarity. The OpenRouter classifier uses `OPENROUTER_API_KEY` and a separate, configurable classifier model; it consumes API tokens.

Each task also receives a domain mode (coding, research, science, or audit) and a thinking level (none, low, medium, deep). The selected classifier backend predicts all three heads: skill, domain, and thinking. The domain changes agent guidance and tools offered to the model; audit is read oriented, research can save notes, and science has a restricted arithmetic checker. Set `BRAVE_SEARCH_API_KEY` to enable web search with source URLs in research mode. Python file writes receive automatic AST validation, with syntax errors fed into the verification and recovery loop. The thinking level sets a token estimate in telemetry and sends OpenRouter reasoning effort for supported reasoning models. These are routing signals, not guarantees of a specific token count. The telemetry panel shows the active classifier, its observed latency, domain, thinking level, entropy, margin, skill probabilities, and model selection.

Clarification opens for detected destructive operations or a genuinely missing task target. Numeric keys 1–9 select options, arrows navigate, Enter confirms, Tab reaches the custom instruction field, and Escape cancels. Long text wraps in a scrollable card, and empty custom input cannot approve an action. Entropy and requests for the agent's design judgment do not pause the task. F2 toggles the telemetry panel on wide terminals; narrow terminals hide it automatically. `/output` shows the last tool result up to the 20,000-character output limit.

> **An Autonomous AI Developer Agent Harness powered by OpenRouter LLMs and Pervasive ML Classifiers everywhere — featuring real-time intent routing, middle-of-development clarification dialogs, multi-tier complexity routing, active output verification, and an interactive Terminal User Interface (TUI).**

---

## 1. Overview & Core Philosophy

Modern AI developer agents often suffer from two major extremes: either blindly trusting a single monolithic LLM prompt to make every decision at exorbitant token cost, or hardcoding rigid scripts that fail when anything unexpected occurs.

The **Adaptive Agent Harness 2.0** solves this by embedding **pervasive machine learning classifiers everywhere** throughout the agentic lifecycle:

1. **Skill & Intent Classifier (`SkillClassifier`)**: Rapidly routes developer tasks into specialized tool pipelines (`code_edit`, `run_command`, `search_explore`, `testing`, `ask_clarification`, `general_reasoning`) with full probability distributions.
2. **Ambiguity & Risk Classifier (`AmbiguityClassifier`)**: Quantifies task uncertainty via Shannon entropy $H(p)$, confidence margin, and destructive risk filters. Destructive actions and explicit decision ambiguity trigger clarification; entropy alone remains telemetry.
3. **Cognitive Complexity & Model Tier Router (`ComplexityRouter`)**: Dynamically routes requests across LLM model tiers:
   - **Fast Tier** (e.g. `google/gemini-2.5-flash-lite`): rapid queries, file reads, git status, typo fixes.
   - **Standard Tier** (e.g. `openai/gpt-4o`): core coding, refactoring, unit test authoring.
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
| **`ComplexityRouter`** | Task scope, vocabulary depth, architectural cues | `FAST` (Gemini Flash), `STANDARD` (GPT-4o), `REASONING` (Claude 3.7 Sonnet) | Dynamic model selection minimizing cost and maximizing reasoning depth |
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
  - `/key <OPENROUTER_API_KEY>`: Set or update OpenRouter API key on the fly.
  - `/model <MODEL_ID>`: Switch active LLM (e.g. `anthropic/claude-sonnet-4`).
  - `/tier <fast|standard|reasoning>`: Switch between optimized cost/capability tiers.
  - `/clear`: Clear terminal history log.
  - `/help`: Display available commands.
  - `/exit`: Terminate application.

---

## 4. Developer Tools Suite

The agent is equipped with a workspace-scoped developer toolbelt (`src/adaptive_harness/tools/`):

- **`RunBashTool` (`run_bash`)**: Executes workspace commands with timeouts, output capture, virtualenv path resolution, and dangerous command safety filtering (blocks `rm -rf /`, fork bombs, etc.).
- **`ReadFileTool` (`read_file`)**: Reads workspace files with line numbering and optional `start_line` / `end_line` slicing.
- **`WriteFileTool` (`write_file`)**: Writes new files or overwrites existing files, creating parent directories on demand.
- **`EditFileTool` (`edit_file`)**: Surgically finds and replaces unique blocks of code in an existing file and produces unified diffs (`difflib`).
- **`ListDirectoryTool` (`list_directory`)**: Lists directory contents with sizes and folder indicators, filtering out hidden noise.
- **`SearchFilesTool` (`search_files`)**: Recursive regex grep across workspace files with extension filtering.
- **`RunPytestTool` (`run_pytest`)**: Executes pytest suites, parses structured passed/failed counts, and extracts failure stack traces.
- **`AskUserTool` (`ask_user`)**: Prompts the developer in the middle of development with multiple-choice buttons or free-text answers.

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
