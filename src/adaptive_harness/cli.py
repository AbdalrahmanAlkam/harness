"""Interactive command-line interface for Adaptive Agent Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import os
import signal
import sys
import threading
import typer
from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from adaptive_harness.tui.formatting import format_model_markdown

from adaptive_harness.cli_contract import (
    Ceilings,
    JsonStream,
    exit_code_for,
)
from adaptive_harness.dashboard.report import (
    print_ablation_table,
    print_execution_trace,
    print_failure_analysis,
    print_history_table,
)
from adaptive_harness.data.dataset import build_synthetic_splits
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.evaluation.benchmark import BenchmarkRunner
from adaptive_harness.evaluation.plots import (
    plot_confidence_vs_accuracy,
    plot_confusion_matrix,
    plot_entropy_distribution,
    plot_harness_improvement,
    plot_learning_curve,
    plot_reliability_diagram,
    plot_threshold_tradeoff,
)
from adaptive_harness.harness.harness import Harness
from adaptive_harness.harness.policy import (
    CascadingPolicy,
    ExplorePolicy,
    GreedyPolicy,
    ThresholdPolicy,
)
from adaptive_harness.models.classifier import DEFAULT_MODEL_PATH, TaskClassifier
from adaptive_harness.models.training import retrain_from_experience, train_routing_model

app = typer.Typer(
    name="adaptive-harness",
    help="Adaptive Agent Harness CLI: ML routing brain with verification and fallback recovery.",
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        from adaptive_harness import __version__

        console.print(f"adaptive-harness {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the installed version and exit."),
) -> None:
    """Adaptive Agent Harness.

    Start with `dev "your task"` for a single headless run, or `tui` for the
    interactive terminal interface. Run `research "a topic"` for the autonomous
    research swarm, which verifies its own results before reporting a verdict.
    """


@app.command()
def distill(
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Experience database"),
    output_dir: Path = typer.Option(Path("output/distillation"), "--output", help="Dataset and LoRA scaffold directory"),
    model: str = typer.Option("qwen2.5-0.5b", "--model", help="qwen2.5-0.5b, qwen2.5-1.5b, or smollm2"),
):
    """Export verified agent traces as local instruction-tuning data and LoRA scripts."""
    from adaptive_harness.learning.distill import export_distillation
    try:
        count = export_distillation(db_path, output_dir, model)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Exported {count} verified traces to {output_dir}[/green]")


def _ensure_classifier(model_path: Path = DEFAULT_MODEL_PATH) -> TaskClassifier:
    """Loads existing classifier or automatically trains a baseline if missing."""
    if not model_path.exists():
        console.print(f"[yellow]No trained model found at {model_path}. Training baseline model...[/yellow]")
        clf, metrics = train_routing_model(samples_per_class=600, save_path=model_path)
        console.print(f"[green]Trained baseline model with test accuracy: {metrics['accuracy']*100:.1f}%[/green]")
        return clf
    return TaskClassifier.load(model_path)


@app.command()
def run(
    task_text: Optional[str] = typer.Argument(None, help="Task query to execute (omit for interactive REPL)"),
    policy_name: str = typer.Option("threshold", "--policy", "-p", help="Routing policy: threshold, greedy, explore, cascading"),
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", "-m", help="Path to trained model"),
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Path to SQLite experience database"),
):
    """Executes a single task or starts an interactive routing session."""
    clf = _ensure_classifier(model_path)
    repo = ExperienceRepository(db_path)

    policy = ThresholdPolicy()
    if policy_name.lower() == "greedy":
        policy = GreedyPolicy()
    elif policy_name.lower() == "explore":
        policy = ExplorePolicy()
    elif policy_name.lower() == "cascading":
        policy = CascadingPolicy()

    harness = Harness(classifier=clf, policy=policy, repository=repo)

    if task_text:
        trace = harness.run(task_text)
        print_execution_trace(trace)
        # Exit non-zero when the task was not solved, so a CI job or a shell
        # script can detect failure. `run` always returning 0 meant an unsolved
        # task was indistinguishable from a solved one.
        if not getattr(trace, "success", True):
            raise typer.Exit(1)
        return

    # Interactive REPL mode
    console.print("\n[bold cyan]=== Adaptive Agent Harness Interactive REPL ===[/bold cyan]")
    console.print("Type any task (arithmetic, equations, algorithms, text statistics, or general questions).")
    console.print("Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to terminate.\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]task>[/bold green]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Exiting interactive harness.[/dim]")
                break

            trace = harness.run(user_input)
            print_execution_trace(trace)
            console.print()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Terminating session.[/dim]")
            break


def _start_operator_reader(agent) -> Optional[threading.Thread]:
    """Let an operator steer a headless run by typing, in the same terminal.

    Reads stdin on a daemon thread that only ever calls
    ``queue_operator_message``; the run itself stays single-threaded on the main
    thread. Nothing is read when stdin is not a TTY, so piped invocations and CI
    are unaffected.
    """
    if not sys.stdin.isatty():
        return None
    console.print("[dim]type a note + Enter to steer the start of the next step · "
                  "prefix it with ! to stop after the current step instead[/dim]")

    def pump() -> None:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            interrupt = text.startswith("!")
            body = text[1:].strip() if interrupt else text
            depth = agent.queue_operator_message(body, kind="interrupt" if interrupt else "steer")
            console.print(f"[cyan]⏳ queued ({depth} pending): {body}[/cyan]")

    thread = threading.Thread(target=pump, daemon=True, name="operator-reader")
    thread.start()
    return thread


@app.command()
def tui(
    api_key: Optional[str] = typer.Option(None, "--key", "-k", help="OpenRouter or OpenAI API key"),
    provider: Optional[str] = typer.Option(None, "--provider", help="openrouter, anthropic, openai, deepseek, google, groq, local"),
    backup_provider: Optional[list[str]] = typer.Option(None, "--backup-provider", help="Fail over on 429/5xx; repeat in priority order"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible task model endpoint"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Default model ID (e.g. anthropic/claude-sonnet-4)"),
    tier: Optional[str] = typer.Option(None, "--tier", help="Force model tier: fast, standard, reasoning"),
    mode: str = typer.Option("auto", "--mode", help="Operational mode: coding, research, science, plan, security, auto"),
    thinking: str = typer.Option("auto", "--thinking", help="Model effort: auto, low, medium, high, xhigh, max (deep = high)"),
    secondary_model: Optional[str] = typer.Option(
        None, "--secondary-model",
        help="Cheap model that compresses low-value tool output (default: the primary model)"),
    safety: Optional[str] = typer.Option(None, "--safety", help="Interaction profile: turbo, balanced, cautious, strict (default turbo)"),
    quality_gate: bool = typer.Option(
        False, "--quality-gate",
        help="Check the final answer against the tool calls that were actually "
             "made, and send it back if it claims work nothing evidences. Reports "
             "either way; this makes it load-bearing."),
    step_policy: str = typer.Option("classifier", "--step-policy", help="Tool-step limits: classifier (default; stops circling loops), fixed (hardcoded per-thinking budgets), unbounded"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", min=1, help="Explicit tool-step cap overriding the step policy"),
    classifier_backend: str = typer.Option("auto", "--classifier-backend", "--classifier-engine", help="auto, semif, sklearn, ollama, local-slm, onnx, openrouter"),
    classifier_model: Optional[str] = typer.Option(None, "--classifier-model", help="Classifier model ID or ONNX directory"),
    classifier_endpoint: Optional[str] = typer.Option(None, "--classifier-endpoint", help="Local or OpenRouter classifier endpoint"),
    overseer_model: Optional[str] = typer.Option(None, "--overseer-model", help="Local ONNX embedding model directory for runtime oversight"),
    semif_model: Optional[str] = typer.Option(None, "--semif-model", help="Cached HuggingFace model ID or local checkpoint path"),
    semif_device: str = typer.Option("auto", "--semif-device", help="SemIf device: auto, cpu, cuda, or mps"),
    semif_4bit: bool = typer.Option(False, "--semif-4bit", help="Load SemIf in 4-bit on CUDA (requires bitsandbytes)"),
    semif_temperature: float = typer.Option(1.0, "--semif-temperature", min=0.01, help="SemIf probability temperature"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Working directory for agent tools (default: launch directory)"),
    session: Optional[str] = typer.Option(None, "--session", help="Resume an existing TUI session ID"),
    skill: Optional[list[str]] = typer.Option(None, "--skill", help="Force a built-in or custom skill (repeatable)"),
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Path to experience database"),
):
    """Launch the interactive terminal interface. Requires a TTY."""
    from adaptive_harness.tui.app import AdaptiveHarnessApp
    from adaptive_harness.llm.client import MODEL_TIERS
    from adaptive_harness.llm.providers import PROVIDER_TIERS
    from adaptive_harness.classifiers.domain_classifier import parse_domain_mode
    from adaptive_harness.classifiers.thinking_classifier import parse_thinking_level

    if tier and tier not in MODEL_TIERS:
        raise typer.BadParameter("Choose fast, standard, or reasoning", param_hint="--tier")
    if safety is not None and safety not in {"turbo", "balanced", "cautious", "strict"}:
        raise typer.BadParameter("Choose turbo, balanced, cautious, or strict", param_hint="--safety")
    from adaptive_harness.agent.agent import STEP_POLICIES
    if step_policy not in STEP_POLICIES:
        raise typer.BadParameter("Choose classifier, fixed, or unbounded", param_hint="--step-policy")
    try:
        parse_domain_mode(mode)
        parse_thinking_level(thinking)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        tui_app = AdaptiveHarnessApp(
            api_key=api_key,
            provider=provider,
            backup_providers=tuple(backup_provider or ()),
            base_url=base_url,
            default_model=("auto" if model and model.lower() == "auto" else model or
                (PROVIDER_TIERS.get(provider or "openrouter", MODEL_TIERS)[tier] if tier in MODEL_TIERS else None)),
            mode=mode,
            thinking=thinking,
            secondary_model=secondary_model,
            safety=safety,
            classifier_backend="semif" if semif_model else classifier_backend,
            classifier_model=semif_model or classifier_model,
            classifier_endpoint=classifier_endpoint,
            step_policy=step_policy,
            max_steps=max_steps,
            overseer_model=overseer_model,
            semif_device=semif_device,
            semif_4bit=semif_4bit,
            semif_temperature=semif_temperature,
            workspace_root=str(Path(workspace or Path.cwd()).expanduser().resolve()),
            session_id=session,
            db_path=db_path,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if skill:
        from adaptive_harness.agent.skills import SkillCatalog
        catalog = SkillCatalog(tui_app.workspace_root)
        for name in skill:
            try:
                tui_app.agent.active_skills[name] = catalog.read(name)
            except (ValueError, OSError) as exc:
                raise typer.BadParameter(str(exc), param_hint="--skill") from exc
    # A full-screen Textual app on a non-TTY (a pipe, cron, CI, `nohup`) waits
    # forever for a keypress that can never arrive. Fail with the one command
    # that does work headlessly instead of hanging.
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        console.print("[bold red]The TUI needs an interactive terminal.[/bold red]")
        console.print("It cannot run piped, redirected, or under a process manager.")
        console.print("For a headless run use: [bold]adaptive-harness dev \"your task\"[/bold]")
        raise typer.Exit(2)
    tui_app.run()


@app.command()
def dev(
    task: str = typer.Argument(..., help="Software engineering task to execute"),
    offline: bool = typer.Option(False, "--offline", help="Use the offline mock engine without a network request"),
    api_key: Optional[str] = typer.Option(None, "--key", "-k", help="OpenRouter or OpenAI API key"),
    provider: Optional[str] = typer.Option(None, "--provider", help="openrouter, anthropic, openai, deepseek, google, groq, local"),
    backup_provider: Optional[list[str]] = typer.Option(None, "--backup-provider", help="Fail over on 429/5xx; repeat in priority order"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible task model endpoint"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Default model ID"),
    tier: Optional[str] = typer.Option(None, "--tier", help="Force model tier: fast, standard, reasoning"),
    mode: str = typer.Option("auto", "--mode", help="Operational mode: coding, research, science, plan, security, auto"),
    thinking: str = typer.Option("auto", "--thinking", help="Model effort: auto, low, medium, high, xhigh, max (deep = high)"),
    safety: Optional[str] = typer.Option(None, "--safety", help="Interaction profile: turbo, balanced, cautious, strict (default turbo)"),
    json_output: bool = typer.Option(
        False, "--json",
        help="Emit one JSON object per event, and a summary object at the end. "
             "For scripting; the human-readable output is suppressed."),
    max_cost: Optional[float] = typer.Option(
        None, "--max-cost", min=0.0,
        help="Stop the run once this much has been spent, in US dollars. The stop "
             "is attributed: the exit code is 3, not 1."),
    max_turns: Optional[int] = typer.Option(
        None, "--max-turns", min=1,
        help="Stop the run after this many model turns. Exit code 3 when reached."),
    quality_gate: bool = typer.Option(
        False, "--quality-gate",
        help="Check the final answer against the tool calls that were actually "
             "made, and send it back if it claims work nothing evidences. Reports "
             "either way; this makes it load-bearing."),
    step_policy: str = typer.Option("classifier", "--step-policy", help="Tool-step limits: classifier (default; stops circling loops), fixed (hardcoded per-thinking budgets), unbounded"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", min=1, help="Explicit tool-step cap overriding the step policy"),
    swarm: bool = typer.Option(False, "--swarm", help="Expose delegate_subagent to the coordinator agent"),
    research_topic: Optional[str] = typer.Option(
        None, "--research", help="Attach a research swarm for TOPIC and expose spawn_subagent, "
                                 "scale_division, verify_proofs, and run_experiments"),
    research_root: Path = typer.Option(Path("research"), "--research-root",
                                       help="Artifact tree root for --research"),
    classifier_backend: str = typer.Option("auto", "--classifier-backend", "--classifier-engine", help="auto, semif, sklearn, ollama, local-slm, onnx, openrouter"),
    classifier_model: Optional[str] = typer.Option(None, "--classifier-model", help="Classifier model ID or ONNX directory"),
    classifier_endpoint: Optional[str] = typer.Option(None, "--classifier-endpoint", help="Classifier endpoint"),
    overseer_model: Optional[str] = typer.Option(None, "--overseer-model", help="Local ONNX embedding model directory for runtime oversight"),
    semif_model: Optional[str] = typer.Option(None, "--semif-model", help="Cached HuggingFace model ID or local checkpoint path"),
    semif_device: str = typer.Option("auto", "--semif-device", help="SemIf device: auto, cpu, cuda, or mps"),
    semif_4bit: bool = typer.Option(False, "--semif-4bit", help="Load SemIf in 4-bit on CUDA (requires bitsandbytes)"),
    semif_temperature: float = typer.Option(1.0, "--semif-temperature", min=0.01, help="SemIf probability temperature"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Working directory for agent tools (default: launch directory)"),
    skill: Optional[list[str]] = typer.Option(None, "--skill", help="Enable a named workspace or user skill"),
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Path to experience database"),
):
    """Run one software-engineering task end to end, and exit non-zero if it is not completed."""
    from adaptive_harness.agent.agent import DeveloperAgent
    from adaptive_harness.llm.client import LLMClient
    from adaptive_harness.llm.client import MODEL_TIERS
    from adaptive_harness.llm.providers import PROVIDERS, PROVIDER_TIERS, provider_for_url
    from adaptive_harness.data.credentials import CredentialsManager
    from adaptive_harness.classifiers.engine import create_backend
    from adaptive_harness.agent.skills import SkillCatalog
    from adaptive_harness.classifiers.domain_classifier import parse_domain_mode
    from adaptive_harness.classifiers.thinking_classifier import parse_thinking_level

    if tier and tier not in MODEL_TIERS:
        raise typer.BadParameter("Choose fast, standard, or reasoning", param_hint="--tier")
    if provider is not None and provider not in PROVIDERS:
        raise typer.BadParameter("Unknown provider", param_hint="--provider")
    selected_provider = provider or provider_for_url(base_url or os.environ.get("OPENROUTER_BASE_URL", ""))
    if selected_provider not in PROVIDERS:
        selected_provider = "openrouter"
    if any(name not in PROVIDERS for name in (backup_provider or [])):
        raise typer.BadParameter("Unknown backup provider", param_hint="--backup-provider")
    if safety is not None and safety not in {"turbo", "balanced", "cautious", "strict"}:
        raise typer.BadParameter("Choose turbo, balanced, cautious, or strict", param_hint="--safety")
    from adaptive_harness.agent.agent import STEP_POLICIES
    if step_policy not in STEP_POLICIES:
        raise typer.BadParameter("Choose classifier, fixed, or unbounded", param_hint="--step-policy")
    try:
        selected_mode = parse_domain_mode(mode)
        selected_thinking = parse_thinking_level(thinking)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    provider_tiers = PROVIDER_TIERS.get(selected_provider, MODEL_TIERS)
    selected_model = (None if model and model.lower() == "auto" else model) or (provider_tiers[tier] if tier in MODEL_TIERS else None)
    if selected_thinking is not None and (not model or model.lower() != "auto"):
        from adaptive_harness.classifiers.thinking_classifier import ThinkingLevel, effort_for_level
        from adaptive_harness.llm.effort import supported_efforts
        active_model = selected_model or PROVIDERS[selected_provider].default_model
        supported = supported_efforts(active_model, selected_provider)
        effort = effort_for_level(selected_thinking)
        if (selected_thinking is ThinkingLevel.NONE and supported) or (effort and effort not in supported):
            raise typer.BadParameter(f"{active_model} supports reasoning efforts: "
                                     f"{', '.join(supported) or 'auto only'}", param_hint="--thinking")
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    from adaptive_harness.data.config import ConfigManager
    try:
        saved_keys = CredentialsManager().load()
    except (OSError, ValueError):
        saved_keys = {}
    if "openrouter" not in saved_keys:
        legacy_key = ConfigManager().load().get("api_key")
        if legacy_key:
            saved_keys["openrouter"] = legacy_key
    resolved_key = api_key or os.environ.get(PROVIDERS[selected_provider].env_key or "") or saved_keys.get(selected_provider)
    client = LLMClient(api_key=resolved_key, base_url=base_url, default_model=selected_model,
                       provider=selected_provider, provider_keys=saved_keys,
                       backup_providers=tuple(name for name in (backup_provider or ()) if name != selected_provider),
                       force_mock=offline)
    repo = ExperienceRepository(db_path)
    if not workspace.is_dir():
        raise typer.BadParameter(f"Workspace directory does not exist: {workspace}", param_hint="--workspace")
    try:
        backend = create_backend("semif" if semif_model else classifier_backend,
            semif_model or classifier_model, classifier_endpoint, api_key=api_key, device=semif_device,
            load_in_4bit=semif_4bit, temperature=semif_temperature)
        overseer_backend = create_backend("onnx", overseer_model) if overseer_model else None
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--classifier-backend") from exc
    agent = DeveloperAgent(llm_client=client, repository=repo, workspace_root=str(workspace),
                           explicit_model="auto" if model and model.lower() == "auto" else
                                          selected_model or client.default_model,
                           classifier_backend=backend, overseer_backend=overseer_backend,
                           forced_mode=selected_mode, forced_thinking=selected_thinking,
                           safety_profile=safety or "turbo", swarm_enabled=swarm,
                           step_policy=step_policy, max_steps=max_steps,
                           quality_gate=quality_gate)
    if research_topic:
        research_swarm = agent.enable_research(research_topic, root=research_root)
        console.print(f"[dim]Research swarm attached: {research_swarm.workspace.root} "
                      f"(mode: {research_swarm.config.author_mode})[/dim]")
    catalog = SkillCatalog(workspace)
    for name in skill or []:
        try:
            agent.active_skills[name] = catalog.read(name)
        except (ValueError, OSError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--skill") from exc

    console.print(f"\n[bold cyan]=== Adaptive Developer Agent Task ===[/bold cyan]")
    console.print(f"Task: [bold white]\"{escape(task)}\"[/bold white]")

    last_agent_content = ""
    task_succeeded = False
    task_stop_reason = ""
    json_stream = None
    if json_output:
        # Straight to stdout, never through Rich: Rich wraps at the terminal
        # width, which turns one JSON object per line into something a parser
        # cannot read. `--json` is a machine contract.
        json_stream = JsonStream(lambda line: (sys.stdout.write(line + "\n"),
                                               sys.stdout.flush()))
    if max_cost is not None or max_turns is not None:
        # Independent of --json: a ceiling is a ceiling whether or not anyone is
        # parsing the output.
        agent.ceilings = Ceilings(max_cost_usd=max_cost, max_turns=max_turns)
    _start_operator_reader(agent)
    for event in agent.run_stream_with_followup(task):
        if json_stream is not None:
            json_stream.emit(event)
        et = event.event_type
        p = event.payload
        if et == "skill_classification":
            console.print(f"  [cyan]Skill Classifier ({p['backend']} · {p['latency_ms']:.1f} ms):[/cyan] {p['primary_skill']} ({p['confidence']*100:.1f}%)")
        elif et == "specialized_skill" and p["skills"]:
            console.print(f"  [cyan]Active skill:[/cyan] {escape(' → '.join(skill['title'] for skill in p['skills']))} ({p['confidence']:.0%}, {p['selection']})")
        elif et == "skill_verification" and p["missing"]:
            console.print(f"  [yellow]Skill checks pending ({escape(p['skill'])}): {escape(', '.join(p['missing']))}[/yellow]")
        elif et == "classifier_loading":
            console.print(f"  [cyan]Loading local SemIf model {p['model']} for its first decision…[/cyan]")
        elif et == "classifier_error":
            console.print(f"  [yellow]Classifier {p['backend']} unavailable: {escape(p['error'])}; using sklearn[/yellow]")
        elif et == "ambiguity_assessment":
            color = "green" if p["risk_level"] == "low" else "yellow"
            console.print(f"  [{color}]Ambiguity & Risk:[/{color}] H={p['entropy']:.2f} bits | Risk={p['risk_level'].upper()} | Ask={p['should_ask_question']}")
        elif et == "model_routing":
            console.print(f"  [magenta]Model Tier:[/magenta] [{p['tier'].upper()}] -> {p['model']}")
        elif et == "domain_mode":
            console.print(f"  [cyan]Domain:[/cyan] {p['mode'].upper()} ({p['selection']})")
        elif et == "thinking_budget":
            console.print(f"  [yellow]Thinking:[/yellow] {p['level'].upper()} ({p['tokens']:,} token budget, {p['selection']})")
        elif et == "agent_stage":
            stage = p["stage"]
            label = {"thinking": "Thinking", "model_processing": "Processing with model", "generating": "Generating",
                     "tool_running": f"Running {p.get('tool', '')}",
                     "verifying": f"Verifying {p.get('tool', '')}"}.get(stage, stage)
            console.print(f"  [dim]◦ {escape(label)}[/dim]")
        elif et == "context_status" and p.get("compacted"):
            console.print(f"  [cyan]Context compacted:[/cyan] ~{p['compacted_tokens']:,} tokens saved "
                          f"({p['used_tokens']:,}/{p['capacity']:,})")
        elif et == "overseer" and p.get("directive"):
            console.print(f"  [yellow]Overseer: {escape(p['state'])} · correction injected[/yellow]")
        elif et == "step_policy":
            cap = "unbounded" if p["max_steps"] is None else f"{p['max_steps']} steps"
            console.print(f"  [cyan]Step policy:[/cyan] {p['policy']} · limit: {cap} ({p['limit_source']}) · "
                          f"classifier supervision {'on' if p['classifier_supervision'] else 'off'}")
        elif et == "system_prompt":
            ingested = ", ".join(f"{key}={value}" for key, value in (p.get("ingested") or {}).items()
                                 if value not in (False, None, "", []))
            console.print(f"  [dim]Ingested system prompt ({len(p['content']):,} chars)"
                          f"{': ' + escape(ingested) if ingested else ''}:[/dim]")
            console.print(Text(p["content"], style="dim"))
        elif et == "prompt_injection":
            label = {"runtime_overseer": "classifier · runtime overseer", "claim_check": "classifier · claim check",
                     "harness": "harness"}.get(p["source"], p["source"])
            console.print(f"  [bold yellow]Prompt injected · {escape(label)}"
                          f"{(' · ' + escape(p['state'])) if p.get('state') else ''}:[/bold yellow]")
            console.print(Text(p["content"], style="yellow"))
        elif et == "operator_queued":
            console.print(f"  [cyan]⏳ operator queued ({p.get('pending', 1)} pending):[/cyan] "
                          f"{escape(p.get('content', ''))}")
        elif et == "operator_message":
            console.print(f"  [bold cyan]⚡ operator:[/bold cyan] {escape(p.get('raw') or '')}")
            note = ("run stops after this step" if p.get("kind") == "interrupt"
                    else "the run had already finished; kept as context" if p.get("late")
                    else f"delivered at the start of step {p.get('step')}")
            console.print(f"  [dim]{escape(note)} · waited {p.get('wait_ms', 0)} ms[/dim]")
        elif et == "requirements":
            items = p.get("requirements", [])
            if items:
                console.print(f"  [dim]{len(items)} obligation(s): "
                              f"{escape(', '.join(i.get('text', '') for i in items))}[/dim]")
        elif et == "context_budget":
            admitted, rejected = p.get("admitted", []), p.get("rejected", [])
            console.print(f"  [dim]context: {p.get('used', 0):,}/{p.get('available', 0):,} "
                          f"tokens · {len(admitted)} admitted, {len(rejected)} deferred[/dim]")
        elif et == "quality_gate":
            colour = {"accept": "green", "revise": "yellow", "reject": "red"}.get(
                p.get("verdict", ""), "yellow")
            console.print(f"  [bold {colour}]quality gate: {escape(str(p.get('verdict', '')).upper())}"
                          f"[/bold {colour}] — {escape(p.get('reason', ''))}")
            for item in p.get("adjudications", []):
                if item.get("status") != "satisfied":
                    console.print(f"    [yellow]{escape(item.get('id', ''))} "
                                  f"{escape(item.get('status', ''))}: "
                                  f"{escape(item.get('reason', ''))}[/yellow]")
            for line in p.get("contradictions", []):
                console.print(f"    [red]contradiction: {escape(line)}[/red]")
        elif et == "provider_failover":
            console.print(f"  [yellow]Provider failover: {escape(p['from'])} → {escape(p['to'])} "
                          f"({escape(p['model'])})[/yellow]")
        elif et == "llm_error":
            console.print(f"[red]Model error: {escape(p['message'])}[/red]")
        elif et == "llm_notice":
            console.print(f"[yellow]{escape(p['message'])}[/yellow]")
        elif et == "storage_error":
            console.print(f"[yellow]{escape(p['error'])}[/yellow]")
        elif et == "thought":
            last_agent_content = p["content"]
            console.print("\n[bold magenta]Agent:[/bold magenta]")
            console.print(Markdown(format_model_markdown(p["content"])))
        elif et == "memory_hit":
            console.print("  [bold green]⚡ Verified memory answer reused (0 API tokens)[/bold green]")
        elif et == "clarification_memory_hit":
            console.print("  [cyan]Using a saved preference for this question.[/cyan]")
        elif et == "tool_call":
            console.print(f"  [bold yellow]Tool Call:[/bold yellow] [cyan]{p['name']}[/cyan] [dim]{escape(str(p['arguments']))}[/dim]")
        elif et == "tool_result":
            status_col = "green" if p["success"] else "red"
            console.print(f"  [{status_col}]Tool Result ({p['time_ms']} ms):[/{status_col}]")
            console.print(Text(format_model_markdown(p['output'][:200], plain=True)))
        elif et == "verification":
            badge_col = "green" if p["status"] == "SUCCESS" else "red bold"
            console.print(f"  [dim]Verification Classifier: [{badge_col}]{p['status']}[/{badge_col}] -> Action: {p['action']}[/dim]")
        elif et == "response":
            task_succeeded = bool(p.get("success"))
            task_stop_reason = str(p.get("stop_reason") or "")
            if p.get("content") and p["content"] != last_agent_content:
                console.print("\n[bold magenta]Agent:[/bold magenta]")
                console.print(Markdown(format_model_markdown(p["content"])))
            status = "✓ Completed" if p.get("success", True) else {
                "classifier_stop": "Stopped by the classifier: no further progress expected",
                "overseer_impasse": "Stopped at an overseer impasse",
                "step_limit": "Stopped at the tool-step limit",
                "verification_failed": "Stopped: needs another verification pass",
                "skill_verification_failed": "Stopped: skill verification checks unmet",
                "missing_file_changes": "Stopped: required file changes were not made",
                "provider_error": "Stopped: provider request failed",
            }.get(p.get("stop_reason"), "Stopped before completion")
            color = "green" if p.get("success", True) else "yellow"
            console.print(f"\n[bold {color}]{status} in {p['total_time_ms']} ms ({p['steps']} steps)[/bold {color}]\n")

    # A script's only signal is the exit status, so it has to carry a meaning.
    # See cli_contract for the vocabulary.
    if json_stream is not None:
        # The summary is part of the machine contract, so it goes out the same
        # unwrapped way the events did.
        sys.stdout.write(json.dumps(json_stream.summary(), ensure_ascii=False,
                                   default=str) + "\n")
        sys.stdout.flush()
    # Exit 4 means "something the run needed was not available", which is
    # usually fixable by configuration. Detect it from the stop reason rather
    # Exit 4 means "something the run needed was not available", which is
    # usually fixable by configuration rather than by changing the task. That is
    # specifically *a missing credential for a live provider* -- not a model that
    # ran and declined the work, which is the agent failing (exit 1), and not an
    # explicitly offline run, which asked for the mock and got it.
    unavailable = bool(
        not offline
        and getattr(agent.llm_client, "is_mock", False)
        and task_stop_reason in {"provider_error", "no_credential", ""}
    )
    raise typer.Exit(code=exit_code_for(task_stop_reason, success=task_succeeded,
                                        unavailable=unavailable))


#: Plugin lifecycle. A plugin is a directory under one of the discovery roots,
#: so these commands take a path and operate on it directly.
from adaptive_harness.plugins.registry import OFFICIAL as _OFFICIAL

plugin_app = typer.Typer(
    help="Create, validate, and package plugins.",
)
app.add_typer(plugin_app, name="plugin")


@plugin_app.command("init")
def plugin_init(
    name: str = typer.Argument(..., help="Plugin name, e.g. csv-inspector"),
    description: str = typer.Option("", "--description", help="One line about what it does"),
    destination: Path = typer.Option(Path("."), "--dest", help="Where to create it (a plugin root)"),
):
    """Scaffold a plugin that passes `validate` with no edits.

    Creates a manifest, a module, a README, and a test, in a directory you can
    copy into your plugins folder.
    """
    from adaptive_harness.plugins.lifecycle import scaffold, validate

    if not name.replace("-", "_").isidentifier():
        console.print(f"[red]{name!r} is not a usable plugin name "
                      f"(letters, digits, - and _ only).[/red]")
        raise typer.Exit(2)
    path = scaffold(destination, name=name, description=description)
    report = validate(path)
    console.print(f"[green]Created[/green] {path}")
    console.print(report.render())
    if not report.ok:
        raise typer.Exit(1)


@plugin_app.command("validate")
def plugin_validate(
    path: Path = typer.Argument(..., help="A plugin directory, or its manifest"),
):
    """Check a plugin's manifest, permissions, and handlers — without running it.

    The module is parsed, not imported, so validating something you have not
    decided to trust does not execute it.
    """
    from adaptive_harness.plugins.lifecycle import validate

    report = validate(path)
    console.print(report.render())
    raise typer.Exit(0 if report.ok else 1)


@plugin_app.command("pack")
def plugin_pack(
    path: Path = typer.Argument(..., help="A plugin directory"),
    destination: Path = typer.Option(Path("dist"), "--out", help="Where to write the tarball"),
):
    """Bundle a validated plugin into a distributable tarball."""
    from adaptive_harness.plugins.lifecycle import pack, validate

    try:
        archive = pack(path, destination)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Packed[/green] {archive}")


@plugin_app.command("install")
def plugin_install(
    name: str = typer.Argument(..., help="Official plugin name, or a path to a plugin directory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an installed plugin of the same name"),
):
    """Install a plugin. The core ships with none, so this is how capabilities arrive.

    An official plugin name is looked up in the bundled `plugins/` directory that
    ships beside the harness; any other argument is treated as a path. The
    manifest is validated first, and validation parses the module rather than
    importing it -- deciding whether to trust something and running it are
    different acts.
    """
    from adaptive_harness.plugins.registry import InstallError, Registry, find_official

    registry = Registry()
    candidate = Path(name).expanduser()
    if not candidate.is_dir():
        found = find_official(name, _official_plugin_paths())
        if found is None:
            console.print(f"[red]No official plugin named {name!r}, and no such directory.[/red]")
            console.print("[dim]Official plugins:[/dim] "
                          + ", ".join(sorted(item["name"] for item in _OFFICIAL)))
            raise typer.Exit(1)
        candidate = found
    try:
        installed = registry.install(candidate, overwrite=overwrite)
    except InstallError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Installed[/green] {installed.name}")
    console.print("[dim]Restart the harness, or it will load on next start.[/dim]")


@plugin_app.command("uninstall")
def plugin_uninstall(name: str = typer.Argument(..., help="Installed plugin name")):
    """Remove an installed plugin."""
    from adaptive_harness.plugins.registry import Registry

    if Registry().uninstall(name):
        console.print(f"[green]Removed[/green] {name}")
        return
    console.print(f"[yellow]{name} is not installed.[/yellow]")
    raise typer.Exit(1)


@plugin_app.command("available")
def plugin_available():
    """List the official plugins and why each one exists.

    Nothing here is installed by default. The harness is a working agent with
    no extras; a capability is something you choose, not something you inherit.
    """
    for item in _OFFICIAL:
        console.print(f"  [bold]{item['name']}[/bold] — {item['summary']}")
        console.print(f"    [dim]{item['why']}[/dim]")
    console.print("[dim]Install one with: adaptive-harness plugin install <name>[/dim]")


def _official_plugin_paths() -> list[Path]:
    """Where the bundled official plugins live, if this checkout has them.

    They are outside the installed package on purpose -- the core ships with no
    plugins -- so a pip install finds none, and a checkout offers them for
    anyone who wants to try one.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "plugins"
        if candidate.is_dir() and any(candidate.glob("*/*.plugin.json")):
            return [candidate]
    return []


@plugin_app.command("list")
def plugin_list(
    with_project_plugins: bool = typer.Option(False, "--with-project-plugins",
                                              help="Also load project-supplied plugins"),
):
    """List every discovered plugin and what it contributes."""
    from adaptive_harness.plugins.host import PluginHost

    host = PluginHost(project_root=Path.cwd(),
                      allow_project_plugins=with_project_plugins)
    found = host.discover()
    if not found:
        console.print("[dim]No plugins found.[/dim]")
        return
    for plugin in sorted(found, key=lambda item: item.name):
        mark = "[green]●[/green]" if plugin.ok else "[red]✗[/red]"
        console.print(f"{mark} [bold]{plugin.name}[/bold] {plugin.version} "
                      f"— {plugin.description or 'no description'}")
        if plugin.ok:
            console.print(f"  [dim]permissions:[/dim] "
                          f"{', '.join(sorted(plugin.permissions)) or 'none'}")
        else:
            console.print(f"  [red]{escape(plugin.error)}[/red]")


@plugin_app.command("info")
def plugin_info(name: str = typer.Argument(..., help="Plugin name")):
    """Show one plugin in detail: what it provides and what it can do."""
    from adaptive_harness.plugins.host import PluginHost

    host = PluginHost(project_root=Path.cwd(), allow_project_plugins=True)
    found = [p for p in host.discover() if p.name == name]
    if not found:
        console.print(f"[red]No plugin named {name!r}.[/red]")
        raise typer.Exit(1)
    plugin = found[0]
    if not plugin.ok:
        console.print(f"[red]{plugin.name} failed to load:[/red] {escape(plugin.error)}")
        raise typer.Exit(1)
    console.print(f"[bold]{plugin.name}[/bold] {plugin.version}")
    console.print(f"  {plugin.description or 'no description'}")
    if plugin.author:
        console.print(f"  [dim]author:[/dim] {plugin.author}")
    console.print(f"  [dim]permissions:[/dim] "
                  f"{', '.join(sorted(plugin.permissions)) or 'none'}")
    if plugin.tools:
        console.print("  [bold]tools:[/bold]")
        for tool in plugin.tools:
            console.print(f"    {tool.name} [{tool.risk}] — {tool.description}")
    if plugin.skills:
        console.print("  [bold]skills:[/bold] " + ", ".join(s.name for s in plugin.skills))
    if plugin.settings:
        console.print("  [bold]settings:[/bold] " + ", ".join(s.key for s in plugin.settings))
    if plugin.commands:
        console.print("  [bold]commands:[/bold] " + ", ".join(sorted(plugin.commands)))
    if plugin.subagents:
        console.print("  [bold]subagents:[/bold] " + ", ".join(s.name for s in plugin.subagents))
    if plugin.mcp_servers:
        console.print("  [bold]mcp servers:[/bold] " + ", ".join(s.name for s in plugin.mcp_servers))
    if plugin.context:
        console.print("  [bold]context:[/bold] " + ", ".join(f.source for f in plugin.context))
    active = [name for name, hook in plugin.hooks.__dict__.items() if hook]
    if active:
        console.print("  [bold]hooks:[/bold] " + ", ".join(sorted(active)))


@app.command("memory")
def memory_cmd(
    action: str = typer.Argument("list", help="list, accept ID, forget ID, or clear"),
    memory_id: str = typer.Argument("", help="Memory id, for accept or forget"),
    add: str = typer.Option("", "--add", help="Propose a memory and show what would be stored"),
):
    """Inspect and manage what the harness remembers about this project.

    A memory the model proposed never takes effect on its own. It is shown as a
    one-line diff and applies only when you accept it, which is the only way a
    model can end up changing what it is told in a later session.
    """
    from adaptive_harness.agent.memory import MemoryStore

    store = MemoryStore()
    try:
        if add:
            proposal = store.propose(add)
            console.print(f"[bold]Proposed[/bold] {proposal.preview}")
            console.print(f"[dim]id: {proposal.memory.id} · {proposal.reason}[/dim]")
            console.print("[dim]It is not in effect. "
                          "Accept it with: adaptive-harness memory accept <id>[/dim]")
            return
        if action == "accept":
            memory = store.accept(memory_id)
            if memory is None:
                console.print(f"[yellow]No pending memory with id {memory_id!r}.[/yellow]")
                raise typer.Exit(1)
            console.print(f"[green]Remembered:[/green] {memory.text}")
            return
        if action == "forget":
            if not store.forget(memory_id):
                console.print(f"[yellow]No memory with id {memory_id!r}.[/yellow]")
                raise typer.Exit(1)
            console.print("[green]Forgotten.[/green]")
            return
        if action == "clear":
            console.print(f"[green]Cleared {store.clear()} memory(ies).[/green]")
            return
        console.print(store.describe())
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from exc


@app.command("plugins")
def plugins_cmd(
    workspace: Optional[Path] = typer.Option(None, "--workspace", help="Project root to discover plugins in"),
    with_project_plugins: bool = typer.Option(
        False, "--with-project-plugins",
        help="Also load plugins from <project>/.harness/plugins. These run code from "
             "the repository, so they are off unless you ask."),
):
    """List installed plugins, what each contributes, and the permissions it holds.

    Plugins extend the harness without editing it: a tool the model can call, a
    skill, a setting, a prompt override, a command. Each one declares what it
    wants permission to do, and anything it did not declare is refused.

    A plugin is third-party code that runs with your privileges. Read the
    permission line before trusting one.
    """
    from adaptive_harness.plugins.host import PluginHost

    host = PluginHost(project_root=workspace or Path.cwd(),
                      allow_project_plugins=with_project_plugins)
    found = host.discover()
    console.print(f"[bold cyan]Plugins[/bold cyan] ({len(found)} found)")
    if not found:
        console.print("[dim]None installed.[/dim]")
    else:
        for plugin in sorted(found, key=lambda item: item.name):
            if plugin.ok:
                granted = ", ".join(sorted(plugin.permissions)) or "none"
                console.print(f"  [green]●[/green] [bold]{plugin.name}[/bold] {plugin.version}"
                              f" — {plugin.description or 'no description'}")
                console.print(f"    [dim]permissions:[/dim] {granted}")
                counts = []
                if plugin.tools:
                    counts.append(f"{len(plugin.tools)} tool(s)")
                if plugin.skills:
                    counts.append(f"{len(plugin.skills)} skill(s)")
                if plugin.settings:
                    counts.append(f"{len(plugin.settings)} setting(s)")
                if plugin.commands:
                    counts.append(f"{len(plugin.commands)} command(s)")
                for spec in plugin.tools:
                    counts.append(f"{spec.name} [{spec.risk}]")
                if counts:
                    console.print(f"    [dim]provides:[/dim] {', '.join(counts)}")
            else:
                console.print(f"  [red]✗[/red] [bold]{plugin.name}[/bold] — failed to load")
                console.print(f"    [red]{plugin.error}[/red]")
    if host.project_plugins_skipped:
        console.print(f"  [yellow]{host.project_plugins_skipped} plugin(s) in this project are "
                      f"not loaded: they run code from the repository, so they need "
                      f"--with-project-plugins.[/yellow]")
    for message in host.load_errors:
        console.print(f"  [yellow]{escape(message)}[/yellow]")


@app.command()
def prompts(
    action: str = typer.Argument(
        "list", help="list, show NAME, edit NAME, reset NAME, reset --all, "
                     "export [PATH], or path"),
    name: str = typer.Argument("", help="Prompt name, or destination file for `export`"),
    scope: str = typer.Option(
        "here", "--scope",
        help="`here` writes into the current project when you are in one, "
             "otherwise into your user config. `user` always writes to the user "
             "config; `project` requires a project."),
):
    """Inspect and customize every prompt the models receive.

    `edit` opens a prompt in $EDITOR and saves it as an override; `reset` puts a
    prompt, or all of them, back to the built-in text. An override file is the
    only thing `reset` touches -- the built-in prompts in the source are never
    modified, so an upgrade never fights with your changes.
    """
    from adaptive_harness.prompts import PromptRegistry, get_default_registry, DEFAULT_CONFIG_DIR
    # Load with the workspace, not the shared user-only registry: `prompts
    # show` and `prompts list` are the audit view, and an audit view that
    # silently omits the project's own overrides is worse than useless.
    registry = PromptRegistry.for_workspace(Path.cwd())
    override_paths = [Path(DEFAULT_CONFIG_DIR) / "prompts.json",
                      Path(".harness") / "prompts.json",
                      Path(os.environ["ADAPTIVE_PROMPTS_FILE"]) if os.getenv("ADAPTIVE_PROMPTS_FILE") else None]
    if action == "list":
        # Grouped by what can trigger each prompt, and previewed to fit the
        # terminal. A flat alphabetical list of 41 rows with mid-word cuts was
        # unreadable, and a user auditing what the models receive needs to see
        # which prompts can actually fire.
        from adaptive_harness.prompt_preview import render as render_prompts

        for line in render_prompts(registry, console=console):
            if line and not line.startswith("  "):
                console.print(f"[bold]{escape(line)}[/bold]")
            elif line:
                console.print(f"[dim]{escape(line)}[/dim]")
            else:
                console.print()
        overridden = sum(1 for name in registry.names() if registry.is_overridden(name))
        note = "\n[dim]Show one in full with `prompts show NAME`; "
        note += "override with a JSON file, paths from `prompts path`."
        if overridden:
            note += f" {overridden} currently overridden.[/dim]"
        console.print(note)
    elif action == "show":
        if not name:
            raise typer.BadParameter("show requires a prompt name", param_hint="name")
        try:
            console.print(Text(registry.get(name)))
        except KeyError as exc:
            raise typer.BadParameter(str(exc), param_hint="name") from exc
    elif action == "edit":
        from adaptive_harness.prompts_edit import PromptEditError, edit_prompt, sources_for

        if not name:
            raise typer.BadParameter("edit requires a prompt name", param_hint="NAME")
        if name not in registry.names():
            known = ", ".join(sorted(registry.names())[:6])
            raise typer.BadParameter(
                f"no prompt named {name!r}. Try `prompts list`; some names: {known}, ...",
                param_hint="NAME")
        workspace = Path.cwd() if scope != "user" else None
        try:
            if scope == "project" and workspace is None:
                raise PromptEditError("--scope project needs to be run inside a project.")
            destination, changed = edit_prompt(name, registry.get(name), workspace)
        except PromptEditError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(1) from exc
        if changed:
            console.print(f"[green]Saved[/green] {name} to {destination}")
            others = [entry.path for entry in sources_for(name, workspace)]
            if len(others) > 1:
                console.print(f"[yellow]Also overridden in "
                              f"{', '.join(str(path) for path in others)}; "
                              f"that copy takes precedence.[/yellow]")
        else:
            console.print("[dim]No change made.[/dim]")
    elif action == "reset":
        from adaptive_harness.prompts_edit import PromptEditError, reset_all, reset_override

        workspace = Path.cwd() if scope != "user" else None
        if name in {"", "all", "*"}:
            try:
                count, message = reset_all(workspace)
            except PromptEditError as exc:
                console.print(f"[red]{escape(str(exc))}[/red]")
                raise typer.Exit(1) from exc
            console.print(f"[{'green' if count else 'dim'}]{escape(message)}[/"
                          f"{'green' if count else 'dim'}]")
            return
        if name not in registry.names():
            raise typer.BadParameter(
                f"no prompt named {name!r}. Try `prompts list`.", param_hint="NAME")
        try:
            changed, message = reset_override(name, workspace)
        except PromptEditError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(1) from exc
        colour = "green" if changed else "yellow"
        console.print(f"[{colour}]{escape(message)}[/{colour}]")
        if not changed:
            raise typer.Exit(1)
    elif action == "export":
        target = Path(name or "output/prompts.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(registry.export(), indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]Wrote {len(registry.names())} prompts to {target}[/green]")
        console.print("[dim]Edit any entry and copy it to one of the override paths listed by `prompts path`.[/dim]")
    elif action == "path":
        for path in override_paths:
            if path is not None:
                console.print(f"  {path}{'  (exists)' if path.exists() else ''}")
    else:
        raise typer.BadParameter("Choose list, show, export, or path", param_hint="action")


@app.command()
def train(
    samples_per_class: int = typer.Option(800, "--samples", "-n", help="Number of synthetic samples per strategy class"),
    calibration: Optional[str] = typer.Option("sigmoid", "--calibration", "-c", help="Calibration method: sigmoid, isotonic, or none"),
    save_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--output", "-o", help="Target model save path"),
):
    """Trains a new task routing classifier on procedurally generated data."""
    cal = None if calibration and calibration.lower() == "none" else calibration
    console.print(f"[bold cyan]Generating synthetic dataset ({samples_per_class} per class)...[/bold cyan]")
    clf, eval_results = train_routing_model(
        samples_per_class=samples_per_class,
        calibration=cal,
        save_path=save_path,
    )

    console.print(f"[bold green]Model successfully trained and saved to {save_path}[/bold green]")
    console.print(f"Test Accuracy: [bold]{eval_results['accuracy']*100:.2f}%[/bold]")
    console.print(f"Top-2 Accuracy: [bold]{eval_results['top2_accuracy']*100:.2f}%[/bold]")
    console.print(f"Expected Calibration Error (ECE): [bold]{eval_results['ece']:.4f}[/bold]")
    console.print(f"Brier Score: [bold]{eval_results['brier_score']:.4f}[/bold]")


@app.command()
def benchmark(
    n_samples: int = typer.Option(100, "--n-samples", "-n", help="Samples per class in unseen benchmark"),
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", "-m", help="Path to routing model"),
    save_plots: bool = typer.Option(True, "--save-plots/--no-plots", help="Generate and save matplotlib plots"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="Directory for plots and reports"),
):
    """Runs the multi-policy ablation benchmark comparing raw classifier vs harness recovery."""
    clf = _ensure_classifier(model_path)

    console.print(f"[bold cyan]Running benchmark on unseen task distribution ({n_samples*5} total tasks)...[/bold cyan]")
    runner = BenchmarkRunner(classifier=clf, n_samples_per_class=n_samples)

    # Run ablation suite
    df, results = runner.run_ablation_suite()
    print_ablation_table(df)

    threshold_res = results.get("Threshold Policy")
    if threshold_res:
        print_failure_analysis(
            recovered_cases=threshold_res.recovered_cases,
            unrecovered_cases=threshold_res.failure_cases,
        )

    if save_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Confusion Matrix
        test_tasks = runner.tasks
        texts = [t[0] for t in test_tasks]
        y_true = [t[1] for t in test_tasks]
        y_pred = clf.predict(texts)
        plot_confusion_matrix(y_true, y_pred, clf.classes_, plots_dir / "confusion_matrix.png")

        # 2. Calibration Reliability Diagram
        probs_matrix = clf.predict_proba_matrix(texts)
        from adaptive_harness.models.calibration import compute_ece
        _, diag = compute_ece(y_true, probs_matrix, clf.classes_)
        plot_reliability_diagram(diag, plots_dir / "calibration_reliability.png")

        # 3. Harness vs Classifier comparison
        plot_harness_improvement(df, plots_dir / "harness_vs_classifier.png")

        # 3. Entropy distribution & confidence vs accuracy
        if threshold_res:
            plot_entropy_distribution(threshold_res.traces, plots_dir / "entropy_distribution.png")
            plot_confidence_vs_accuracy(threshold_res.traces, plots_dir / "confidence_vs_accuracy.png")

        console.print(f"[bold green]Plots generated in: {plots_dir}[/bold green]")


@app.command()
def experiment(
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="Target output directory"),
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", "-m", help="Path to routing model"),
):
    """Executes empirical research sweeps: confidence threshold trade-offs and learning curves."""
    clf = _ensure_classifier(model_path)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Running Confidence Threshold Sweep (0.30 - 0.95)...[/bold cyan]")
    threshold_vals = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    runner = BenchmarkRunner(classifier=clf, n_samples_per_class=60, seed=999)

    success_rates = []
    avg_attempts = []
    latencies = []

    for tau in threshold_vals:
        pol = ThresholdPolicy(high_confidence=tau, low_confidence=max(tau - 0.3, 0.2))
        res = runner.run_policy(pol)
        m = res.metrics
        success_rates.append(m["harness_success_rate"])
        avg_attempts.append(m["average_attempts"])
        latencies.append(m["average_time_ms"])

    plot_threshold_tradeoff(
        thresholds=threshold_vals,
        success_rates=success_rates,
        avg_attempts=avg_attempts,
        latencies=latencies,
        output_path=plots_dir / "threshold_tradeoff.png",
    )

    console.print("[bold cyan]Running Data Scaling / Learning Curve Sweep...[/bold cyan]")
    train_sizes = [100, 250, 500, 1000, 2000]
    clf_accs = []
    harness_succs = []

    for size in train_sizes:
        sub_clf, metrics = train_routing_model(
            samples_per_class=size // 5,
            save_path=None,
        )
        sub_runner = BenchmarkRunner(classifier=sub_clf, n_samples_per_class=40, seed=555)
        res = sub_runner.run_policy(ThresholdPolicy())
        clf_accs.append(res.metrics["classifier_accuracy"])
        harness_succs.append(res.metrics["harness_success_rate"])

    plot_learning_curve(
        train_sizes=train_sizes,
        clf_accuracies=clf_accs,
        harness_successes=harness_succs,
        output_path=plots_dir / "learning_curve.png",
    )

    console.print(f"[bold green]Experiment suite complete! Visualizations saved to {plots_dir}[/bold green]")


@app.command()
def retrain(
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Path to SQLite experience database"),
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", "-m", help="Path to model file"),
):
    """Learns from historical verified executions stored in SQLite and retrains the model."""
    console.print(f"[bold cyan]Checking experience database: {db_path}...[/bold cyan]")
    clf, res = retrain_from_experience(db_path=db_path, model_path=model_path)

    if not res.get("retrained"):
        console.print(f"[yellow]{res.get('reason')}[/yellow]")
        return

    console.print(f"[bold green]Retraining finished successfully![/bold green]")
    console.print(f"Verified experience samples used: [bold]{res['experience_samples_used']}[/bold]")
    console.print(f"Validation accuracy: [bold]{res['validation_accuracy']*100:.2f}%[/bold]")
    console.print(f"Model saved: [bold]{res['saved']}[/bold]")


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of records to display"),
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Path to SQLite database"),
):
    """Displays recent execution traces recorded in the experience store."""
    repo = ExperienceRepository(db_path)
    traces = repo.get_recent_traces(limit=limit)
    if not traces:
        console.print("[dim]No execution history found in database.[/dim]")
        return
    print_history_table(traces)


@app.command()
def report(
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Path to SQLite database"),
):
    """Summarizes overall execution and recovery metrics from historical experience."""
    repo = ExperienceRepository(db_path)
    stats = repo.get_statistics()

    console.print("\n[bold cyan]=== Historical Harness Execution Summary ===[/bold cyan]")
    console.print(f"Total Tasks Executed: [bold]{stats['total_executions']}[/bold]")
    console.print(f"Overall Success Rate: [bold green]{stats['success_rate']*100:.1f}%[/bold green]")
    console.print(f"Classifier Errors Rescued: [bold yellow]{stats['recovery_count']}[/bold yellow]")
    console.print(f"Average Execution Latency: [bold]{stats['avg_time_ms']} ms[/bold]")
    console.print(f"Average Attempts per Task: [bold]{stats['avg_attempts']}[/bold]")
    console.print(f"Mean Classifier Confidence: [bold]{stats['avg_confidence']*100:.1f}%[/bold]")
    console.print(f"Mean Routing Entropy: [bold]{stats['avg_entropy']:.3f} bits[/bold]")

    if stats.get("strategy_distribution"):
        console.print("\n[bold magenta]Strategy Execution Distribution:[/bold magenta]")
        for strat, count in stats["strategy_distribution"].items():
            console.print(f"  {strat:15s}: {count}")
    console.print()


@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic, e.g. 'optimal routing under heavy-tailed delay'"),
    root: Path = typer.Option(Path("research"), "--root", help="Artifact tree root"),
    objective: str = typer.Option("", "--objective", help="Explicit statement of what must be proven"),
    claim: str = typer.Option("", "--claim",
                              help="A checkable claim as 'lhs == rhs' in SymPy syntax, decided directly"),
    symbols: str = typer.Option("", "--symbols", help="Comma-separated free symbols for --claim"),
    seed: int = typer.Option(20260926, "--seed", help="Pinned seed for every experiment"),
    worker_budget: int = typer.Option(
        16000, "--worker-budget-tokens", min=0,
        help="Token ceiling per worker attempt. 0 removes the ceiling. A research "
             "run fans out to many workers, so this is what bounds what one "
             "invocation can cost."),
    install_typst: bool = typer.Option(
        False, "--install-typst",
        help="Allow the harness to pip install the pinned Typst binding if no "
             "engine is found. Off by default: it mutates your environment."),
    max_cycles: Optional[int] = typer.Option(None, "--max-cycles",
                                             help="Operator safety valve; the loop is not turn-limited by default"),
    patience: int = typer.Option(2, "--patience", help="No-progress cycles tolerated before escalating"),
    max_workers: int = typer.Option(8, "--max-workers", help="Per-division worker cap"),
    worker_steps: int = typer.Option(24, "--worker-steps", min=1,
                                     help="Maximum tool-loop steps per worker attempt"),
    absolute_ceiling: int = typer.Option(64, "--absolute-ceiling",
                                       min=1, help="Maximum convergence cycles, even without --max-cycles"),
    parallel_workers: int = typer.Option(4, "--parallel-workers", min=1,
                                         help="Worker tool loops a division may run at once; 1 is serial"),
    worker_timeout: Optional[float] = typer.Option(None, "--worker-timeout",
                                                  help="Seconds a single worker tool loop may run before TIMED_OUT"),
    task_lease: float = typer.Option(1800.0, "--task-lease",
                                     help="Seconds a worker may hold a task before another may "
                                          "take it; renewed automatically while the worker runs"),
    task_attempts: int = typer.Option(3, "--task-attempts", min=1,
                                      help="Attempts per task before it is written off"),
    overseer_stop: int = typer.Option(3, "--overseer-stop", min=1,
                                      help="Runtime-overseer escalations in one attempt before a lead "
                                           "stops the worker; 1 is aggressive, higher is patient"),
    resume: bool = typer.Option(True, "--resume/--no-resume",
                                help="Recover the persisted task board and ledger from an interrupted run"),
    provider: str = typer.Option("openrouter", "--provider",
                                 help="Configured API provider for independent research agents"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model the research agents use. Defaults to the provider's own "
             "default, which any account with that provider can call."),
    author: bool = typer.Option(True, "--author/--offline-legacy",
                               help="Run the live LLM research swarm (default); "
                                    "offline legacy mode is for reproducibility only"),
):
    """Runs the autonomous research swarm and publishes a paper with a verdict.

    The live Director and division agents author claims and artifacts through
    separate LLM tool loops. SymPy, Lean, seeded experiments, and Typst then
    verify the artifacts. Offline legacy mode keeps the historical fixed-topic
    demonstration available explicitly.

    The loop is stagnation-limited rather than turn-limited: it exits on
    convergence, or when a cycle provably cannot change the verdict and the
    escalated worker pool does not help.

    Ctrl-C is safe: the first signal asks every active worker to stop and the run
    reports EXTERNAL_STOP with the operator named, rather than leaving orphaned
    child processes and a ledger that claims the loop converged.
    """
    from adaptive_harness.research.claim import Verdict
    from adaptive_harness.research.swarm import ResearchSwarm, SwarmConfig

    client_factory = None
    if author:
        try:
            client_factory = _llm_client_factory(provider=provider, model=model)
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            raise typer.BadParameter(str(exc), param_hint="--author") from exc

    config = SwarmConfig(max_cycles=max_cycles, stagnation_patience=patience,
                         max_workers_per_division=max_workers,
                         absolute_ceiling=absolute_ceiling, worker_max_steps=worker_steps,
                         max_parallel_workers=parallel_workers,
                         worker_timeout_s=worker_timeout, resume=resume,
                         overseer_stop_threshold=overseer_stop,
                         task_lease_s=task_lease,
                         task_lease_renew_s=max(30.0, task_lease / 6.0),
                         max_task_attempts=task_attempts,
                         seed=seed,
                         llm_client_factory=client_factory,
                         claim=claim,
                         worker_budget_tokens=worker_budget,
                         auto_install_typst=install_typst,
                         claim_symbols=tuple(name.strip() for name in symbols.split(",") if name.strip()))
    swarm = ResearchSwarm(topic, root=root, config=config)
    console.print(f"[bold cyan]Research swarm[/bold cyan] {topic}")
    console.print(f"[dim]strategy: {swarm.plan.strategy} | propositions: "
                  f"{len(swarm.plan.propositions)}[/dim]")
    if swarm.plan.notes:
        console.print(f"[dim]{swarm.plan.notes}[/dim]")
    recovered = swarm.recovery_report()
    if recovered.get("resumed"):
        console.print(f"[dim]resumed: {recovered['tasks']} task(s) restored, "
                      f"{recovered['healed']} healed, "
                      f"{recovered['messages']} message(s) replayed; "
                      f"completed tasks are not re-run[/dim]")
    console.print(f"[dim]artifacts: {swarm.workspace.root}[/dim]")
    console.print(f"[dim]interrupt with Ctrl-C to stop every active worker; "
                  f"state is preserved in {swarm.workspace.ledger_path.name}[/dim]\n")

    def on_status(message: str) -> None:
        console.print(f"[dim]{message}[/dim]")

    def on_interrupt(*_args: object) -> None:
        # The first Ctrl-C is a *request*: the control plane records the actor,
        # cancels every live worker, and kills their process groups so the run ends
        # with an attributable EXTERNAL_STOP instead of a torn-off traceback.
        console.print("\n[yellow]interrupt received: stopping every active worker "
                      "(press Ctrl-C again to abort immediately)[/yellow]")
        swarm.stop_run(actor="operator", reason="SIGINT from the operator")

    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, on_interrupt)
    except ValueError:  # pragma: no cover - non-main thread
        previous_handler = None
    try:
        outcome = swarm.run(objective=objective, on_status=on_status)
    except KeyboardInterrupt:  # pragma: no cover - second Ctrl-C
        swarm.stop_run(actor="operator", reason="operator forced an abort")
        raise
    finally:
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGINT, previous_handler)
            except ValueError:  # pragma: no cover - non-main thread
                pass

    status = swarm.swarm_status()
    console.print("\n[bold]Swarm state at exit[/bold]")
    for line in swarm.render_status().splitlines():
        console.print(line)
    if status["open_tasks"]:
        console.print(f"[dim]open tasks: "
                      f"{', '.join(task['task_id'] for task in status['open_tasks'])}[/dim]")
    if status["unacknowledged_requests"]:
        console.print(f"[dim]unanswered help requests: "
                      f"{len(status['unacknowledged_requests'])}[/dim]")
    if status["stopped_by"]:
        console.print(f"[yellow]run was stopped by {status['stopped_by']}[/yellow]")

    verdict = swarm.claims.headline
    color = {"PROVEN": "green", "DISPROVEN": "red"}.get(verdict.value, "yellow")
    # Two different verdicts are reported: the mathematical one (what the
    # derivations decided) and the process one (whether the run met every
    # invariant). Printing a green PROVEN next to UNSOLVED and exiting 2 was
    # the single most misleading thing this command could do, so the process
    # verdict is stated first and the relationship is spelled out.
    process = "COMPLETE" if outcome.solved else "INCOMPLETE"
    process_color = "green" if outcome.solved else "red"
    console.print(f"\n[bold {process_color}]RUN: {process} — {outcome.stop_reason.value}[/bold {process_color}]")
    if not outcome.solved and verdict is Verdict.PROVEN:
        console.print("[yellow]The mathematics was decided, but the run did not satisfy every "
                      "invariant, so nothing is published as settled.[/yellow]")
    console.print(f"[bold {color}]MATHEMATICAL VERDICT: {verdict.value}[/bold {color}]")
    console.print(swarm.claims.summary())
    for item in swarm.claims.adjudications:
        console.print(f"  [dim]{item.prop_id}[/dim] {item.verdict.value:<13} {item.statement[:66]}")
    console.print()
    for line in outcome.render().splitlines():
        console.print(line)
    if swarm._last_report is not None:
        for item in swarm._last_report.statuses:
            mark = "[green]PASS[/green]" if item.satisfied else "[red]FAIL[/red]"
            console.print(f"  {mark} {item.invariant.value}: {item.detail}")
    if outcome.pdf:
        label = "Paper" if outcome.solved else "Progress report"
        console.print(f"\n[bold green]{label}:[/bold green] {outcome.pdf}")
    console.print(f"[dim]Ledger: {swarm.workspace.ledger_path}[/dim]")
    console.print(f"[dim]Task board: {swarm.workspace.root / 'task_board.json'}[/dim]")
    if not outcome.solved:
        raise typer.Exit(code=2)


def _llm_client_factory(provider: str = "openrouter", model: Optional[str] = None):
    """Build a factory that mints a fresh LLM client per research worker.

    A new client per worker is deliberate: each subagent needs its own
    conversation, tool set, and system prompt, and sharing one would interleave
    their histories.

    ``model`` lets an operator choose the research model. The default comes from
    the provider's entry, which is a model any account with that provider can
    actually call -- research previously pinned one private slug, so a customer
    without access to it could not run the product's headline feature at all.
    """
    from adaptive_harness.llm.client import LLMClient
    from adaptive_harness.data.credentials import CredentialsManager

    saved_keys = CredentialsManager().load()
    probe = LLMClient(provider=provider, provider_keys=saved_keys)
    if probe.is_mock:
        raise RuntimeError(
            f"no live {provider} model is configured. Set the provider's environment "
            f"variable (for example OPENROUTER_API_KEY), or save a key in the TUI with "
            f"/key {provider} <key>, then run research again.")
    chosen = model or probe.default_model

    def factory():
        return LLMClient(api_key=probe.api_key, base_url=probe.base_url,
                         default_model=chosen, force_mock=False,
                         provider=probe.provider, provider_keys=probe.provider_keys,
                         backup_providers=probe.backup_providers)

    return factory


if __name__ == "__main__":
    app()
