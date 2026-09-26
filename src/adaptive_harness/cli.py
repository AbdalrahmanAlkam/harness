"""Interactive command-line interface for Adaptive Agent Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import os
import typer
from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from adaptive_harness.tui.formatting import format_model_markdown

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


@app.command()
def tui(
    api_key: Optional[str] = typer.Option(None, "--key", "-k", help="OpenRouter or OpenAI API key"),
    provider: Optional[str] = typer.Option(None, "--provider", help="openrouter, anthropic, openai, deepseek, google, groq, local"),
    backup_provider: Optional[list[str]] = typer.Option(None, "--backup-provider", help="Fail over on 429/5xx; repeat in priority order"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible task model endpoint"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Default model ID (e.g. anthropic/claude-sonnet-4)"),
    tier: Optional[str] = typer.Option(None, "--tier", help="Force model tier: fast, standard, reasoning"),
    mode: str = typer.Option("auto", "--mode", help="Operational mode: coding, research, science, security, auto"),
    thinking: str = typer.Option("auto", "--thinking", help="Thinking level: none, low, medium, deep, auto"),
    safety: Optional[str] = typer.Option(None, "--safety", help="Interaction profile: turbo, balanced, cautious, strict (default turbo)"),
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
    """Launches the interactive Textual TUI development environment with pervasive classifiers."""
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
    mode: str = typer.Option("auto", "--mode", help="Operational mode: coding, research, science, security, auto"),
    thinking: str = typer.Option("auto", "--thinking", help="Thinking level: none, low, medium, deep, auto"),
    safety: Optional[str] = typer.Option(None, "--safety", help="Interaction profile: turbo, balanced, cautious, strict (default turbo)"),
    step_policy: str = typer.Option("classifier", "--step-policy", help="Tool-step limits: classifier (default; stops circling loops), fixed (hardcoded per-thinking budgets), unbounded"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", min=1, help="Explicit tool-step cap overriding the step policy"),
    swarm: bool = typer.Option(False, "--swarm", help="Expose delegate_subagent to the coordinator agent"),
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
    """Runs a developer task through the DeveloperAgent with pervasive classification and verification."""
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
                           step_policy=step_policy, max_steps=max_steps)
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
    for event in agent.run_stream(task):
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
            label = {"thinking": "Thinking", "generating": "Generating",
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

    if not task_succeeded:
        raise typer.Exit(code=1)


@app.command()
def prompts(
    action: str = typer.Argument("list", help="list, show NAME, export [PATH], or path"),
    name: str = typer.Argument("", help="Prompt name for `show`, or destination file for `export`"),
):
    """Inspect and customize every prompt the models receive."""
    from adaptive_harness.prompts import PromptRegistry, get_default_registry, DEFAULT_CONFIG_DIR
    registry = get_default_registry()
    override_paths = [Path(DEFAULT_CONFIG_DIR) / "prompts.json",
                      Path(".harness") / "prompts.json",
                      Path(os.environ["ADAPTIVE_PROMPTS_FILE"]) if os.getenv("ADAPTIVE_PROMPTS_FILE") else None]
    if action == "list":
        for prompt_name in registry.names():
            marker = " [overridden]" if registry.is_overridden(prompt_name) else ""
            preview = registry.get(prompt_name).strip().splitlines()[0][:90]
            console.print(f"  [cyan]{prompt_name}[/cyan]{marker} [dim]{escape(preview)}[/dim]")
        console.print("\n[dim]Override with a JSON file mapping names to new text; see `prompts path`.[/dim]")
    elif action == "show":
        if not name:
            raise typer.BadParameter("show requires a prompt name", param_hint="name")
        try:
            console.print(Text(registry.get(name)))
        except KeyError as exc:
            raise typer.BadParameter(str(exc), param_hint="name") from exc
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


if __name__ == "__main__":
    app()
