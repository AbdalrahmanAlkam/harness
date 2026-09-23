"""Interactive command-line interface for Adaptive Agent Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt

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
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible task model endpoint"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Default model ID (e.g. anthropic/claude-3.7-sonnet)"),
    tier: Optional[str] = typer.Option(None, "--tier", help="Force model tier: fast, standard, reasoning"),
    classifier_backend: str = typer.Option("sklearn", "--classifier-backend", help="sklearn, ollama, onnx, openrouter"),
    classifier_model: Optional[str] = typer.Option(None, "--classifier-model", help="Classifier model ID or ONNX directory"),
    classifier_endpoint: Optional[str] = typer.Option(None, "--classifier-endpoint", help="Local or OpenRouter classifier endpoint"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Working directory for agent tools"),
    session: Optional[str] = typer.Option(None, "--session", help="Resume an existing TUI session ID"),
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Path to experience database"),
):
    """Launches the interactive Textual TUI development environment with pervasive classifiers."""
    from adaptive_harness.tui.app import AdaptiveHarnessApp
    from adaptive_harness.llm.client import MODEL_TIERS

    if tier and tier not in MODEL_TIERS:
        raise typer.BadParameter("Choose fast, standard, or reasoning", param_hint="--tier")

    try:
        tui_app = AdaptiveHarnessApp(
            api_key=api_key,
            base_url=base_url,
            default_model=(None if model and model.lower() == "auto" else model) or (MODEL_TIERS[tier] if tier in MODEL_TIERS else None),
            classifier_backend=classifier_backend,
            classifier_model=classifier_model,
            classifier_endpoint=classifier_endpoint,
            workspace_root=str(workspace),
            session_id=session,
            db_path=db_path,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    tui_app.run()


@app.command()
def dev(
    task: str = typer.Argument(..., help="Software engineering task to execute"),
    api_key: Optional[str] = typer.Option(None, "--key", "-k", help="OpenRouter or OpenAI API key"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible task model endpoint"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Default model ID"),
    tier: Optional[str] = typer.Option(None, "--tier", help="Force model tier: fast, standard, reasoning"),
    classifier_backend: str = typer.Option("sklearn", "--classifier-backend", help="sklearn, ollama, onnx, openrouter"),
    classifier_model: Optional[str] = typer.Option(None, "--classifier-model", help="Classifier model ID or ONNX directory"),
    classifier_endpoint: Optional[str] = typer.Option(None, "--classifier-endpoint", help="Classifier endpoint"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Working directory for agent tools"),
    skill: Optional[list[str]] = typer.Option(None, "--skill", help="Enable a named workspace or user skill"),
    db_path: Path = typer.Option(Path("output/experience.db"), "--db", help="Path to experience database"),
):
    """Runs a developer task through the DeveloperAgent with pervasive classification and verification."""
    from adaptive_harness.agent.agent import DeveloperAgent
    from adaptive_harness.llm.client import LLMClient
    from adaptive_harness.llm.client import MODEL_TIERS
    from adaptive_harness.classifiers.engine import create_backend
    from adaptive_harness.agent.skills import SkillCatalog

    if tier and tier not in MODEL_TIERS:
        raise typer.BadParameter("Choose fast, standard, or reasoning", param_hint="--tier")

    selected_model = (None if model and model.lower() == "auto" else model) or (MODEL_TIERS[tier] if tier in MODEL_TIERS else None)
    client = LLMClient(api_key=api_key, base_url=base_url, default_model=selected_model)
    repo = ExperienceRepository(db_path)
    if not workspace.is_dir():
        raise typer.BadParameter(f"Workspace directory does not exist: {workspace}", param_hint="--workspace")
    try:
        backend = create_backend(classifier_backend, classifier_model, classifier_endpoint)
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--classifier-backend") from exc
    agent = DeveloperAgent(llm_client=client, repository=repo, workspace_root=str(workspace), explicit_model=selected_model,
                           classifier_backend=backend)
    catalog = SkillCatalog(workspace)
    for name in skill or []:
        try:
            agent.active_skills[name] = catalog.read(name)
        except (ValueError, OSError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--skill") from exc

    console.print(f"\n[bold cyan]=== Adaptive Developer Agent Task ===[/bold cyan]")
    console.print(f"Task: [bold white]\"{task}\"[/bold white]")

    for event in agent.run_stream(task):
        et = event.event_type
        p = event.payload
        if et == "skill_classification":
            console.print(f"  [cyan]Skill Classifier:[/cyan] {p['primary_skill']} ({p['confidence']*100:.1f}%)")
        elif et == "ambiguity_assessment":
            color = "green" if p["risk_level"] == "low" else "yellow"
            console.print(f"  [{color}]Ambiguity & Risk:[/{color}] H={p['entropy']:.2f} bits | Risk={p['risk_level'].upper()} | Ask={p['should_ask_question']}")
        elif et == "model_routing":
            console.print(f"  [magenta]Model Tier:[/magenta] [{p['tier'].upper()}] -> {p['model']}")
        elif et == "llm_error":
            console.print(f"[red]Model error: {escape(p['message'])}[/red]")
        elif et == "thought":
            console.print(f"\n[bold magenta]Agent Thought:[/bold magenta] {escape(p['content'])}")
        elif et == "tool_call":
            console.print(f"  [bold yellow]Tool Call:[/bold yellow] [cyan]{p['name']}[/cyan] [dim]{escape(str(p['arguments']))}[/dim]")
        elif et == "tool_result":
            status_col = "green" if p["success"] else "red"
            console.print(f"  [{status_col}]Tool Result ({p['time_ms']} ms):[/{status_col}] {escape(p['output'][:200])}")
        elif et == "verification":
            badge_col = "green" if p["status"] == "SUCCESS" else "red bold"
            console.print(f"  [dim]Verification Classifier: [{badge_col}]{p['status']}[/{badge_col}] -> Action: {p['action']}[/dim]")
        elif et == "response":
            status = "✓ Completed" if p.get("success", True) else "Stopped before completion"
            color = "green" if p.get("success", True) else "yellow"
            console.print(f"\n[bold {color}]{status} in {p['total_time_ms']} ms ({p['steps']} steps)[/bold {color}]\n")


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
