"""End-to-end self-contained demonstration of Adaptive Agent Harness."""

from __future__ import annotations

from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from adaptive_harness.cli import _ensure_classifier
from adaptive_harness.dashboard.report import (
    print_ablation_table,
    print_execution_trace,
    print_failure_analysis,
)
from adaptive_harness.data.storage import ExperienceRepository
from adaptive_harness.evaluation.benchmark import BenchmarkRunner
from adaptive_harness.evaluation.plots import (
    plot_confidence_vs_accuracy,
    plot_confusion_matrix,
    plot_entropy_distribution,
    plot_harness_improvement,
    plot_reliability_diagram,
)
from adaptive_harness.harness.harness import Harness
from adaptive_harness.harness.policy import ThresholdPolicy
from adaptive_harness.models.classifier import DEFAULT_MODEL_PATH

console = Console()


def run_demo() -> None:
    """Executes full automated demonstration showcasing classification, verification, and recovery."""
    console.print(
        Panel(
            "[bold white]Adaptive Agent Harness[/bold white]\n"
            "[cyan]Investigating whether an execution harness can compensate for classifier uncertainty and error[/cyan]",
            title="System Demonstration",
            border_style="cyan",
        )
    )

    # 1. Train / Load routing model
    console.print("\n[bold]Step 1:[/bold] Ensuring routing model is ready...")
    clf = _ensure_classifier(DEFAULT_MODEL_PATH)
    console.print(f"[green]Classifier loaded with {len(clf.classes_)} strategies: {', '.join(clf.classes_)}[/green]")

    repo = ExperienceRepository("output/experience.db")
    harness = Harness(classifier=clf, policy=ThresholdPolicy(), repository=repo)

    # 2. Run representative tasks
    demo_tasks = [
        ("calculate 37 * 14 + 19", "arithmetic"),
        ("solve 4*x - 12 = 28", "equations"),
        ("roots of x^2 - 7*x + 12 = 0", "equations"),
        ("sort: 42 17 88 3 99 21 5", "algorithms"),
        ("shortest-path: start=A end=D edges=A-B:2,B-D:3,A-C:1,C-D:5", "algorithms"),
        ("word-count: The adaptive harness enables resilient multi-agent execution.", "text_stats"),
        ("entropy: abbcccddddeeeee", "text_stats"),
        ("what is the capital of Iceland?", "fallback"),
    ]

    console.print("\n[bold]Step 2:[/bold] Executing diverse representative tasks...")
    for text, label in demo_tasks:
        trace = harness.run(text, ground_truth=label)
        console.print(
            f"• [bold]{text[:45]:45s}[/bold] -> "
            f"Pred: [cyan]{trace.classifier_top1:12s}[/cyan] ({trace.confidence*100:4.1f}%) | "
            f"Final: [magenta]{trace.final_strategy:12s}[/magenta] | "
            f"Pass: [{'green' if trace.success else 'yellow'}]{str(trace.success):5s}[/] | "
            f"Time: {trace.total_time_ms:4.1f}ms"
        )

    # 3. Deliberately show an ambiguous task where classifier makes a mistake and harness recovers!
    console.print("\n[bold]Step 3:[/bold] Demonstrating [bold yellow]Uncertainty Detection & Recovery[/bold yellow] on ambiguous query...")
    ambiguous_task = "calculate the value of x in 5*x + 15 = 45"
    console.print(f"[italic]Ambiguous Task query:[/italic] [bold white]\"{ambiguous_task}\"[/bold white]")
    console.print("[dim](Task contains arithmetic keyword 'calculate' but is algebraically an equation)[/dim]")

    recovery_trace = harness.run(ambiguous_task, ground_truth="equations")
    print_execution_trace(recovery_trace)

    # 4. Run benchmark
    console.print("\n[bold]Step 4:[/bold] Running multi-policy benchmark suite on held-out tasks...")
    runner = BenchmarkRunner(classifier=clf, n_samples_per_class=40, seed=777)
    df, results = runner.run_ablation_suite()
    print_ablation_table(df)

    # 5. Failure analysis
    thresh_res = results.get("Threshold Policy")
    if thresh_res:
        print_failure_analysis(
            recovered_cases=thresh_res.recovered_cases,
            unrecovered_cases=thresh_res.failure_cases,
            max_items=3,
        )

    # 6. Generate plots
    console.print("\n[bold]Step 5:[/bold] Generating visualization plots...")
    plots_dir = Path("output/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_true = [t[1] for t in runner.tasks]
    y_pred = clf.predict([t[0] for t in runner.tasks])
    plot_confusion_matrix(y_true, y_pred, clf.classes_, plots_dir / "confusion_matrix.png")
    plot_harness_improvement(df, plots_dir / "harness_vs_classifier.png")

    if thresh_res:
        plot_entropy_distribution(thresh_res.traces, plots_dir / "entropy_distribution.png")
        plot_confidence_vs_accuracy(thresh_res.traces, plots_dir / "confidence_vs_accuracy.png")

    console.print(f"[bold green]Visualizations saved to {plots_dir.resolve()}[/bold green]")
    console.print("\n[bold green]✓ Demo completed successfully![/bold green]\n")


if __name__ == "__main__":
    run_demo()
