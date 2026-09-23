"""Rich terminal reporting for benchmarks, failure analysis, and individual execution traces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from adaptive_harness.models.domain import ExecutionTrace

console = Console()


def print_ablation_table(df: pd.DataFrame) -> None:
    """Renders the ablation experiment table using Rich."""
    table = Table(
        title="[bold green]Adaptive Agent Harness - Policy Ablation Benchmark[/bold green]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )

    for col in df.columns:
        table.add_column(col, justify="center" if col != "Policy" else "left")

    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)


def print_failure_analysis(
    recovered_cases: List[Dict[str, Any]],
    unrecovered_cases: List[Dict[str, Any]],
    max_items: int = 5,
) -> None:
    """Formats and prints detailed analysis of confident errors and harness recoveries."""
    # 1. Recovered cases
    table_rec = Table(
        title="[bold yellow]Most Confident Classifier Mispredictions Rescued by Harness[/bold yellow]",
        show_header=True,
        header_style="bold yellow",
        border_style="yellow",
    )
    table_rec.add_column("Task Text", style="white", max_width=45, overflow="ellipsis")
    table_rec.add_column("Classifier Top-1 (Conf)", justify="center", style="red")
    table_rec.add_column("Ground Truth", justify="center", style="green")
    table_rec.add_column("Final Rescued Strategy", justify="center", style="cyan")
    table_rec.add_column("Attempts Sequence", justify="center", style="dim")

    for item in recovered_cases[:max_items]:
        table_rec.add_row(
            item["task"],
            f"{item['predicted']} ({item['confidence']*100:.1f}%)",
            item["ground_truth"],
            item["final_strategy"],
            " → ".join(item["attempts"]),
        )

    if recovered_cases:
        console.print(table_rec)
    else:
        console.print("[dim]No classifier mispredictions occurred to rescue.[/dim]")

    # 2. Unrecovered failures
    if unrecovered_cases:
        table_fail = Table(
            title="[bold red]Unrecovered Execution Failures[/bold red]",
            show_header=True,
            header_style="bold red",
            border_style="red",
        )
        table_fail.add_column("Task Text", style="white", max_width=45, overflow="ellipsis")
        table_fail.add_column("Predicted", justify="center", style="red")
        table_fail.add_column("Ground Truth", justify="center", style="green")
        table_fail.add_column("Final Strategy", justify="center", style="dim")
        table_fail.add_column("Attempts", justify="center", style="dim")

        for item in unrecovered_cases[:max_items]:
            table_fail.add_row(
                item["task"],
                f"{item['predicted']} ({item['confidence']*100:.1f}%)",
                item["ground_truth"],
                item["final_strategy"],
                " → ".join(item["attempts"]),
            )
        console.print(table_fail)


def print_execution_trace(trace: ExecutionTrace) -> None:
    """Displays an interactive / real-time breakdown of an individual task execution."""
    console.print(f"\n[bold cyan]Task ID:[/bold cyan] {trace.task_id}")
    console.print(f"[bold cyan]Input:[/bold cyan] [white bold]\"{trace.task_text}\"[/white bold]")

    # Probability Distribution
    prob_table = Table(
        title="[bold magenta]Classifier Routing Probabilities[/bold magenta]",
        box=None,
        show_header=False,
    )
    prob_table.add_column("Strategy", justify="left", style="bold")
    prob_table.add_column("Bar", justify="left")
    prob_table.add_column("Percentage", justify="right")

    sorted_probs = sorted(trace.classifier_probabilities.items(), key=lambda x: x[1], reverse=True)
    for strat, prob in sorted_probs:
        bar_len = int(prob * 30)
        bar_str = "█" * bar_len + "░" * (30 - bar_len)
        color = "green" if prob >= 0.5 else ("yellow" if prob >= 0.2 else "dim")
        prob_table.add_row(
            strat,
            f"[{color}]{bar_str}[/{color}]",
            f"[{color}]{prob*100:5.1f}%[/{color}]",
        )
    console.print(prob_table)

    # Uncertainty Metrics
    console.print(
        f"[bold]Entropy:[/bold] {trace.entropy:.3f} bits | "
        f"[bold]Confidence:[/bold] {trace.confidence*100:.1f}% | "
        f"[bold]Margin:[/bold] {trace.margin*100:.1f}% | "
        f"[bold]Policy:[/bold] {trace.policy}"
    )

    # Execution Attempts
    console.print("\n[bold yellow]Execution & Verification Trail:[/bold yellow]")
    for i, a in enumerate(trace.attempts):
        status_icon = "[green]✓ PASS[/green]" if a.success else "[red]✗ FAIL[/red]"
        console.print(f"  Attempt {i+1}: Strategy [cyan]{a.strategy}[/cyan] ({a.time_ms:.2f} ms) -> {status_icon}")
        if a.error:
            console.print(f"    [dim red]Error/Reason: {a.error}[/dim red]")
        if a.value is not None:
            console.print(f"    [dim green]Output: {a.value}[/dim green]")

    # Outcome
    outcome_text = (
        "[bold green]SUCCESS[/bold green]"
        if trace.success
        else "[bold red]FALLBACK / UNSOLVED[/bold red]"
    )
    rec_text = " ([bold yellow]Recovered from classifier error![/bold yellow])" if trace.recovered else ""
    console.print(
        Panel(
            f"Final Strategy: [bold cyan]{trace.final_strategy}[/bold cyan] | "
            f"Outcome: {outcome_text}{rec_text} | "
            f"Total Wall Time: [bold]{trace.total_time_ms:.2f} ms[/bold]",
            title="Execution Result",
            border_style="green" if trace.success else "red",
        )
    )


def print_history_table(history: List[Dict[str, Any]]) -> None:
    """Renders recent history records from SQLite store."""
    table = Table(
        title="[bold blue]Recent Execution History from SQLite[/bold blue]",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("ID", justify="right")
    table.add_column("Task Text", style="white", max_width=40, overflow="ellipsis")
    table.add_column("Predicted", justify="center")
    table.add_column("Final", justify="center", style="cyan")
    table.add_column("Success", justify="center")
    table.add_column("Recovered", justify="center")
    table.add_column("Attempts", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Timestamp", justify="center", style="dim")

    for r in history:
        succ = "[green]Yes[/green]" if r["harness_success"] else "[red]No[/red]"
        rec = "[yellow]Yes[/yellow]" if r["recovered"] else "[dim]No[/dim]"
        table.add_row(
            str(r["id"]),
            r["task_text"],
            r["predicted_strategy"],
            r["selected_strategy"],
            succ,
            rec,
            str(r["attempts_count"]),
            f"{r['execution_time_ms']:.1f}",
            str(r["created_at"])[:19],
        )

    console.print(table)
