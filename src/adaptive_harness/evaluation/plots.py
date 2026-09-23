"""Matplotlib visualizations for calibration, confidence, entropy, and harness gains."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from adaptive_harness.models.domain import ExecutionTrace


OUTPUT_DIR = Path("output/plots")


def ensure_output_dir(path: Path | str = OUTPUT_DIR) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str],
    output_path: Path | str = OUTPUT_DIR / "confusion_matrix.png",
) -> Path:
    """Renders and saves a normalized confusion matrix plot."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Strategy",
        xlabel="Predicted Strategy",
        title="Routing Classifier Confusion Matrix (Normalized)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val*100:.1f}%\n" if val > 0.005 else "-",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(target_path, dpi=200)
    plt.close(fig)
    return target_path


def plot_reliability_diagram(
    diagram_data: Dict[str, Any],
    output_path: Path | str = OUTPUT_DIR / "calibration_reliability.png",
) -> Path:
    """Plots reliability diagram comparing confidence to empirical accuracy with ECE."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    bin_accs = diagram_data["bin_accuracies"]
    bin_confs = diagram_data["bin_confidences"]
    bin_counts = diagram_data["bin_counts"]
    ece = diagram_data.get("ece", 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Diagonal ideal line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    # Actual confidence vs accuracy bars
    bin_edges = np.array(diagram_data["bin_edges"])
    widths = np.diff(bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    ax1.bar(
        centers,
        bin_accs,
        width=widths * 0.9,
        alpha=0.6,
        color="#1f77b4",
        edgecolor="#0d47a1",
        label="Observed Accuracy",
    )
    ax1.plot(centers, bin_confs, "ro-", label="Mean Confidence", alpha=0.8)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_title(f"Reliability Diagram (Expected Calibration Error: {ece:.4f})", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left")

    # Lower plot: sample count histogram per bin
    ax2.bar(centers, bin_counts, width=widths * 0.9, color="#78909c", alpha=0.8, edgecolor="#455a64")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence Bin", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(target_path, dpi=200)
    plt.close(fig)
    return target_path


def plot_confidence_vs_accuracy(
    traces: List[ExecutionTrace],
    output_path: Path | str = OUTPUT_DIR / "confidence_vs_accuracy.png",
    n_bins: int = 10,
) -> Path:
    """Plots binned prediction accuracy as a function of classifier confidence."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    confidences = np.array([t.confidence for t in traces])
    correct = np.array([
        (t.classifier_top1 == t.metadata.get("ground_truth"))
        for t in traces
    ], dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    widths = np.diff(bins)
    bin_acc = []
    bin_n = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences <= bins[i + 1]) if i == n_bins - 1 else (confidences >= bins[i]) & (confidences < bins[i + 1])
        cnt = int(np.sum(mask))
        bin_n.append(cnt)
        if cnt > 0:
            bin_acc.append(float(np.mean(correct[mask])))
        else:
            bin_acc.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Ideal")
    ax.scatter(centers, bin_acc, s=[max(n * 2, 20) for n in bin_n], color="#2e7d32", alpha=0.8, label="Binned Accuracy", zorder=3)
    ax.plot(centers, bin_acc, color="#2e7d32", alpha=0.6)

    ax.set_xlabel(r"Classifier Confidence ($\max p_i$)", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title("Classifier Confidence vs. Actual Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(target_path, dpi=200)
    plt.close(fig)
    return target_path


def plot_entropy_distribution(
    traces: List[ExecutionTrace],
    output_path: Path | str = OUTPUT_DIR / "entropy_distribution.png",
) -> Path:
    """Plots distribution of routing entropy for correct vs incorrect predictions."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    correct_entropies = [
        t.entropy for t in traces
        if t.metadata.get("ground_truth") is not None and t.classifier_top1 == t.metadata["ground_truth"]
    ]
    incorrect_entropies = [
        t.entropy for t in traces
        if t.metadata.get("ground_truth") is not None and t.classifier_top1 != t.metadata["ground_truth"]
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 2.5, 25)

    if correct_entropies:
        ax.hist(correct_entropies, bins=bins, alpha=0.6, color="#1b5e20", label=f"Correct Predictions (n={len(correct_entropies)})", density=True)
    if incorrect_entropies:
        ax.hist(incorrect_entropies, bins=bins, alpha=0.6, color="#b71c1c", label=f"Incorrect Predictions (n={len(incorrect_entropies)})", density=True)

    ax.set_xlabel("Shannon Entropy $H(p)$ (bits)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Routing Entropy Distribution: Correct vs. Misclassified Tasks", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(target_path, dpi=200)
    plt.close(fig)
    return target_path


def plot_harness_improvement(
    ablation_df_or_results: Any,
    output_path: Path | str = OUTPUT_DIR / "harness_vs_classifier.png",
) -> Path:
    """Generates comparison bar chart between raw classifier accuracy and harness success rate."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # If dict of results passed
    if isinstance(ablation_df_or_results, dict):
        policies = list(ablation_df_or_results.keys())
        clf_accs = [res.metrics["classifier_accuracy"] * 100 for res in ablation_df_or_results.values()]
        harness_accs = [res.metrics["harness_success_rate"] * 100 for res in ablation_df_or_results.values()]
    else:
        df = ablation_df_or_results
        policies = df["Policy"].tolist()
        clf_accs = [float(x.replace("%", "")) for x in df["Classifier Acc"]]
        harness_accs = [float(x.replace("%", "")) for x in df["Harness Success"]]

    x = np.arange(len(policies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, clf_accs, width, label="Raw Classifier Accuracy", color="#5c6bc0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, harness_accs, width, label="Harness End-to-End Success", color="#2e7d32", alpha=0.85)

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Classifier Prediction Quality vs. Harness System Quality", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(target_path, dpi=200)
    plt.close(fig)
    return target_path


def plot_threshold_tradeoff(
    thresholds: List[float],
    success_rates: List[float],
    avg_attempts: List[float],
    latencies: List[float],
    output_path: Path | str = OUTPUT_DIR / "threshold_tradeoff.png",
) -> Path:
    """Plots tradeoff curves as confidence threshold varies."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#1b5e20"
    ax1.set_xlabel("High-Confidence Escalation Threshold ($\\tau$)", fontsize=11)
    ax1.set_ylabel("Harness Success Rate (%)", color=color1, fontsize=11)
    line1 = ax1.plot(thresholds, [s * 100 for s in success_rates], "o-", color=color1, label="Success Rate", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#0d47a1"
    ax2.set_ylabel("Avg Attempts per Task", color=color2, fontsize=11)
    line2 = ax2.plot(thresholds, avg_attempts, "s--", color=color2, label="Avg Attempts", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left")
    plt.title("Threshold Sweep: Success Rate vs. Attempt Overhead", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(target_path, dpi=200)
    plt.close(fig)
    return target_path


def plot_learning_curve(
    train_sizes: List[int],
    clf_accuracies: List[float],
    harness_successes: List[float],
    output_path: Path | str = OUTPUT_DIR / "learning_curve.png",
) -> Path:
    """Plots learning curve showing classifier accuracy and harness recovery as data scales."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, [a * 100 for a in clf_accuracies], "o-", color="#3949ab", label="Raw Classifier Accuracy", linewidth=2)
    ax.plot(train_sizes, [s * 100 for s in harness_successes], "s-", color="#2e7d32", label="Harness Success Rate", linewidth=2)

    ax.set_xlabel("Number of Training Examples", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Learning Curve: Raw Classifier vs. Harness Resilience", fontsize=12, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(target_path, dpi=200)
    plt.close(fig)
    return target_path
