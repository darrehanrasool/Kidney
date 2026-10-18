"""Comparative model bar charts and threshold tuning curves."""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

from kidneycare.utils.file_utils import ensure_dir


def plot_model_comparison_bars(
    comparison_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
) -> Path:
    """Render grouped bar chart comparing performance across models."""
    df = comparison_df.copy()
    eval_metrics = metrics or ["Accuracy", "F1-Score", "Sensitivity (Recall)", "Specificity", "ROC-AUC"]
    available_metrics = [m for m in eval_metrics if m in df.columns]

    models = df["Model"].tolist()
    n_models = len(models)
    n_metrics = len(available_metrics)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, metric in enumerate(available_metrics):
        offset = (i - n_metrics / 2) * width + width / 2
        ax.bar(x + offset, df[metric], width, label=metric, color=palette[i % len(palette)])

    ax.set_title("Clinical Model Benchmark Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylim([0.0, 1.08])
    ax.legend(loc="lower right", frameon=True, ncol=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = output_path or Path("visualizations/comparisons/model_comparison_bars.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_threshold_vs_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    optimal_th: float = 0.5,
) -> Path:
    """Plot Sensitivity, Precision, and F1 across decision thresholds."""
    thresholds = np.linspace(0.05, 0.95, 91)
    f1_vals = []
    prec_vals = []
    rec_vals = []

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        f1_vals.append(f1)
        prec_vals.append(prec)
        rec_vals.append(rec)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(thresholds, f1_vals, label="F1-Score", color="#1f77b4", lw=2.2)
    ax.plot(thresholds, prec_vals, label="Precision (PPV)", color="#2ca02c", lw=1.8, linestyle=":")
    ax.plot(thresholds, rec_vals, label="Sensitivity (Recall)", color="#d62728", lw=1.8, linestyle="--")

    ax.axvline(x=optimal_th, color="black", linestyle="-.", lw=1.5, label=f"Selected Threshold ({optimal_th:.2f})")
    ax.set_xlabel("Decision Threshold", fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_title("Performance Trade-offs vs Classification Threshold", fontsize=11, fontweight="bold")
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = output_path or Path("visualizations/validation/threshold_vs_performance.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
