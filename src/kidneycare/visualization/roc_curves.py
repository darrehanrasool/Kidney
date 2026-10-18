"""ROC curve plotting."""

from pathlib import Path
from typing import Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from kidneycare.utils.file_utils import ensure_dir


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    model_name: str = "Classifier",
) -> Path:
    """Render single model Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = output_path or Path("visualizations/testing/roc_curve.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_multi_model_roc(
    models_predictions: Dict[str, Dict[str, np.ndarray]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot overlaid ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (m_name, data) in enumerate(models_predictions.items()):
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{m_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title("Multi-Model ROC Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = output_path or Path("visualizations/comparisons/multi_model_roc.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
