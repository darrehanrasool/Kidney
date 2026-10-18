"""Precision-Recall curve visualization."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

from kidneycare.utils.file_utils import ensure_dir


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    model_name: str = "Classifier",
) -> Path:
    """Render Precision-Recall curve with baseline positive prevalence."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.plot(recall, precision, color="#ff7f0e", lw=2.5, label=f"{model_name} (PR-AUC = {ap:.3f})")
    ax.axhline(y=prevalence, color="grey", linestyle="--", lw=1.5, label=f"Prevalence ({prevalence:.1%})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax.set_title(f"Precision-Recall Curve - {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = output_path or Path("visualizations/testing/precision_recall_curve.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
