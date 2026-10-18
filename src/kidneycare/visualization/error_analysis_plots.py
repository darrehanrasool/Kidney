"""Prediction error and residual analysis plots."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from kidneycare.utils.file_utils import ensure_dir


def plot_prediction_error_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    threshold: float = 0.5,
) -> Path:
    """Analyze distribution of predicted probabilities for correct vs incorrect cases."""
    y_pred = (y_prob >= threshold).astype(int)
    correct = (y_pred == y_true)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Histogram of probabilities for correct vs incorrect
    ax.hist(
        y_prob[correct],
        bins=20,
        alpha=0.6,
        color="#2ca02c",
        label=f"Correctly Classified (n={int(np.sum(correct))})",
        edgecolor="black",
    )
    ax.hist(
        y_prob[~correct],
        bins=20,
        alpha=0.75,
        color="#d62728",
        label=f"Misclassified (n={int(np.sum(~correct))})",
        edgecolor="black",
    )

    ax.axvline(x=threshold, color="black", linestyle="--", lw=1.5, label=f"Decision Threshold ({threshold:.2f})")
    ax.set_xlabel("Predicted Probability of CKD", fontsize=10)
    ax.set_ylabel("Patient Count", fontsize=10)
    ax.set_title("Diagnostic Error Analysis: Probability Density", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = output_path or Path("visualizations/testing/prediction_error_analysis.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
