"""Calibration plots and reliability diagrams."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from kidneycare.evaluation.metrics import calculate_expected_calibration_error
from kidneycare.utils.file_utils import ensure_dir


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    n_bins: int = 10,
    model_name: str = "Classifier",
) -> Path:
    """Render calibration curve and confidence histogram."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 7), dpi=300, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Reliability curve
    ax1.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
    ax1.plot(prob_pred, prob_true, marker="o", lw=2.0, color="#1f77b4", label=f"{model_name} (ECE={ece:.3f})")
    ax1.set_ylabel("Empirical Probability (Fraction of Positives)", fontsize=10)
    ax1.set_title(f"Calibration Diagram - {model_name}", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Histogram of predicted probabilities
    ax2.hist(y_prob, range=(0, 1), bins=n_bins, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = output_path or Path("visualizations/validation/calibration_curve.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
