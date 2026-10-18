"""Epoch-level training and validation curves."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

from kidneycare.utils.file_utils import ensure_dir


def plot_training_history(
    history: Dict[str, Any],
    output_path: Optional[Path] = None,
    model_name: str = "Tabular MLP",
) -> Path:
    """Render dual-panel training curves (Loss and F1-Score)."""
    records = history.get("history", [])
    if not records:
        raise ValueError("History contains no epoch records")

    epochs = [r["epoch"] for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_loss = [r["val_loss"] for r in records]
    val_f1 = [r.get("val_metrics", {}).get("f1", 0.0) for r in records]
    train_f1 = [r.get("train_metrics", {}).get("f1", 0.0) for r in records]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5), dpi=300)

    # Panel 1: Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=2.0)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#d62728", lw=2.0, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title(f"{model_name} - Loss Trajectory", fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: F1 Metric
    ax2.plot(epochs, train_f1, label="Train F1", color="#1f77b4", lw=2.0)
    ax2.plot(epochs, val_f1, label="Val F1", color="#2ca02c", lw=2.0, linestyle="--")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Macro F1-Score", fontsize=10)
    ax2.set_title(f"{model_name} - F1 Metric Trajectory", fontsize=11, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = output_path or Path("visualizations/training/training_curves.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
