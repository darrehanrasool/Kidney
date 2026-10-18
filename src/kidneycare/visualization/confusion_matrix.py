"""Confusion matrix visualization."""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from kidneycare.utils.file_utils import ensure_dir


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> Path:
    """Render and save normalized confusion matrix with raw counts annotated."""
    classes = class_names or ["NotCKD", "CKD"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Format annotations: "Count\n(Percent%)"
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=10, labelpad=8)
    ax.set_ylabel("True Diagnosis", fontsize=10, labelpad=8)
    plt.tight_layout()

    out = output_path or Path("visualizations/testing/confusion_matrix.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
