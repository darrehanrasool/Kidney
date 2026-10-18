"""Feature correlation heatmap visualization."""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from kidneycare.core.constants import NUMERICAL_FEATURES
from kidneycare.utils.file_utils import ensure_dir


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    features: Optional[List[str]] = None,
) -> Path:
    """Render correlation matrix heatmap for clinical biomarkers."""
    cols = features or [c for c in NUMERICAL_FEATURES if c in df.columns]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Clinical Biomarker Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    out = output_path or Path("visualizations/data_exploration/correlation_heatmap.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
