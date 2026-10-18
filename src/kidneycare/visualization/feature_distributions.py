"""Feature distribution and target balance visualization."""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from kidneycare.core.constants import TARGET_COLUMN
from kidneycare.utils.file_utils import ensure_dir


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    output_path: Optional[Path] = None,
) -> Path:
    """Plot target class distribution and class balance."""
    counts = df[target_col].astype(str).str.strip().str.lower().value_counts()
    total = len(df)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    colors = ["#d62728" if c == "ckd" else "#2ca02c" for c in counts.index]
    bars = ax.bar(counts.index.str.upper(), counts.values, color=colors, width=0.55, edgecolor="black")

    ax.set_title("Chronic Kidney Disease Target Distribution", fontsize=11, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=10)
    ax.set_xlabel("Clinical Status", fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h}\n({pct:.1f}%)", ha="center", fontsize=9)

    plt.tight_layout()
    out = output_path or Path("visualizations/data_exploration/target_distribution.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    target_col: str = TARGET_COLUMN,
) -> Path:
    """Plot stratified KDE distributions for key numerical biomarkers."""
    cols = features or ["sc", "hemo", "bu", "bgr"]
    cols = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7), dpi=300)
    axes_flat = axes.flatten()

    for i, col in enumerate(cols[:4]):
        ax = axes_flat[i]
        df_clean = df[[col, target_col]].dropna().copy()
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean[target_col] = df_clean[target_col].astype(str).str.strip().str.lower()

        sns.kdeplot(
            data=df_clean,
            x=col,
            hue=target_col,
            common_norm=False,
            fill=True,
            palette={"ckd": "#d62728", "notckd": "#2ca02c"},
            alpha=0.4,
            ax=ax,
        )
        ax.set_title(f"Distribution of `{col}`", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = output_path or Path("visualizations/data_exploration/feature_distributions.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
