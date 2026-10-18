"""Feature importance bar chart visualization."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from kidneycare.utils.file_utils import ensure_dir


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n: int = 15,
    title: str = "Top Clinical Feature Importances",
) -> Path:
    """Render horizontal bar chart of feature attributions."""
    df_top = importance_df.head(top_n).copy()
    val_col = "importance" if "importance" in df_top.columns else "mean_importance"
    # Sort ascending so top features appear at the top of the chart
    df_top = df_top.sort_values(by=val_col, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.barh(df_top["feature"], df_top[val_col], color="#1f77b4", edgecolor="none", height=0.65)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.set_ylabel("Clinical Biomarker / Feature", fontsize=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Annotate values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out = output_path or Path("visualizations/explainability/feature_importance.png")
    ensure_dir(out.parent)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
