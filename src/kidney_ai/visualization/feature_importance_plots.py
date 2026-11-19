"""Feature importance horizontal bar plot generator."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance(df_importance: pd.DataFrame, save_path: Path, top_n: int = 15) -> None:
    top_df = df_importance.head(top_n).sort_values(by="importance", ascending=True)

    plt.figure(figsize=(9, 6), dpi=300)
    plt.barh(top_df["feature"], top_df["importance"], color="#2b5c8f")
    plt.xlabel("Relative Importance Score")
    plt.title(f"Top {top_n} Clinical Renal Biomarkers", fontsize=12, fontweight="bold")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
