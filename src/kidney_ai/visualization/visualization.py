"""Unified visualizer for generating complete publication figure sets."""

from pathlib import Path
import numpy as np
import pandas as pd
from kidney_ai.visualization.confusion_matrix import plot_confusion_matrix
from kidney_ai.visualization.roc_curves import plot_roc_curve
from kidney_ai.visualization.feature_importance_plots import plot_feature_importance


class Visualizer:
    """Orchestrates figure generation across validation splits."""

    def __init__(self, viz_dir: Path = None):
        self.viz_dir = Path(viz_dir or "visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, feature_importance_df: pd.DataFrame = None) -> None:
        plot_confusion_matrix(y_true, y_pred, self.viz_dir / "testing" / "confusion_matrix.png")
        if y_prob is not None:
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            plot_roc_curve(y_true, prob_pos, self.viz_dir / "testing" / "roc_curve.png")
        if feature_importance_df is not None:
            plot_feature_importance(feature_importance_df, self.viz_dir / "explainability" / "feature_importance.png")
