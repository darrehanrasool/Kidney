"""Unified visualization orchestrator for KidneyCare-AI."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from kidneycare.visualization.confusion_matrix import plot_confusion_matrix
from kidneycare.visualization.roc_curves import plot_roc_curve, plot_multi_model_roc
from kidneycare.visualization.precision_recall import plot_precision_recall_curve
from kidneycare.visualization.calibration_plots import plot_calibration_curve
from kidneycare.visualization.feature_importance_plots import plot_feature_importance
from kidneycare.visualization.training_curves import plot_training_history
from kidneycare.visualization.correlation_heatmap import plot_correlation_heatmap
from kidneycare.visualization.feature_distributions import plot_feature_distributions, plot_target_distribution
from kidneycare.visualization.error_analysis_plots import plot_prediction_error_analysis
from kidneycare.visualization.model_comparison_plots import plot_model_comparison_bars, plot_threshold_vs_performance


class ClinicalVisualizer:
    """Coordinates generation of all publication-grade clinical visualizations."""

    def __init__(self, base_dir: str = "visualizations"):
        self.base_dir = Path(base_dir)

    def generate_all_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "RandomForest",
        optimal_th: float = 0.5,
    ) -> Dict[str, Path]:
        """Generate evaluation visualizations for a trained model."""
        return {
            "confusion_matrix": plot_confusion_matrix(
                y_true, y_pred, output_path=self.base_dir / "testing" / f"{model_name}_confusion_matrix.png"
            ),
            "roc_curve": plot_roc_curve(
                y_true, y_prob, output_path=self.base_dir / "testing" / f"{model_name}_roc_curve.png", model_name=model_name
            ),
            "precision_recall": plot_precision_recall_curve(
                y_true, y_prob, output_path=self.base_dir / "testing" / f"{model_name}_pr_curve.png", model_name=model_name
            ),
            "calibration": plot_calibration_curve(
                y_true, y_prob, output_path=self.base_dir / "validation" / f"{model_name}_calibration.png", model_name=model_name
            ),
            "error_analysis": plot_prediction_error_analysis(
                y_true, y_prob, output_path=self.base_dir / "testing" / f"{model_name}_error_analysis.png", threshold=optimal_th
            ),
            "threshold_performance": plot_threshold_vs_performance(
                y_true, y_prob, output_path=self.base_dir / "validation" / f"{model_name}_threshold_performance.png", optimal_th=optimal_th
            ),
        }

    def generate_data_plots(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Generate EDA and distribution plots for raw or interim datasets."""
        return {
            "target_distribution": plot_target_distribution(
                df, output_path=self.base_dir / "data_exploration" / "target_distribution.png"
            ),
            "feature_distributions": plot_feature_distributions(
                df, output_path=self.base_dir / "data_exploration" / "biomarker_distributions.png"
            ),
            "correlation_heatmap": plot_correlation_heatmap(
                df, output_path=self.base_dir / "data_exploration" / "biomarker_correlation_matrix.png"
            ),
        }
