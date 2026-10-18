"""Clinical decision threshold optimization strictly using validation data."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve


class ThresholdOptimizer:
    """Finds optimal classification threshold on validation set to avoid test leakage."""

    def __init__(self, strategy: str = "youden_j"):
        self.strategy = strategy.lower()
        self.optimal_threshold: float = 0.5

    def fit(self, y_true_val: np.ndarray, y_prob_val: np.ndarray) -> float:
        """Find optimal threshold on validation predictions.

        Args:
            y_true_val: Ground truth validation labels.
            y_prob_val: Predicted validation probabilities for positive class.

        Returns:
            Optimal decision threshold float.
        """
        thresholds = np.linspace(0.05, 0.95, 91)

        if self.strategy == "f1":
            best_f1 = -1.0
            best_th = 0.5
            for th in thresholds:
                preds = (y_prob_val >= th).astype(int)
                score = f1_score(y_true_val, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_th = th
            self.optimal_threshold = float(best_th)

        elif self.strategy == "youden_j":
            # Youden's J = Sensitivity + Specificity - 1
            fpr, tpr, roc_thresh = roc_curve(y_true_val, y_prob_val)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            chosen_th = roc_thresh[best_idx]
            # Clip between [0.05, 0.95]
            self.optimal_threshold = float(np.clip(chosen_th, 0.05, 0.95))

        else: # default 0.5
            self.optimal_threshold = 0.5

        return self.optimal_threshold

    def apply(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply fitted optimal threshold to test or inference probabilities."""
        return (y_prob >= self.optimal_threshold).astype(int)
