"""Inductive Conformal Prediction for distribution-free clinical uncertainty quantification.

Guarantees finite-sample coverage at confidence level (1 - alpha):
    P(y in C(x)) >= 1 - alpha
Returns prediction sets instead of overconfident point predictions. Triggers clinical
abstention when the prediction set contains multiple conflicting stages.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple


class ClinicalConformalPredictor:
    """Computes conformal prediction sets and abstention criteria for kidney staging."""

    def __init__(self, alpha: float = 0.05):
        """Args:
            alpha: Significance level (default 0.05 for 95% coverage guarantee).
        """
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None

    @property
    def is_calibrated(self) -> bool:
        return self.q_hat is not None

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray) -> float:
        """Calibrate threshold on hold-out validation predictions.

        Non-conformity score:
            s_i = 1.0 - P(y_i | x_i)

        Args:
            val_probs: Array of predicted probability distributions (N, K).
            val_labels: Array of true class integer indices (N,).

        Returns:
            Computed q_hat quantile cutoff.
        """
        probs = np.asarray(val_probs, dtype=float)
        labels = np.asarray(val_labels, dtype=int)
        n = len(labels)

        # Non-conformity scores for true labels
        true_class_probs = probs[np.arange(n), labels]
        scores = 1.0 - true_class_probs
        self.calibration_scores = scores

        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self.q_hat

    def predict_set(self, prob_vector: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Construct conformal prediction set for a test instance.

        Prediction set:
            C(x) = { y : P(y | x) >= 1.0 - q_hat }

        Returns:
            Dictionary containing prediction set, coverage guarantee, and abstention flag.
        """
        if self.q_hat is None:
            raise RuntimeError("ConformalPredictor must be calibrated on validation data before predict_set.")

        probs = np.asarray(prob_vector, dtype=float).ravel()
        cutoff = max(0.0, 1.0 - self.q_hat)

        included_indices = np.where(probs >= cutoff)[0]
        # In rare edge cases where no class meets cutoff, include the top argmax
        if len(included_indices) == 0:
            included_indices = np.array([np.argmax(probs)])

        if class_names:
            included_labels = [class_names[idx] for idx in included_indices]
        else:
            included_labels = [int(idx) for idx in included_indices]

        # Clinical abstention rule: if model cannot narrow down to <= 2 stages, abstain
        abstain = len(included_indices) > 2

        return {
            "prediction_set": included_labels,
            "prediction_indices": included_indices.tolist(),
            "set_size": len(included_indices),
            "cutoff_probability": round(cutoff, 4),
            "coverage_guarantee": f"{round((1.0 - self.alpha) * 100, 1)}%",
            "abstain_refer_to_specialist": abstain,
        }
