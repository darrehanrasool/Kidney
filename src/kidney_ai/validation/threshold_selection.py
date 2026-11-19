"""Safety threshold selection prioritizing clinical sensitivity over raw accuracy."""

from typing import Tuple
import numpy as np
from sklearn.metrics import roc_curve


def select_high_sensitivity_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_sensitivity: float = 0.95) -> Tuple[float, float, float]:
    """Finds decision threshold achieving at least `min_sensitivity` while maximizing specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_idx = np.where(tpr >= min_sensitivity)[0]

    if len(valid_idx) == 0:
        return 0.5, float(tpr[0]), float(1.0 - fpr[0])

    # Pick index with minimum false positive rate (highest specificity) among candidates
    best_idx = valid_idx[np.argmin(fpr[valid_idx])]
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(1.0 - fpr[best_idx])
