"""Statistical significance and confidence intervals for model comparisons."""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats

from kidneycare.evaluation.metrics import bootstrap_confidence_interval


def compare_models_statistical(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    """Perform McNemar test for paired binary classification comparisons.

    Args:
        y_true: Ground truth binary labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.

    Returns:
        Dictionary with contingency table, test statistic, and p-value.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table:
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # McNemar's test statistic with continuity correction
    if (b + c) > 0:
        stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = float(stats.chi2.sf(stat, df=1))
    else:
        stat = 0.0
        p_val = 1.0

    return {
        "mcnemar_statistic": round(float(stat), 4),
        "p_value": round(p_val, 5),
        "significant_p_05": p_val < 0.05,
        "model_a_superior_discordant": b,
        "model_b_superior_discordant": c,
    }
