"""Statistical significance tests between model predictions."""

from typing import Dict, Any
import numpy as np
from scipy import stats


def compare_model_significance(errors_model_a: np.ndarray, errors_model_b: np.ndarray) -> Dict[str, float]:
    """Performs Wilcoxon signed-rank and paired t-test on model error distributions."""
    diff = errors_model_a - errors_model_b
    t_stat, t_pval = stats.ttest_rel(errors_model_a, errors_model_b)
    
    try:
        w_stat, w_pval = stats.wilcoxon(diff)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    return {
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_pval),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "is_significant_p05": bool(w_pval < 0.05)
    }
