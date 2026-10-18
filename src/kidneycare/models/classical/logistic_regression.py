"""Logistic Regression baseline classifier."""

from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression


def get_logistic_regression(params: Optional[Dict[str, Any]] = None) -> LogisticRegression:
    """Build and configure LogisticRegression estimator.

    Args:
        params: Optional hyperparameter overrides.

    Returns:
        Configured LogisticRegression estimator.
    """
    default_params = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    return LogisticRegression(**default_params)
