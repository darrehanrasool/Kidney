"""Support Vector Classifier with probability estimates."""

from typing import Dict, Any, Optional
from sklearn.svm import SVC


def get_svm(params: Optional[Dict[str, Any]] = None) -> SVC:
    """Build and configure Support Vector Classifier.

    Args:
        params: Hyperparameter dictionary.

    Returns:
        Configured SVC estimator with calibrated probability estimates.
    """
    default_params = {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    return SVC(**default_params)
