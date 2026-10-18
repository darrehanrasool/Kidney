"""Random Forest classifier for clinical tabular classification."""

from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier


def get_random_forest(params: Optional[Dict[str, Any]] = None) -> RandomForestClassifier:
    """Build and configure RandomForestClassifier.

    Args:
        params: Hyperparameter dictionary.

    Returns:
        Configured RandomForestClassifier estimator.
    """
    default_params = {
        "n_estimators": 150,
        "max_depth": 8,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    return RandomForestClassifier(**default_params)
