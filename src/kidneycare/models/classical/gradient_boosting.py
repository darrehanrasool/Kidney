"""Gradient Boosting classifier with optional XGBoost/LightGBM acceleration."""

from typing import Dict, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier


def get_gradient_boosting(params: Optional[Dict[str, Any]] = None) -> Any:
    """Build and configure Gradient Boosting Classifier.

    Falls back cleanly to Scikit-Learn's native GradientBoostingClassifier.

    Args:
        params: Hyperparameter dictionary.

    Returns:
        Configured gradient boosting estimator.
    """
    default_params = {
        "learning_rate": 0.05,
        "n_estimators": 150,
        "max_depth": 4,
        "subsample": 0.85,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    # Use sklearn GradientBoostingClassifier by default for maximum reproducibility and zero binary deps
    return GradientBoostingClassifier(**default_params)
