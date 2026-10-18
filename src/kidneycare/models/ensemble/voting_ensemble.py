"""Voting Ensemble combining top classical classifiers."""

from typing import List, Tuple, Optional, Dict, Any
from sklearn.ensemble import VotingClassifier

from kidneycare.models.classical.logistic_regression import get_logistic_regression
from kidneycare.models.classical.random_forest import get_random_forest
from kidneycare.models.classical.gradient_boosting import get_gradient_boosting


def get_voting_ensemble(
    estimators: Optional[List[Tuple[str, Any]]] = None,
    voting: str = "soft",
    weights: Optional[List[float]] = None,
) -> VotingClassifier:
    """Build a calibrated VotingClassifier combining multiple diverse estimators.

    Args:
        estimators: Optional list of (name, estimator) tuples.
        voting: 'soft' for predicted probabilities, or 'hard'.
        weights: Optional sequence of weights.

    Returns:
        Configured VotingClassifier.
    """
    if estimators is None:
        estimators = [
            ("rf", get_random_forest()),
            ("gb", get_gradient_boosting()),
            ("lr", get_logistic_regression()),
        ]

    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1,
    )
