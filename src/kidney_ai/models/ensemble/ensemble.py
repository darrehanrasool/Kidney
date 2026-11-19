"""Ensemble meta-models: Soft Voting and Stacking Classifiers."""

from typing import List, Tuple
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator


def build_voting_ensemble(estimators: List[Tuple[str, BaseEstimator]], weights: List[float] = None) -> VotingClassifier:
    """Builds a soft-voting probability ensemble."""
    return VotingClassifier(estimators=estimators, voting="soft", weights=weights)


def build_stacking_ensemble(estimators: List[Tuple[str, BaseEstimator]], final_estimator: BaseEstimator = None) -> StackingClassifier:
    """Builds a meta-learning stacking classifier."""
    if final_estimator is None:
        final_estimator = LogisticRegression()
    return StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)
