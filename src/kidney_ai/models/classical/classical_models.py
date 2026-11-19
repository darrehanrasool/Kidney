"""Classical machine learning models for renal diagnosis."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from typing import Any, Dict


def build_logistic_regression(params: Dict[str, Any] = None) -> LogisticRegression:
    default_params = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    }
    if params:
        default_params.update(params)
    return LogisticRegression(**default_params)


def build_random_forest(params: Dict[str, Any] = None) -> RandomForestClassifier:
    default_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)
    return RandomForestClassifier(**default_params)


def build_gradient_boosting(params: Dict[str, Any] = None) -> GradientBoostingClassifier:
    default_params = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 5,
        "random_state": 42
    }
    if params:
        default_params.update(params)
    return GradientBoostingClassifier(**default_params)


def build_svm(params: Dict[str, Any] = None) -> SVC:
    default_params = {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42
    }
    if params:
        default_params.update(params)
    return SVC(**default_params)
