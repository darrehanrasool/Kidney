"""Classical machine learning models."""

from kidneycare.models.classical.logistic_regression import get_logistic_regression
from kidneycare.models.classical.random_forest import get_random_forest
from kidneycare.models.classical.svm import get_svm
from kidneycare.models.classical.gradient_boosting import get_gradient_boosting

__all__ = [
    "get_logistic_regression",
    "get_random_forest",
    "get_svm",
    "get_gradient_boosting",
]
