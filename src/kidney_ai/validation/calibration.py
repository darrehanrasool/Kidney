"""Probability calibration via Platt scaling / Isotonic regression."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator


def calibrate_probabilities(estimator: BaseEstimator, method: str = "sigmoid", cv: int = 5) -> CalibratedClassifierCV:
    """Wraps an estimator in a CalibratedClassifierCV."""
    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
