"""Probability calibration using Platt scaling and Isotonic regression."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from kidneycare.evaluation.metrics import calculate_expected_calibration_error


class ModelCalibrator:
    """Evaluates and improves probability calibration for clinical risk assessment."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method # 'sigmoid' (Platt) or 'isotonic'
        self.calibrated_model: Optional[CalibratedClassifierCV] = None

    def fit_calibrator(self, base_estimator: Any, X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Calibrate pre-fitted estimator using hold-out validation partition."""
        calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.method,
            cv="prefit",
        )
        calibrator.fit(X_val, y_val)
        self.calibrated_model = calibrator
        return calibrator

    def get_calibration_curve_data(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute reliability curve and Expected Calibration Error."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        return prob_true, prob_pred, ece
