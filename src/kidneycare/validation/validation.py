"""High-level clinical validation pipeline."""

from typing import Dict, Any, Optional
import numpy as np

from kidneycare.evaluation.metrics import calculate_clinical_metrics
from kidneycare.validation.threshold_selection import ThresholdOptimizer
from kidneycare.validation.calibration import ModelCalibrator


class ClinicalValidator:
    """Manages validation, threshold tuning, and probability calibration."""

    def __init__(self, optimize_threshold: bool = True, calibrate_probabilities: bool = True):
        self.optimize_threshold = optimize_threshold
        self.calibrate_probabilities = calibrate_probabilities
        self.threshold_optimizer = ThresholdOptimizer(strategy="youden_j")
        self.calibrator = ModelCalibrator(method="sigmoid")

    def validate(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform validation, threshold optimization, and initial calibration audit."""
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "predict"):
            y_prob = model.predict(X_val)
        else:
            raise ValueError("Model does not expose predict or predict_proba")

        if self.optimize_threshold:
            optimal_th = self.threshold_optimizer.fit(y_val, y_prob)
            y_pred = self.threshold_optimizer.apply(y_prob)
        else:
            optimal_th = 0.5
            y_pred = (y_prob >= 0.5).astype(int)

        metrics = calculate_clinical_metrics(y_val, y_pred, y_prob, threshold=optimal_th)
        return {
            "metrics": metrics,
            "optimal_threshold": optimal_th,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
