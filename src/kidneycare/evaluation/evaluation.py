"""Hold-out test set evaluation with calibrated reporting."""

from typing import Dict, Any, Optional
import numpy as np

from kidneycare.evaluation.metrics import calculate_clinical_metrics


class ModelEvaluator:
    """Evaluates final fitted models on untouched test partitions."""

    def __init__(self, optimal_threshold: float = 0.5):
        self.optimal_threshold = optimal_threshold

    def evaluate_test_set(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Generate final unbiased test evaluation metrics.

        Args:
            model: Trained estimator.
            X_test: Test features.
            y_test: True test labels.
            model_name: Model identifier.

        Returns:
            Dictionary with test metrics.
        """
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "predict"):
            y_prob = model.predict(X_test)
        else:
            raise ValueError(f"Model {model_name} does not have predict method")

        y_pred = (y_prob >= self.optimal_threshold).astype(int)
        metrics = calculate_clinical_metrics(
            y_test, y_pred, y_prob, threshold=self.optimal_threshold
        )

        return {
            "model_name": model_name,
            "metrics": metrics,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "y_true": y_test,
        }
