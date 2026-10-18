"""SHAP (SHapley Additive exPlanations) for global and local clinical interpretability."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.explainability")


class ClinicalSHAPExplainer:
    """Computes Shapley values for tree models and tabular classifiers."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.explainer: Optional[Any] = None

    def compute_shap_values(
        self,
        model: Any,
        X_sample: np.ndarray,
        max_evals: int = 500,
    ) -> Optional[np.ndarray]:
        """Compute SHAP attribution matrix.

        Args:
            model: Trained estimator.
            X_sample: Feature matrix subset.
            max_evals: Max evaluations for kernel approximation.

        Returns:
            2D or 3D numpy array of SHAP values, or None if SHAP is unavailable.
        """
        if not HAS_SHAP:
            logger.warning("SHAP library is not installed; skipping Shapley value calculation.")
            return None

        try:
            if hasattr(model, "tree_"):
                self.explainer = shap.TreeExplainer(model)
                shap_values = self.explainer.shap_values(X_sample)
            else:
                # KernelExplainer fallback for general models
                background = shap.sample(X_sample, min(50, len(X_sample)))
                self.explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = self.explainer.shap_values(X_sample[:30], nsamples=max_evals)

            # If multi-class list returned, take positive class (CKD = 1)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                return shap_values[1]
            return shap_values

        except Exception as e:
            logger.warning(f"SHAP explanation generation encountered error: {e}")
            return None
