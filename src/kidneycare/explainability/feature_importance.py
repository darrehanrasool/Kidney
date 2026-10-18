"""Feature importance analysis for classical and ensemble models."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class ClinicalFeatureImportance:
    """Extracts tree-based and permutation feature importance scores."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def get_model_importance(self, model: Any) -> pd.DataFrame:
        """Extract intrinsic model feature importance (e.g. tree impurity or linear weights)."""
        scores = None

        if hasattr(model, "feature_importances_"):
            scores = model.feature_importances_
        elif hasattr(model, "coef_"):
            # Use absolute magnitude of coefficients
            scores = np.abs(model.coef_[0])

        if scores is None or len(scores) != len(self.feature_names):
            # Fallback uniform
            scores = np.ones(len(self.feature_names)) / len(self.feature_names)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": scores,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df

    def compute_permutation_importance(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute model-agnostic permutation feature importance on validation set."""
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="f1",
            n_jobs=-1,
        )

        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_importance": result.importances_mean,
            "std_importance": result.importances_std,
        }).sort_values(by="mean_importance", ascending=False).reset_index(drop=True)

        return df
