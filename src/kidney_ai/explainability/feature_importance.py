"""Extraction of tree feature importances and permutation scores."""

from typing import Dict, List
import numpy as np
import pandas as pd


def extract_feature_importances(model, feature_names: List[str]) -> pd.DataFrame:
    """Extracts normalized feature importance rankings from tree-based estimators."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)
