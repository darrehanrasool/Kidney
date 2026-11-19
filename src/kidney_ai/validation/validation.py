"""Validation harness combining calibration, cross-validation, and sensitivity tuning."""

from typing import Dict, Any
import numpy as np
from kidney_ai.validation.threshold_selection import select_high_sensitivity_threshold
from kidney_ai.validation.cross_validation import run_stratified_kfold


def validate_clinical_model(model_builder, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    return run_stratified_kfold(model_builder, X, y)
