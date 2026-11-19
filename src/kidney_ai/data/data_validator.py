"""Validation of clinical records against physiological bounds and schema integrity."""

from typing import Dict, List, Tuple
import pandas as pd
from kidney_ai.core.constants import PHYSIOLOGICAL_BOUNDS, TARGET_MAPPING
from kidney_ai.core.exceptions import ClinicalDataValidationError


class DataValidator:
    """Verifies clinical bounds, valid categories, and data completeness."""

    def __init__(self, bounds: Dict[str, Tuple[float, float]] = None):
        self.bounds = bounds or PHYSIOLOGICAL_BOUNDS

    def validate_schema(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Verifies presence of required clinical features."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ClinicalDataValidationError(f"Missing mandatory clinical features: {missing}")
        return True

    def audit_bounds(self, df: pd.DataFrame) -> Dict[str, int]:
        """Flags record counts exceeding biological physiological limits."""
        anomalies = {}
        for col, (min_val, max_val) in self.bounds.items():
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                out_of_bounds = series[(series < min_val) | (series > max_val)]
                if len(out_of_bounds) > 0:
                    anomalies[col] = len(out_of_bounds)
        return anomalies

    def clean_target(self, series: pd.Series) -> pd.Series:
        """Standardizes target variable into binary 0/1 labels."""
        cleaned = series.astype(str).str.strip().str.lower().map(TARGET_MAPPING)
        if cleaned.isna().any():
            raise ClinicalDataValidationError("Target column contains unrecognized class labels.")
        return cleaned.astype(int)
