"""Schema and data validation utility functions."""

from typing import List, Optional, Any
try:
    import pandas as pd
except ImportError:
    pd = None
from kidneycare.core.exceptions import DatasetValidationError


def validate_dataframe_schema(
    df: Any,
    required_columns: List[str],
    target_column: Optional[str] = None,
) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        required_columns: List of column names that must be present.
        target_column: Optional target column that must also be present.

    Raises:
        DatasetValidationError: If any required column is missing.
    """
    if df is None or df.empty:
        raise DatasetValidationError("DataFrame is empty or None")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required columns in DataFrame: {missing}")

    if target_column and target_column not in df.columns:
        raise DatasetValidationError(f"Target column '{target_column}' is missing from DataFrame")
