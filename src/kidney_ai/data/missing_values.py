"""Missing value handlers and imputation strategies."""

from typing import List, Optional
import pandas as pd


class MissingValueHandler:
    """Provides clinical missingness diagnostics and fallback strategies."""

    @staticmethod
    def identify_missing_patterns(df: pd.DataFrame) -> pd.DataFrame:
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df)) * 100
        return pd.DataFrame({"missing_count": null_counts, "missing_percent": null_pct.round(2)})
