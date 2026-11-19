"""Data quality assessment, completeness ratios, and anomaly reporting."""

from typing import Dict, Any
import pandas as pd


class DataQualityAuditor:
    """Computes completeness, missingness profiles, and column statistics."""

    @staticmethod
    def audit(df: pd.DataFrame) -> Dict[str, Any]:
        total_rows = len(df)
        total_cells = df.size
        missing_cells = int(df.isna().sum().sum())
        completeness = round(100.0 * (1.0 - missing_cells / total_cells), 2) if total_cells > 0 else 0.0

        missing_per_col = df.isna().sum().to_dict()
        missing_pct_per_col = {k: round(v / total_rows * 100, 2) for k, v in missing_per_col.items()}

        return {
            "total_records": total_rows,
            "total_columns": len(df.columns),
            "missing_cells": missing_cells,
            "overall_completeness_pct": completeness,
            "missing_per_column": missing_per_col,
            "missing_pct_per_column": missing_pct_per_col,
        }
