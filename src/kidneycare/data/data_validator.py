"""Data validation and integrity auditing for clinical datasets."""

from pathlib import Path
from typing import Dict, Any, List, Optional
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from kidneycare.core.constants import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    CLASS_MAPPING,
    CLINICAL_RANGES,
)
from kidneycare.core.exceptions import DatasetValidationError, DataLeakageError
from kidneycare.utils.hashing import calculate_sha256
from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.validator")


class DataValidator:
    """Validates schema, types, clinical ranges, and leakage in kidney disease datasets."""

    def __init__(self, clinical_ranges: Optional[Dict[str, Dict[str, Any]]] = None):
        self.clinical_ranges = clinical_ranges or CLINICAL_RANGES

    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate raw file existence, readability, and checksum."""
        if not file_path.exists():
            raise DatasetValidationError(f"Dataset file does not exist: {file_path}")
        if file_path.stat().st_size == 0:
            raise DatasetValidationError(f"Dataset file is empty (0 bytes): {file_path}")

        checksum = calculate_sha256(file_path)
        return {
            "file_name": file_path.name,
            "size_bytes": file_path.stat().st_size,
            "sha256": checksum,
            "is_readable": True,
        }

    def validate_dataset(self, df: Any) -> Dict[str, Any]:
        """Perform comprehensive clinical data validation.

        Args:
            df: Clinical DataFrame.

        Returns:
            Dictionary containing audit report findings.
        """
        if df is None or df.empty:
            raise DatasetValidationError("Dataset is empty or None")

        report: Dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "missing_values": {},
            "duplicate_rows": int(df.duplicated().sum()),
            "range_violations": {},
            "target_distribution": {},
            "leakage_warnings": [],
            "status": "PASS",
        }

        # 1. Target column presence & consistency
        if TARGET_COLUMN not in df.columns:
            raise DatasetValidationError(f"Target column '{TARGET_COLUMN}' missing from dataset")

        clean_target = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
        # Clean potential typographical errors (e.g. 'ckd\t' or 'notckd ')
        clean_target = clean_target.replace({"ckd\t": "ckd", "notckd": "notckd"})
        target_counts = clean_target.value_counts().to_dict()
        report["target_distribution"] = target_counts

        invalid_targets = [val for val in target_counts if val not in CLASS_MAPPING]
        if invalid_targets:
            raise DatasetValidationError(f"Unknown target class labels found: {invalid_targets}")

        # 2. Missing value counts
        missing = df.isna().sum()
        report["missing_values"] = {k: int(v) for k, v in missing.items() if v > 0}

        # 3. Clinical range validation
        for col, rules in self.clinical_ranges.items():
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                min_val = rules.get("min", float("-inf"))
                max_val = rules.get("max", float("inf"))
                violations = ((series < min_val) | (series > max_val)).sum()
                if violations > 0:
                    report["range_violations"][col] = {
                        "violations_count": int(violations),
                        "min_allowed": min_val,
                        "max_allowed": max_val,
                        "unit": rules.get("unit", ""),
                    }

        # 4. Target leakage check (e.g. correlation > 0.999 or identical column)
        for col in df.columns:
            if col == TARGET_COLUMN:
                continue
            # Check if column is a direct proxy for target
            if df[col].astype(str).str.strip().str.lower().equals(clean_target):
                msg = f"Potential target leakage: Feature '{col}' is identical to target!"
                report["leakage_warnings"].append(msg)
                report["status"] = "WARNING_LEAKAGE"
                raise DataLeakageError(msg)

        logger.info(
            f"Dataset validation completed: {len(df)} rows, "
            f"{report['duplicate_rows']} duplicates, "
            f"{len(report['range_violations'])} range violations"
        )
        return report
