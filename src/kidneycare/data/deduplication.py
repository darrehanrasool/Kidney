"""Deduplication and duplicate record analysis for clinical data."""

from typing import Tuple, Dict, Any
import pandas as pd
from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.deduplication")


class Deduplicator:
    """Detects and removes duplicate patient observations."""

    def __init__(self, keep: str = "first"):
        self.keep = keep

    def deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove exact duplicate records.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (deduplicated DataFrame, metadata summary dict).
        """
        initial_count = len(df)
        duplicates_mask = df.duplicated(keep=self.keep)
        duplicates_count = int(duplicates_mask.sum())

        deduped_df = df[~duplicates_mask].copy().reset_index(drop=True)
        final_count = len(deduped_df)

        stats = {
            "initial_record_count": initial_count,
            "duplicate_records_removed": duplicates_count,
            "final_record_count": final_count,
            "deduplication_ratio": round(duplicates_count / initial_count, 4) if initial_count > 0 else 0.0,
        }

        if duplicates_count > 0:
            logger.info(f"Deduplication removed {duplicates_count} duplicate records ({initial_count} -> {final_count})")
        else:
            logger.info("Deduplication check passed: 0 duplicate records found.")

        return deduped_df, stats
