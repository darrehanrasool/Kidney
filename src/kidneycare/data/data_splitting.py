"""Reproducible, leakage-safe data splitting for training, validation, and testing."""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from kidneycare.core.constants import TARGET_COLUMN
from kidneycare.core.exceptions import DataLeakageError
from kidneycare.utils.file_utils import ensure_dir
from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.splitting")


class DataSplitter:
    """Partitions clinical datasets into train, validation, and test splits."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        stratify_col: Optional[str] = TARGET_COLUMN,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.stratify_col = stratify_col

        total = round(train_ratio + val_ratio + test_ratio, 4)
        if total != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform 3-way stratified partition.

        Args:
            df: Clinical DataFrame.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        stratify = df[self.stratify_col] if self.stratify_col and self.stratify_col in df.columns else None

        # First split into train and temp (val + test)
        temp_ratio = self.val_ratio + self.test_ratio
        df_train, df_temp = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=self.random_state,
            stratify=stratify,
            shuffle=True,
        )

        # Split temp into val and test
        stratify_temp = df_temp[self.stratify_col] if self.stratify_col and self.stratify_col in df_temp.columns else None
        relative_test_ratio = self.test_ratio / temp_ratio

        df_val, df_test = train_test_split(
            df_temp,
            test_size=relative_test_ratio,
            random_state=self.random_state,
            stratify=stratify_temp,
            shuffle=True,
        )

        # Integrity verification: Zero index overlap
        idx_train = set(df_train.index)
        idx_val = set(df_val.index)
        idx_test = set(df_test.index)

        if not idx_train.isdisjoint(idx_val):
            raise DataLeakageError("Data leakage detected: Training and validation partitions share indices!")
        if not idx_train.isdisjoint(idx_test):
            raise DataLeakageError("Data leakage detected: Training and test partitions share indices!")
        if not idx_val.isdisjoint(idx_test):
            raise DataLeakageError("Data leakage detected: Validation and test partitions share indices!")

        logger.info(
            f"Dataset partitioned: Train={len(df_train)} ({self.train_ratio:.0%}), "
            f"Val={len(df_val)} ({self.val_ratio:.0%}), Test={len(df_test)} ({self.test_ratio:.0%})"
        )

        return (
            df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True),
        )

    def save_splits(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        splits_dir: str = "data/splits",
    ) -> Dict[str, Path]:
        """Save train, validation, and test splits into designated directories."""
        base = Path(splits_dir)
        p_train = base / "train" / "train.csv"
        p_val = base / "validation" / "val.csv"
        p_test = base / "test" / "test.csv"

        ensure_dir(p_train.parent)
        ensure_dir(p_val.parent)
        ensure_dir(p_test.parent)

        df_train.to_csv(p_train, index=False)
        df_val.to_csv(p_val, index=False)
        df_test.to_csv(p_test, index=False)

        logger.info(f"Splits saved to {p_train}, {p_val}, {p_test}")
        return {"train": p_train, "val": p_val, "test": p_test}
