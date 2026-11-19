"""Stratified clinical data splitting for reproducible evaluation."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Partitions clinical cohorts into train, validation, and holdout test sets."""

    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, df: pd.DataFrame, target_col: str = "classification") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        stratify = df[target_col] if target_col in df.columns else None

        # First split off holdout test set
        test_size = self.test_ratio
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=self.seed, stratify=stratify
        )

        # Split remaining into train and validation
        val_rel_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        stratify_tv = df_train_val[target_col] if target_col in df_train_val.columns else None

        df_train, df_val = train_test_split(
            df_train_val, test_size=val_rel_size, random_state=self.seed, stratify=stratify_tv
        )

        return df_train, df_val, df_test

    def save_splits(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, splits_dir: Optional[Path] = None) -> None:
        if splits_dir is None:
            splits_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "splits"

        splits_dir = Path(splits_dir)
        (splits_dir / "train").mkdir(parents=True, exist_ok=True)
        (splits_dir / "validation").mkdir(parents=True, exist_ok=True)
        (splits_dir / "test").mkdir(parents=True, exist_ok=True)

        df_train.to_csv(splits_dir / "train" / "train.csv", index=False)
        df_val.to_csv(splits_dir / "validation" / "val.csv", index=False)
        df_test.to_csv(splits_dir / "test" / "test.csv", index=False)
