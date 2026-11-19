#!/usr/bin/env python3
"""Data cleaning, feature derivation, and splitting runner."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidney_ai.data.dataset_downloader import DatasetDownloader
from kidney_ai.data.feature_engineering import FeatureEngineer
from kidney_ai.data.data_splitting import DataSplitter


def main():
    raw_file = PROJECT_ROOT / "data" / "raw" / "clinical" / "chronic_kidney_disease.csv"
    if not raw_file.exists():
        downloader = DatasetDownloader()
        downloader.download_all()

    df = pd.read_csv(raw_file)
    fe = FeatureEngineer()
    df_aug = fe.transform(df)

    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = splitter.split(df_aug)
    splitter.save_splits(train_df, val_df, test_df)

    print("=" * 70)
    print("               KIDNEY-AI DATA PREPARATION COMPLETE              ")
    print("=" * 70)
    print(f"Engineered Features: eGFR, BUN/Creatinine Ratio, Electrolyte Ratio")
    print(f"Train Split:      {len(train_df)} records (data/splits/train/train.csv)")
    print(f"Validation Split: {len(val_df)} records (data/splits/validation/val.csv)")
    print(f"Test Split:       {len(test_df)} records (data/splits/test/test.csv)")
    print("=" * 70)


if __name__ == "__main__":
    main()
