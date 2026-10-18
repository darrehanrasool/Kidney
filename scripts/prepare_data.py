#!/usr/bin/env python3
"""Data preparation, cleaning, feature engineering, and splitting script for KidneyCare-AI."""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.data.dataset_downloader import DatasetDownloader
from kidneycare.data.deduplication import Deduplicator
from kidneycare.data.missing_values import MissingValueHandler
from kidneycare.data.feature_engineering import ClinicalFeatureEngineer
from kidneycare.data.data_splitting import DataSplitter
from kidneycare.data.data_validator import DataValidator


def main():
    clinical_dir = Path("data/raw/clinical")
    csv_files = list(clinical_dir.glob("*.csv"))

    if not csv_files:
        print("No raw clinical datasets found. Downloading / generating benchmark cohort...")
        downloader = DatasetDownloader()
        sample_path = downloader.download_uci_ckd()
        csv_files = [sample_path]

    raw_file = csv_files[0]
    print(f"Loading raw clinical dataset: {raw_file}...")
    df = pd.read_csv(raw_file)

    # 1. Clean column names
    df.columns = [c.strip().lower() for c in df.columns]

    # 2. Validation
    validator = DataValidator()
    validator.validate_dataset(df)

    # 3. Deduplication
    deduper = Deduplicator()
    df_dedup, dedup_stats = deduper.deduplicate(df)
    print(f"Removed {dedup_stats['duplicate_records_removed']} duplicate records.")

    # 4. Save clean interim dataset
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)
    interim_file = interim_dir / "clean_clinical_records.csv"
    df_dedup.to_csv(interim_file, index=False)
    print(f"Clean interim dataset saved to {interim_file} ({df_dedup.shape})")

    # 5. Create train/val/test splits (Leakage-safe)
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    train_df, val_df, test_df = splitter.split(df_dedup)
    splits = splitter.save_splits(train_df, val_df, test_df)
    print(f"Splits successfully created in data/splits/:\n  - Train: {splits['train']}\n  - Val: {splits['val']}\n  - Test: {splits['test']}")


if __name__ == "__main__":
    main()
