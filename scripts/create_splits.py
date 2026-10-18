#!/usr/bin/env python3
"""Dataset splitting script for KidneyCare-AI."""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.data.data_splitting import DataSplitter


def main():
    interim_file = Path("data/interim/clean_clinical_records.csv")
    if not interim_file.exists():
        print("data/interim/clean_clinical_records.csv not found. Running prepare_data.py first...")
        from scripts.prepare_data import main as run_prepare
        run_prepare()

    df = pd.read_csv(interim_file)
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    train_df, val_df, test_df = splitter.split(df)
    splits = splitter.save_splits(train_df, val_df, test_df)
    print(f"Splits successfully created in data/splits/:\n  - Train: {splits['train']}\n  - Val: {splits['val']}\n  - Test: {splits['test']}")


if __name__ == "__main__":
    main()
