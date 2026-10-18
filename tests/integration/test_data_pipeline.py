"""Integration test verifying end-to-end clinical data pipeline from download/synthesis to splitting and preprocessing."""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from kidneycare.data.dataset_downloader import DatasetDownloader
from kidneycare.data.data_validator import DataValidator
from kidneycare.data.deduplication import Deduplicator
from kidneycare.data.data_splitting import DataSplitter
from kidneycare.data.preprocessing import ClinicalPreprocessor
from kidneycare.core.constants import TARGET_COLUMN


def test_full_clinical_data_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = DatasetDownloader(raw_clinical_dir=tmpdir)
        synth_path = downloader.generate_synthetic_benchmark(n_samples=200, seed=42)

        assert synth_path.exists()
        df = pd.read_csv(synth_path)
        assert len(df) == 200

        # Validation
        validator = DataValidator()
        report = validator.validate_dataset(df)
        assert report["status"] == "PASS"

        # Deduplication
        deduper = Deduplicator()
        df_dedup, stats = deduper.deduplicate(df)
        assert len(df_dedup) == 200

        # Splitting
        splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
        train_df, val_df, test_df = splitter.split(df_dedup)
        assert len(train_df) == 140
        assert len(val_df) == 30
        assert len(test_df) == 30

        # Preprocessing (fit on train, transform test)
        preprocessor = ClinicalPreprocessor(scaling="standard", feature_engineering=True)
        X_train_raw = train_df.drop(columns=[TARGET_COLUMN])
        y_train = (train_df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "ckd").astype(int).values

        X_test_raw = test_df.drop(columns=[TARGET_COLUMN])
        y_test = (test_df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "ckd").astype(int).values

        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        assert X_train.shape[0] == 140
        assert X_test.shape[0] == 30
        assert X_train.shape[1] == X_test.shape[1]
