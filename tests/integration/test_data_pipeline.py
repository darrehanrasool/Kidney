import pytest
from pathlib import Path
from kidney_ai.data.dataset_downloader import DatasetDownloader
from kidney_ai.data.feature_engineering import FeatureEngineer
from kidney_ai.data.data_splitting import DataSplitter
import pandas as pd


def test_data_pipeline_flow(tmp_path):
    downloader = DatasetDownloader(raw_dir=tmp_path / "raw")
    csv_file = downloader.generate_synthetic_ckd_cohort(n_samples=50)
    assert csv_file.exists()

    df = pd.read_csv(csv_file)
    fe = FeatureEngineer()
    df_aug = fe.transform(df)
    assert len(df_aug) == 50

    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    tr, va, te = splitter.split(df_aug)
    assert len(tr) + len(va) + len(te) == 50
