#!/usr/bin/env python3
"""Data preparation, cleaning, feature engineering, and splitting script for KidneyCare-AI."""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.data.dataset_downloader import DatasetDownloader
from kidneycare.data.deduplication import Deduplicator
from kidneycare.data.missing_values import MissingValueHandler
from kidneycare.data.feature_engineering import ClinicalFeatureEngineer
from kidneycare.data.data_splitting import DataSplitter
from kidneycare.data.data_validator import DataValidator
from kidneycare.data.data_quality import DataQualityAuditor
from kidneycare.monitoring.logging_config import setup_logging, get_logger
from kidneycare.utils.file_utils import ensure_dir

logger = get_logger("kidneycare.data_preparation")


def main():
    setup_logging(log_file="logs/datasets/datasets.log")
    logger.info("Starting Clinical Data Preparation Pipeline...")

    clinical_dir = Path("data/raw/clinical")
    uci_path = clinical_dir / "uci_chronic_kidney_disease.csv"
    synth_path = clinical_dir / "synthetic_ckd_benchmark.csv"

    if uci_path.exists():
        raw_file = uci_path
    elif synth_path.exists():
        raw_file = synth_path
    else:
        csv_files = list(clinical_dir.glob("*.csv"))
        if not csv_files:
            logger.info("No raw clinical datasets found. Downloading / generating benchmark cohort...")
            downloader = DatasetDownloader()
            raw_file = downloader.download_uci_ckd()
        else:
            raw_file = csv_files[0]

    logger.info(f"Loading raw clinical dataset from {raw_file}...")
    print(f"Loading raw clinical dataset: {raw_file}...")
    df = pd.read_csv(raw_file)

    # 1. Clean column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Map ckd_binary to classification if needed
    if "ckd_binary" in df.columns and "classification" not in df.columns:
        df["classification"] = df["ckd_binary"].map({1: "ckd", 0: "notckd", "1": "ckd", "0": "notckd"})

    # 2. Validation
    validator = DataValidator()
    validator.validate_dataset(df)
    logger.info("Data schema and range validation passed.")

    # 3. Deduplication
    deduper = Deduplicator()
    df_dedup, dedup_stats = deduper.deduplicate(df)
    logger.info(f"Removed {dedup_stats['duplicate_records_removed']} duplicate records.")
    print(f"Removed {dedup_stats['duplicate_records_removed']} duplicate records.")

    # 4. Save clean interim dataset
    interim_dir = Path("data/interim")
    ensure_dir(interim_dir)
    interim_file = interim_dir / "clean_clinical_records.csv"
    df_dedup.to_csv(interim_file, index=False)
    logger.info(f"Clean interim dataset saved to {interim_file} ({df_dedup.shape})")
    print(f"Clean interim dataset saved to {interim_file} ({df_dedup.shape})")

    # 5. Data Quality Audit Report (reports/data_quality/)
    auditor = DataQualityAuditor(output_dir="reports/data_quality")
    quality_report = auditor.audit(df_dedup, dataset_name="UCI_CKD_Cohort")
    logger.info("Saved data quality audit report to reports/data_quality/data_quality_uci_ckd_cohort.json and .md")
    print(f"Saved data quality audit report to reports/data_quality/data_quality_uci_ckd_cohort.md")

    # 6. Data Quality Visualization (visualizations/data_quality/)
    vq_dir = Path("visualizations/data_quality")
    ensure_dir(vq_dir)
    missing_pct = (df_dedup.isna().sum() / len(df_dedup) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if not missing_pct.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        missing_pct.plot(kind="bar", color="#e74c3c", ax=ax)
        ax.set_title("Missing Value Rate by Clinical Biomarker (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Missing Percentage (%)")
        ax.set_xlabel("Biomarker")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(vq_dir / "missing_values_audit.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved missingness visualization to {vq_dir / 'missing_values_audit.png'}")

    # 7. Create train/val/test splits (Leakage-safe)
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    train_df, val_df, test_df = splitter.split(df_dedup)
    splits = splitter.save_splits(train_df, val_df, test_df)
    logger.info(f"Splits successfully created in data/splits/: Train={splits['train']}, Val={splits['val']}, Test={splits['test']}")
    print(f"Splits successfully created in data/splits/:\n  - Train: {splits['train']}\n  - Val: {splits['val']}\n  - Test: {splits['test']}")


if __name__ == "__main__":
    main()
