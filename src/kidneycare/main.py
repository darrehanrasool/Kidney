"""KidneyCare-AI: Modular Clinical Machine Learning Platform Entrypoint."""

import argparse
import sys
from pathlib import Path

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kidneycare.core.config import load_config
from kidneycare.core.constants import CLINICAL_BOUNDARY_DISCLAIMER
from kidneycare.core.reproducibility import set_seed
from kidneycare.hardware.hardware_detector import HardwareDetector
from kidneycare.data.dataset_downloader import DatasetDownloader
from kidneycare.monitoring.logging_config import setup_logging, get_logger

logger = get_logger("kidneycare")


def run_hardware_audit():
    """Run hardware capability detection."""
    detector = HardwareDetector()
    print(detector.generate_startup_report())


def run_data_download(source: str = "primary"):
    """Download or generate clinical CKD datasets."""
    downloader = DatasetDownloader()
    if source == "synthetic":
        p = downloader.generate_synthetic_benchmark()
        print(f"Generated synthetic benchmark cohort: {p}")
    else:
        p = downloader.download_uci_ckd()
        print(f"Acquired dataset: {p}")


def run_end_to_end_pipeline():
    """Execute complete end-to-end KidneyCare-AI clinical machine learning pipeline."""
    print("=" * 65)
    print("      KidneyCare-AI Clinical Machine Learning Platform")
    print("=" * 65)
    print(CLINICAL_BOUNDARY_DISCLAIMER)
    print("-" * 65)

    # 1. Hardware audit
    detector = HardwareDetector()
    print(detector.generate_startup_report())

    # 2. Config & Seed
    cfg = load_config()
    set_seed(cfg.project.get("environment", {}).get("seed", 42))

    # 3. Data Ingestion & Download
    downloader = DatasetDownloader()
    raw_path = downloader.download_uci_ckd()

    # 4. Validation & Quality Audit
    import pandas as pd
    from kidneycare.data.data_validator import DataValidator
    from kidneycare.data.data_quality import DataQualityAuditor
    from kidneycare.data.deduplication import Deduplicator
    from kidneycare.data.data_splitting import DataSplitter
    from kidneycare.data.preprocessing import ClinicalPreprocessor
    from kidneycare.training.train import train_model
    from kidneycare.evaluation.evaluation import ModelEvaluator
    from kidneycare.evaluation.model_comparison import ModelComparator
    from kidneycare.visualization.visualization import ClinicalVisualizer

    df_raw = pd.read_csv(raw_path)
    validator = DataValidator()
    val_report = validator.validate_dataset(df_raw)

    auditor = DataQualityAuditor()
    auditor.audit(df_raw, dataset_name="UCI-CKD")

    # 5. Deduplication
    deduper = Deduplicator()
    df_clean, _ = deduper.deduplicate(df_raw)

    # 6. Data Splitting
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    df_train, df_val, df_test = splitter.split(df_clean)
    splitter.save_splits(df_train, df_val, df_test)

    # 7. Preprocessing (Fit on train ONLY)
    preprocessor = ClinicalPreprocessor(scaling="standard", feature_engineering=True)
    target_col = "classification"

    X_train_raw = df_train.drop(columns=[target_col])
    y_train = (df_train[target_col].astype(str).str.strip().str.lower() == "ckd").astype(int).values

    X_val_raw = df_val.drop(columns=[target_col])
    y_val = (df_val[target_col].astype(str).str.strip().str.lower() == "ckd").astype(int).values

    X_test_raw = df_test.drop(columns=[target_col])
    y_test = (df_test[target_col].astype(str).str.strip().str.lower() == "ckd").astype(int).values

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names()

    # 8. Train Models
    comparator = ModelComparator()
    models_to_train = ["random_forest", "gradient_boosting", "logistic_regression", "tabular_mlp", "voting_ensemble"]
    trained_models = {}
    test_evaluations = {}

    for m_name in models_to_train:
        model, train_res = train_model(
            model_name=m_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=cfg.raw_configs,
            preprocessor=preprocessor,
            feature_names=feature_names,
        )
        trained_models[m_name] = model

        # Evaluate on test set
        evaluator = ModelEvaluator(optimal_threshold=0.5)
        test_res = evaluator.evaluate_test_set(model, X_test, y_test, model_name=m_name)
        test_evaluations[m_name] = test_res
        comparator.add_model_result(m_name, test_res["metrics"])

    # Save benchmark table
    csv_rep, md_rep = comparator.save_reports()
    print("\n" + "=" * 65)
    print("               FINAL MODEL BENCHMARK RESULTS")
    print("=" * 65)
    print(comparator.to_dataframe().to_string(index=False))

    # 9. Visualizations
    viz = ClinicalVisualizer()
    viz.generate_data_plots(df_raw)
    # Generate evaluation plots for top model
    best_m = "random_forest"
    if best_m in test_evaluations:
        res = test_evaluations[best_m]
        viz.generate_all_evaluation_plots(
            y_true=y_test,
            y_prob=res["y_prob"],
            y_pred=res["y_pred"],
            model_name="RandomForest",
        )
    print("\nVisualizations successfully saved to `visualizations/`.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="KidneyCare-AI Clinical Platform CLI")
    parser.add_argument("command", choices=["hardware", "download", "prepare", "pipeline"], nargs="?", default="pipeline")
    args = parser.parse_args()

    setup_logging()

    if args.command == "hardware":
        run_hardware_audit()
    elif args.command == "download":
        run_data_download()
    elif args.command == "pipeline":
        run_end_to_end_pipeline()


if __name__ == "__main__":
    main()
