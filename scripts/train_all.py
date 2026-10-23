#!/usr/bin/env python3
"""Master model training script for classical, ensemble, and deep learning CKD models."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.core.config import load_config
from kidneycare.core.constants import TARGET_COLUMN
from kidneycare.core.reproducibility import set_seed
from kidneycare.data.preprocessing import ClinicalPreprocessor
from kidneycare.training.train import train_model
from kidneycare.visualization.training_curves import plot_training_history
from kidneycare.monitoring.logging_config import setup_logging, get_logger
from kidneycare.utils.file_utils import ensure_dir

logger = get_logger("kidneycare.train_all")


def parse_args():
    parser = argparse.ArgumentParser(description="Train all KidneyCare-AI detection models.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--train-data", type=str, default="data/splits/train/train.csv")
    parser.add_argument("--val-data", type=str, default="data/splits/validation/val.csv")
    parser.add_argument("--models", nargs="+", default=["random_forest", "gradient_boosting", "logistic_regression", "svm", "tabular_mlp", "voting_ensemble"])
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file="logs/training/classical_training.log")

    print("=" * 80)
    print("                KIDNEYCARE-AI FULL MODEL TRAINING PIPELINE                       ")
    print("=" * 80)

    # Ensure splits exist
    if not Path(args.train_data).exists() or not Path(args.val_data).exists():
        logger.info("Splits not found. Executing prepare_data first...")
        from scripts.prepare_data import main as run_prepare
        run_prepare()

    cfg = load_config()
    set_seed(cfg.project.get("environment", {}).get("seed", 42))

    df_train = pd.read_csv(args.train_data)
    df_val = pd.read_csv(args.val_data)

    target_col = TARGET_COLUMN
    X_train_raw = df_train.drop(columns=[target_col])
    y_train = (df_train[target_col].astype(str).str.strip().str.lower() == "ckd").astype(int).values

    X_val_raw = df_val.drop(columns=[target_col])
    y_val = (df_val[target_col].astype(str).str.strip().str.lower() == "ckd").astype(int).values

    # Fit preprocessor on train ONLY
    preprocessor = ClinicalPreprocessor(scaling="standard", feature_engineering=True)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    feature_names = preprocessor.get_feature_names()

    trained_models = {}
    for m_name in args.models:
        print(f"\nTraining Model: {m_name.upper()}...")
        logger.info(f"Training Model: {m_name.upper()} on {len(X_train)} samples...")
        model, res = train_model(
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

        # If model has training epoch history (Tabular MLP), plot curves in visualizations/training/
        if isinstance(res, dict) and "history" in res and res["history"].get("history"):
            try:
                vt_dir = Path("visualizations/training")
                ensure_dir(vt_dir)
                curve_path = vt_dir / f"{m_name}_training_curves.png"
                plot_training_history(res["history"], output_path=curve_path, model_name=m_name.replace("_", " ").title())
                logger.info(f"Saved training loss & metric curves to {curve_path}")
            except Exception as e:
                logger.warning(f"Failed to plot training curve for {m_name}: {e}")

    logger.info(f"All {len(trained_models)} models trained and saved to outputs/trained_models and checkpoints/!")
    print(f"\nAll {len(trained_models)} models trained and checkpoints saved in outputs/trained_models and checkpoints/!")


if __name__ == "__main__":
    main()
