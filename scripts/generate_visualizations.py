#!/usr/bin/env python3
"""Visualization generation script producing all 10+ clinical figures and matrices."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.core.constants import TARGET_COLUMN
from kidneycare.visualization.visualization import ClinicalVisualizer
from kidneycare.visualization.feature_importance_plots import plot_feature_importance
from kidneycare.explainability.feature_importance import ClinicalFeatureImportance


def main():
    test_path = Path("data/splits/test/test.csv")
    raw_path = Path("data/raw/clinical/uci_chronic_kidney_disease.csv")
    models_dir = Path("outputs/trained_models")

    if not test_path.exists() or not (models_dir / "random_forest.pkl").exists():
        print("Required test data or models missing. Running evaluate_all first...")
        from scripts.evaluate_all import main as run_eval
        run_eval()

    test_df = pd.read_csv(test_path)
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")

    target_col = TARGET_COLUMN
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = (test_df[target_col].astype(str).str.strip().str.lower() == "ckd").astype(int).values
    X_test = preprocessor.transform(X_test_raw)

    bundle = joblib.load(models_dir / "random_forest.pkl")
    rf = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    probs = rf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    viz = ClinicalVisualizer(base_dir="visualizations")

    # 1. Generate data exploration plots if raw dataset available
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
        viz.generate_data_plots(df_raw)

    # 2. Generate evaluation plots
    saved = viz.generate_all_evaluation_plots(
        y_true=y_test,
        y_prob=probs,
        y_pred=preds,
        model_name="RandomForest",
    )

    # 3. Feature importance plot
    fi_analyzer = ClinicalFeatureImportance(feature_names=preprocessor.get_feature_names())
    imp_df = fi_analyzer.get_model_importance(rf)
    imp_path = plot_feature_importance(imp_df, output_path=Path("visualizations/explainability/feature_importance.png"))
    saved["feature_importance"] = imp_path

    print("=" * 80)
    print("                 KIDNEYCARE-AI VISUALIZATIONS GENERATED                         ")
    print("=" * 80)
    for plot_type, path in saved.items():
        print(f"  - {plot_type.replace('_', ' ').title()}: {path}")


if __name__ == "__main__":
    main()
