#!/usr/bin/env python3
"""Generates full publication figures: ROC curves, confusion matrices, and importance plots."""

import sys
from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidney_ai.visualization.visualization import Visualizer
from kidney_ai.explainability.feature_importance import extract_feature_importances


def main():
    test_path = PROJECT_ROOT / "data" / "splits" / "test" / "test.csv"
    models_dir = PROJECT_ROOT / "outputs" / "trained_models"

    if not test_path.exists() or not (models_dir / "voting_ensemble.joblib").exists():
        print("Models or test split missing. Running pipeline first...")
        from kidney_ai.training.train import train_all_models
        train_all_models()

    df_test = pd.read_csv(test_path)
    y_test = df_test["classification"].values
    X_test_raw = df_test.drop(columns=["classification"])

    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    model = joblib.load(models_dir / "voting_ensemble.joblib")
    rf_model = joblib.load(models_dir / "random_forest.joblib")

    X_test = preprocessor.transform(X_test_raw)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    feature_names = getattr(preprocessor, "feature_names_", [f"f_{i}" for i in range(X_test.shape[1])])
    imp_df = extract_feature_importances(rf_model, feature_names)

    viz = Visualizer(PROJECT_ROOT / "visualizations")
    viz.generate_all(y_test, y_pred, y_prob, imp_df)
    print("All visualizations successfully generated in visualizations/")


if __name__ == "__main__":
    main()
