#!/usr/bin/env python3
"""Master evaluation script benchmarking all trained KidneyCare-AI models on holdout test set."""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kidneycare.core.constants import TARGET_COLUMN
from kidneycare.evaluation.evaluation import ModelEvaluator
from kidneycare.evaluation.model_comparison import ModelComparator
from kidneycare.training.recovery_manager import RecoveryManager
from kidneycare.monitoring.logging_config import setup_logging


def main():
    setup_logging()
    test_path = Path("data/splits/test/test.csv")
    models_dir = Path("outputs/trained_models")

    if not test_path.exists():
        print("Test data not found. Running prepare_data.py first...")
        from scripts.prepare_data import main as run_prepare
        run_prepare()

    prep_path = models_dir / "preprocessor.pkl"
    if not prep_path.exists():
        print("Fitted preprocessor not found. Running train_all.py first...")
        from scripts.train_all import main as run_train
        run_train()

    test_df = pd.read_csv(test_path)
    preprocessor = joblib.load(prep_path)

    target_col = TARGET_COLUMN
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = (test_df[target_col].astype(str).str.strip().str.lower() == "ckd").astype(int).values
    X_test = preprocessor.transform(X_test_raw)

    comparator = ModelComparator()
    evaluator = ModelEvaluator(optimal_threshold=0.5)

    models_to_eval = [
        ("Random Forest", "random_forest.pkl"),
        ("Gradient Boosting", "gradient_boosting.pkl"),
        ("Logistic Regression", "logistic_regression.pkl"),
        ("Support Vector Machine", "svm.pkl"),
    ]

    for display_name, file_name in models_to_eval:
        p = models_dir / file_name
        if p.exists():
            bundle = joblib.load(p)
            model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
            res = evaluator.evaluate_test_set(model, X_test, y_test, model_name=display_name)
            comparator.add_model_result(display_name, res["metrics"], model_type="classical")

    print("\n" + "=" * 80)
    print("                KIDNEYCARE-AI BENCHMARK EVALUATION RESULTS                      ")
    print("=" * 80)
    comp_df = comparator.to_dataframe()
    print(comp_df.to_string(index=False))

    csv_path, md_path = comparator.save_reports()
    print(f"\nEvaluation summary saved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
