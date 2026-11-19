"""Evaluator benchmarking all serialized models on holdout clinical test set."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import joblib

from kidney_ai.evaluation.metrics import compute_clinical_metrics


class Evaluator:
    """Computes comprehensive test set metrics across model checkpoints."""

    def __init__(self, models_dir: Path = None, test_path: Path = None):
        self.models_dir = Path(models_dir or "outputs/trained_models")
        self.test_path = Path(test_path or "data/splits/test/test.csv")

    def evaluate_all(self) -> pd.DataFrame:
        if not self.test_path.exists():
            raise FileNotFoundError(f"Holdout test split not found at {self.test_path}")

        df_test = pd.read_csv(self.test_path)
        y_test = df_test["classification"].values
        X_test_raw = df_test.drop(columns=["classification"])

        preprocessor_path = self.models_dir / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError("Preprocessor not found. Please train models first.")
        preprocessor = joblib.load(preprocessor_path)
        X_test = preprocessor.transform(X_test_raw)

        results = []
        for model_file in sorted(self.models_dir.glob("*.joblib")):
            if model_file.name == "preprocessor.joblib":
                continue
            model_name = model_file.stem
            model = joblib.load(model_file)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metrics = compute_clinical_metrics(y_test, y_pred, y_prob)
            metrics["model"] = model_name
            results.append(metrics)

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            cols = ["model", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc", "brier_score"]
            available_cols = [c for c in cols if c in res_df.columns]
            res_df = res_df[available_cols]

        return res_df
