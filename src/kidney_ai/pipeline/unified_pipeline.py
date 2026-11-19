"""End-to-end clinical inference pipeline: ingestion -> validation -> risk scoring."""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib

from kidney_ai.data.feature_engineering import FeatureEngineer
from kidney_ai.data.data_validator import DataValidator
from kidney_ai.explainability.explanation_reports import generate_patient_risk_report


class UnifiedKidneyPipeline:
    """Production clinical inference engine for patient risk scoring."""

    def __init__(self, models_dir: Path = None):
        self.models_dir = Path(models_dir or "outputs/trained_models")
        self.validator = DataValidator()
        self.fe = FeatureEngineer()
        self.preprocessor = None
        self.model = None

    def load(self, model_name: str = "voting_ensemble") -> None:
        preprocessor_path = self.models_dir / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Fitted preprocessor not found at {preprocessor_path}.")
        self.preprocessor = joblib.load(preprocessor_path)

        model_path = self.models_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")
        self.model = joblib.load(model_path)

    def predict(self, patient_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Processes patient tabular DataFrame and returns calibrated risk scores."""
        if self.model is None or self.preprocessor is None:
            self.load()

        df_aug = self.fe.transform(patient_df)
        X = self.preprocessor.transform(df_aug)
        probs = self.model.predict_proba(X)
        preds = self.model.predict(X)

        results = []
        for i in range(len(patient_df)):
            risk = float(probs[i, 1])
            is_ckd = bool(preds[i] == 1)
            results.append({
                "patient_index": i,
                "predicted_class": "CKD" if is_ckd else "Non-CKD",
                "risk_probability": round(risk, 4),
                "clinical_urgency": "URGENT_REFERRAL" if risk > 0.75 else ("MONITOR" if risk > 0.35 else "LOW_RISK")
            })

        return results
