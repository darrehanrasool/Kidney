#!/usr/bin/env python3
"""Unified pipeline demonstration running clinical risk prediction."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidney_ai.pipeline.unified_pipeline import UnifiedKidneyPipeline


def main():
    test_path = PROJECT_ROOT / "data" / "splits" / "test" / "test.csv"
    if not test_path.exists():
        from kidney_ai.training.train import train_all_models
        train_all_models()

    df_test = pd.read_csv(test_path)
    sample_patients = df_test.head(5).drop(columns=["classification"])

    pipeline = UnifiedKidneyPipeline()
    results = pipeline.predict(sample_patients)

    print("=" * 70)
    print("             KIDNEY-AI UNIFIED CLINICAL RISK INFERENCE          ")
    print("=" * 70)
    for r in results:
        print(f"Patient #{r['patient_index']}: Prediction = {r['predicted_class']:<8} | "
              f"Risk Score = {r['risk_probability']:.4f} | Urgency: {r['clinical_urgency']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
