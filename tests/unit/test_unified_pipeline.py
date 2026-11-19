import pytest
import pandas as pd
from kidney_ai.data.dataset_downloader import DatasetDownloader
from kidney_ai.training.train import train_all_models
from kidney_ai.pipeline.unified_pipeline import UnifiedKidneyPipeline


def test_unified_pipeline():
    # Ensure sample data exists
    downloader = DatasetDownloader()
    downloader.generate_synthetic_ckd_cohort(n_samples=60)

    # Train models
    train_all_models()

    pipeline = UnifiedKidneyPipeline()
    sample_df = pd.DataFrame({
        "age": [55],
        "blood_pressure": [80],
        "specific_gravity": [1.020],
        "albumin": [0],
        "sugar": [0],
        "blood_glucose_random": [100],
        "blood_urea": [30],
        "serum_creatinine": [0.9],
        "sodium": [138],
        "potassium": [4.5],
        "hemoglobin": [14.0],
        "packed_cell_volume": [42],
        "white_blood_cell_count": [7000],
        "red_blood_cell_count": [5.0],
        "hypertension": ["no"],
        "diabetes_mellitus": ["no"],
        "coronary_artery_disease": ["no"],
        "appetite": ["good"],
        "pedal_edema": ["no"],
        "anemia": ["no"]
    })

    res = pipeline.predict(sample_df)
    assert len(res) == 1
    assert "predicted_class" in res[0]
    assert "risk_probability" in res[0]
    assert 0.0 <= res[0]["risk_probability"] <= 1.0
