import pandas as pd
import numpy as np
from kidney_ai.data.preprocessing import Preprocessor
from kidney_ai.data.feature_engineering import FeatureEngineer
from kidney_ai.data.data_validator import DataValidator


def test_feature_engineer():
    fe = FeatureEngineer()
    df = pd.DataFrame({
        "age": [50, 60],
        "serum_creatinine": [1.2, 2.5],
        "blood_urea": [40, 90],
        "sodium": [140, 135],
        "potassium": [4.5, 5.0]
    })
    df_aug = fe.transform(df)
    assert "egfr_ckd_epi" in df_aug.columns
    assert "bun_creatinine_ratio" in df_aug.columns
    assert "electrolyte_ratio" in df_aug.columns
    assert df_aug["egfr_ckd_epi"].iloc[0] > df_aug["egfr_ckd_epi"].iloc[1]


def test_preprocessor_fit_transform():
    df = pd.DataFrame({
        "age": [45, 62, np.nan],
        "serum_creatinine": [0.9, 1.8, 1.2],
        "hypertension": ["no", "yes", "no"]
    })
    prep = Preprocessor(numerical_features=["age", "serum_creatinine"], categorical_features=["hypertension"])
    mat = prep.fit_transform(df)
    assert mat.shape[0] == 3
    assert not np.isnan(mat).any()
