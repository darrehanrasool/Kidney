"""Unit tests for clinical preprocessing, missing value handling, and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from kidneycare.data.missing_values import MissingValueHandler
from kidneycare.data.feature_engineering import ClinicalFeatureEngineer
from kidneycare.data.preprocessing import ClinicalPreprocessor
from kidneycare.core.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def test_missing_value_handler():
    df_train = pd.DataFrame({
        "age": [50.0, np.nan, 70.0],
        "sc": [1.0, 2.0, np.nan],
        "htn": ["yes", "no", np.nan],
    })
    handler = MissingValueHandler(numerical_strategy="median", categorical_strategy="most_frequent")
    handler.fit(df_train)

    df_test = pd.DataFrame({
        "age": [np.nan],
        "sc": [np.nan],
        "htn": [np.nan],
    })
    df_imputed = handler.transform(df_test)

    # Median of [50, 70] is 60.0
    assert df_imputed["age"].iloc[0] == 60.0
    # Median of [1.0, 2.0] is 1.5
    assert df_imputed["sc"].iloc[0] == 1.5
    assert df_imputed["htn"].iloc[0] in ["yes", "no"]


def test_clinical_feature_engineering():
    df = pd.DataFrame({
        "age": [60.0, 45.0],
        "sc": [2.0, 1.0],
        "bu": [50.0, 30.0],
        "sod": [138.0, 142.0],
        "pot": [4.8, 4.2],
        "htn": ["yes", "no"],
        "dm": ["yes", "no"],
    })
    fe = ClinicalFeatureEngineer(compute_egfr=True, compute_ratios=True)
    df_rich = fe.transform(df)

    assert "egfr_mdrd" in df_rich.columns
    assert "bun_cr_ratio" in df_rich.columns
    assert "anion_gap" in df_rich.columns
    assert "cv_metabolic_risk" in df_rich.columns

    # BUN/Cr ratio = 50 / 2 = 25.0
    assert df_rich["bun_cr_ratio"].iloc[0] == 25.0
    assert df_rich["cv_metabolic_risk"].iloc[0] == 2
    assert df_rich["cv_metabolic_risk"].iloc[1] == 0


def test_clinical_preprocessor_end_to_end():
    df = pd.DataFrame({
        "age": [48.0, 62.0, 54.0, 70.0],
        "bp": [80.0, 70.0, 90.0, 80.0],
        "bgr": [121.0, 180.0, 110.0, 150.0],
        "bu": [36.0, 50.0, 40.0, 65.0],
        "sc": [1.2, 2.5, 1.0, 3.2],
        "sod": [135.0, 130.0, 140.0, 128.0],
        "pot": [4.5, 5.2, 4.0, 5.8],
        "hemo": [15.4, 11.2, 14.8, 9.8],
        "pcv": [44.0, 32.0, 42.0, 29.0],
        "wc": [7800.0, 9200.0, 7500.0, 11000.0],
        "rc": [5.2, 3.8, 5.0, 3.2],
        "sg": ["1.020", "1.010", "1.025", "1.010"],
        "al": ["0", "2", "0", "3"],
        "su": ["0", "1", "0", "2"],
        "rbc": ["normal", "abnormal", "normal", "abnormal"],
        "pc": ["normal", "normal", "normal", "abnormal"],
        "pcc": ["notpresent", "present", "notpresent", "present"],
        "ba": ["notpresent", "notpresent", "notpresent", "present"],
        "htn": ["no", "yes", "no", "yes"],
        "dm": ["no", "yes", "no", "yes"],
        "cad": ["no", "no", "no", "yes"],
        "appet": ["good", "poor", "good", "poor"],
        "pe": ["no", "yes", "no", "yes"],
        "ane": ["no", "yes", "no", "yes"],
    })

    pre = ClinicalPreprocessor(scaling="standard", feature_engineering=True)
    X = pre.fit_transform(df)

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 4
    assert X.shape[1] > 20
    assert not np.isnan(X).any()
