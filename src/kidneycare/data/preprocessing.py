"""Clinical preprocessing pipeline with strict leakage prevention."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder

from kidneycare.core.constants import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    CLASS_MAPPING,
    CLINICAL_RANGES,
)
from kidneycare.data.missing_values import MissingValueHandler
from kidneycare.data.feature_engineering import ClinicalFeatureEngineer
from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.preprocessing")


class ClinicalPreprocessor:
    """End-to-end scikit-learn compatible preprocessing pipeline for clinical tabular data."""

    def __init__(
        self,
        scaling: str = "standard",
        impute_numerical: str = "median",
        impute_categorical: str = "most_frequent",
        feature_engineering: bool = True,
        clip_outliers: bool = True,
    ):
        self.scaling = scaling
        self.impute_numerical = impute_numerical
        self.impute_categorical = impute_categorical
        self.feature_engineering = feature_engineering
        self.clip_outliers = clip_outliers

        self.missing_handler = MissingValueHandler(
            numerical_strategy=impute_numerical,
            categorical_strategy=impute_categorical,
        )
        self.feature_engineer = ClinicalFeatureEngineer()

        if scaling == "robust":
            self.scaler = RobustScaler()
        elif scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.is_fitted = False

        self.fitted_num_cols: List[str] = []
        self.fitted_cat_cols: List[str] = []
        self.encoded_feature_names: List[str] = []
        self.transformation_log: List[Dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ClinicalPreprocessor":
        """Fit preprocessing transformations strictly on the training partition.

        Args:
            X: Training features DataFrame.
            y: Optional training target series.

        Returns:
            Fitted preprocessor.
        """
        df = X.copy()
        initial_records = len(df)

        # 1. Fit missing value handler
        self.missing_handler.fit(df)
        df_imputed = self.missing_handler.transform(df)

        # 2. Derive clinical features
        if self.feature_engineering:
            df_fe = self.feature_engineer.transform(df_imputed)
        else:
            df_fe = df_imputed

        # 3. Separate numerical and categorical columns
        self.fitted_num_cols = [
            c for c in df_fe.columns
            if c in NUMERICAL_FEATURES or c in ["egfr_mdrd", "bun_cr_ratio", "anion_gap", "cv_metabolic_risk"]
        ]
        self.fitted_cat_cols = [c for c in df_fe.columns if c in CATEGORICAL_FEATURES]

        # 4. Outlier clipping bounds on training data
        if self.clip_outliers:
            df_fe[self.fitted_num_cols] = df_fe[self.fitted_num_cols].clip(lower=-1e6, upper=1e6)

        # 5. Fit scaler on numericals
        if self.fitted_num_cols:
            self.scaler.fit(df_fe[self.fitted_num_cols])

        # 6. Fit encoder on categoricals
        if self.fitted_cat_cols:
            self.encoder.fit(df_fe[self.fitted_cat_cols].astype(str))
            cat_feature_names = list(self.encoder.get_feature_names_out(self.fitted_cat_cols))
        else:
            cat_feature_names = []

        self.encoded_feature_names = self.fitted_num_cols + cat_feature_names
        self.is_fitted = True

        self.transformation_log.append({
            "stage": "fit",
            "input_records": initial_records,
            "fitted_num_features": len(self.fitted_num_cols),
            "fitted_cat_features": len(self.fitted_cat_cols),
            "total_output_features": len(self.encoded_feature_names),
        })

        logger.info(
            f"Fitted ClinicalPreprocessor: {len(self.fitted_num_cols)} numerical features, "
            f"{len(self.fitted_cat_cols)} categorical features -> {len(self.encoded_feature_names)} encoded columns"
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformations to a feature set (train, val, or test).

        Args:
            X: Input DataFrame.

        Returns:
            Preprocessed 2D numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted before transforming.")

        df = X.copy()

        # 1. Missing value imputation
        df_imputed = self.missing_handler.transform(df)

        # 2. Clinical feature engineering
        if self.feature_engineering:
            df_fe = self.feature_engineer.transform(df_imputed)
        else:
            df_fe = df_imputed

        # 3. Numerical transformation
        if self.fitted_num_cols:
            X_num = self.scaler.transform(df_fe[self.fitted_num_cols])
        else:
            X_num = np.empty((len(df_fe), 0))

        # 4. Categorical transformation
        if self.fitted_cat_cols:
            X_cat = self.encoder.transform(df_fe[self.fitted_cat_cols].astype(str))
        else:
            X_cat = np.empty((len(df_fe), 0))

        # 5. Concatenate
        X_out = np.hstack([X_num, X_cat])
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform simultaneously on training partition."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """Return list of transformed feature names."""
        return self.encoded_feature_names
