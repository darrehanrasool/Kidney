"""Data preprocessing pipeline: imputation, encoding, and robust feature scaling."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class Preprocessor:
    """Preprocesses mixed numerical and categorical clinical records."""

    def __init__(self, numerical_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        num_cols = [c for c in self.numerical_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]

        if num_cols:
            self.num_imputer.fit(df[num_cols])
            imputed_nums = self.num_imputer.transform(df[num_cols])
            self.scaler.fit(imputed_nums)

        if cat_cols:
            self.cat_imputer.fit(df[cat_cols])
            imputed_cats = self.cat_imputer.transform(df[cat_cols])
            self.encoder.fit(imputed_cats)

        self.feature_names_ = num_cols.copy()
        if cat_cols and hasattr(self.encoder, "get_feature_names_out"):
            self.feature_names_.extend(self.encoder.get_feature_names_out(cat_cols).tolist())

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num_cols = [c for c in self.numerical_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]

        parts = []
        if num_cols:
            imputed_nums = self.num_imputer.transform(df[num_cols])
            scaled_nums = self.scaler.transform(imputed_nums)
            parts.append(scaled_nums)

        if cat_cols:
            imputed_cats = self.cat_imputer.transform(df[cat_cols])
            encoded_cats = self.encoder.transform(imputed_cats)
            parts.append(encoded_cats)

        if not parts:
            return np.empty((len(df), 0))

        return np.hstack(parts)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
