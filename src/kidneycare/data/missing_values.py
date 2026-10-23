"""Missing value handling with separate training-fit transformations."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from kidneycare.core.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


class MissingValueHandler:
    """Configurable clinical missing-value imputer preventing data leakage."""

    def __init__(
        self,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
    ):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.numerical_impute_values: Dict[str, float] = {}
        self.categorical_impute_values: Dict[str, str] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        """Compute imputation statistics strictly on training partition."""
        # Numerical
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if self.numerical_strategy == "mean":
                val = float(s.mean()) if not s.dropna().empty else 0.0
            else: # default median
                val = float(s.median()) if not s.dropna().empty else 0.0
            self.numerical_impute_values[col] = val

        # Categorical
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        for col in cat_cols:
            s = df[col].dropna().astype(str).str.strip().str.lower()
            if self.categorical_strategy == "most_frequent" and not s.empty:
                val = str(s.mode().iloc[0])
            else:
                val = "missing"
            self.categorical_impute_values[col] = val

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation statistics to target partition."""
        if not self.is_fitted:
            raise RuntimeError("MissingValueHandler must be fitted on training data before transform.")

        df_out = df.copy()

        # Impute numericals
        for col, val in self.numerical_impute_values.items():
            if col in df_out.columns:
                s = pd.to_numeric(df_out[col], errors="coerce")
                df_out[col] = s.fillna(val)

        # Impute categoricals
        for col, val in self.categorical_impute_values.items():
            if col in df_out.columns:
                s = df_out[col].astype(object)
                str_s = s.astype(str).str.strip().str.lower()
                clean_mask = str_s.isin(["nan", "?", "", "none", "null"]) | s.isna()
                df_out[col] = s.mask(clean_mask, val)

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on DataFrame and transform immediately."""
        return self.fit(df).transform(df)
