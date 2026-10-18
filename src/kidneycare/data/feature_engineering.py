"""Clinical feature engineering: eGFR calculation and biomarker ratios."""

import numpy as np
import pandas as pd
from typing import Optional


class ClinicalFeatureEngineer:
    """Engineers physiologically grounded nephrological features."""

    def __init__(self, compute_egfr: bool = True, compute_ratios: bool = True):
        self.compute_egfr = compute_egfr
        self.compute_ratios = compute_ratios

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive clinical nephrology biomarkers.

        Args:
            df: Clinical DataFrame with imputed numerical values.

        Returns:
            DataFrame augmented with engineered biomarkers.
        """
        df_out = df.copy()

        # 1. Estimated Glomerular Filtration Rate (MDRD equation)
        # eGFR = 175 * (sc ** -1.154) * (age ** -0.203)
        if self.compute_egfr and "sc" in df_out.columns and "age" in df_out.columns:
            sc = pd.to_numeric(df_out["sc"], errors="coerce").clip(lower=0.2, upper=30.0)
            age = pd.to_numeric(df_out["age"], errors="coerce").clip(lower=10.0, upper=100.0)
            egfr = 175.0 * (sc ** -1.154) * (age ** -0.203)
            df_out["egfr_mdrd"] = egfr.round(2)

        # 2. Blood Urea Nitrogen to Creatinine Ratio (bu / sc)
        if self.compute_ratios and "bu" in df_out.columns and "sc" in df_out.columns:
            bu = pd.to_numeric(df_out["bu"], errors="coerce").clip(lower=1.0)
            sc = pd.to_numeric(df_out["sc"], errors="coerce").clip(lower=0.1)
            df_out["bun_cr_ratio"] = (bu / sc).round(2)

        # 3. Electrolyte Imbalance Index (Sodium & Potassium deviation)
        if "sod" in df_out.columns and "pot" in df_out.columns:
            sod = pd.to_numeric(df_out["sod"], errors="coerce").fillna(140.0)
            pot = pd.to_numeric(df_out["pot"], errors="coerce").fillna(4.5)
            # Distance from normal homeostatic midpoints: Na 140 mEq/L, K 4.5 mEq/L
            df_out["anion_gap"] = ((sod - 140.0).abs() + 8.0 * (pot - 4.5).abs()).round(2)

        # 4. Metabolic-Cardiovascular Combined Risk Flag (htn + dm)
        if "htn" in df_out.columns and "dm" in df_out.columns:
            htn_yes = df_out["htn"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
            dm_yes = df_out["dm"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
            df_out["cv_metabolic_risk"] = htn_yes + dm_yes

        return df_out
