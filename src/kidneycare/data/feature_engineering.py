"""Clinical feature engineering: eGFR calculation, KDIGO staging, and biomarker ratios."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin

from kidneycare.core.constants import KDIGO_STAGES, KDIGO_ALBUMINURIA_STAGES


def compute_ckd_epi_2021(sc: np.ndarray, age: np.ndarray, is_female: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate 2021 CKD-EPI Creatinine Equation (Race-Free standard).

    Formula:
        eGFR = 142 * min(Scr / kappa, 1)**alpha * max(Scr / kappa, 1)**(-1.200) * (0.9938 ** Age) [* 1.012 if female]

    Args:
        sc: Serum Creatinine in mg/dL.
        age: Age in years.
        is_female: Boolean array or None (if None, assumes 50% population average).

    Returns:
        eGFR in mL/min/1.73 m^2.
    """
    sc = np.clip(np.asarray(sc, dtype=float), 0.2, 30.0)
    age = np.clip(np.asarray(age, dtype=float), 10.0, 105.0)

    if is_female is not None:
        fem = np.asarray(is_female, dtype=bool)
        kappa = np.where(fem, 0.7, 0.9)
        alpha = np.where(fem, -0.241, -0.302)
        sex_mult = np.where(fem, 1.012, 1.0)
    else:
        # Population weighted average
        kappa = 0.8
        alpha = -0.271
        sex_mult = 1.006

    ratio = sc / kappa
    term_min = np.minimum(ratio, 1.0) ** alpha
    term_max = np.maximum(ratio, 1.0) ** (-1.200)
    term_age = 0.9938 ** age

    egfr = 142.0 * term_min * term_max * term_age * sex_mult
    return np.round(egfr, 2)


def assign_kdigo_g_stage(egfr: np.ndarray) -> np.ndarray:
    """Map eGFR to KDIGO G1-G5 ordinal stages (0 to 5)."""
    egfr_arr = np.asarray(egfr, dtype=float)
    stages = np.zeros_like(egfr_arr, dtype=int)
    # 0 = G1 (>=90), 1 = G2 (60-89), 2 = G3a (45-59), 3 = G3b (30-44), 4 = G4 (15-29), 5 = G5 (<15)
    stages[egfr_arr >= 90.0] = 0
    stages[(egfr_arr >= 60.0) & (egfr_arr < 90.0)] = 1
    stages[(egfr_arr >= 45.0) & (egfr_arr < 60.0)] = 2
    stages[(egfr_arr >= 30.0) & (egfr_arr < 45.0)] = 3
    stages[(egfr_arr >= 15.0) & (egfr_arr < 30.0)] = 4
    stages[egfr_arr < 15.0] = 5
    return stages


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers physiologically grounded nephrological features for tabular models."""

    def __init__(self, compute_egfr: bool = True, compute_ratios: bool = True):
        self.compute_egfr = compute_egfr
        self.compute_ratios = compute_ratios

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive clinical nephrology biomarkers.

        Args:
            df: Clinical DataFrame with imputed numerical values.

        Returns:
            DataFrame augmented with engineered biomarkers.
        """
        df_out = df.copy()

        # 1. 2021 CKD-EPI & MDRD eGFR
        if self.compute_egfr and "sc" in df_out.columns and "age" in df_out.columns:
            sc = pd.to_numeric(df_out["sc"], errors="coerce").fillna(1.0).clip(lower=0.2, upper=30.0)
            age = pd.to_numeric(df_out["age"], errors="coerce").fillna(50.0).clip(lower=10.0, upper=100.0)

            # 4-variable MDRD formula
            egfr_mdrd = 175.0 * (sc ** -1.154) * (age ** -0.203)
            df_out["egfr_mdrd"] = egfr_mdrd.round(2)

            # Official 2021 CKD-EPI formula
            is_female = None
            if "gender" in df_out.columns or "sex" in df_out.columns:
                col = "gender" if "gender" in df_out.columns else "sex"
                is_female = df_out[col].astype(str).str.lower().str.startswith("f").values

            egfr_epi = compute_ckd_epi_2021(sc.values, age.values, is_female)
            df_out["egfr_ckd_epi_2021"] = egfr_epi
            df_out["kdigo_g_stage"] = assign_kdigo_g_stage(egfr_epi)

        # 2. Blood Urea Nitrogen to Creatinine Ratio (bu / sc)
        if self.compute_ratios and "bu" in df_out.columns and "sc" in df_out.columns:
            bu = pd.to_numeric(df_out["bu"], errors="coerce").fillna(20.0).clip(lower=1.0)
            sc = pd.to_numeric(df_out["sc"], errors="coerce").fillna(1.0).clip(lower=0.1)
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
            df_out["htn_dm_risk"] = htn_yes + dm_yes
            df_out["cv_metabolic_risk"] = df_out["htn_dm_risk"]

        # 5. Proteinuria / Albuminuria risk proxy
        if "al" in df_out.columns:
            al_val = pd.to_numeric(df_out["al"], errors="coerce").fillna(0)
            df_out["albuminuria_stage"] = np.where(al_val == 0, 0, np.where(al_val <= 2, 1, 2))

        return df_out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
