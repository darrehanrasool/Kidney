"""Clinical feature engineering: eGFR, BUN/Creatinine ratios, and metabolic risk scores."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Computes evidence-based nephrology biomarkers and derived indices."""

    @staticmethod
    def estimate_egfr_ckd_epi(serum_creatinine: pd.Series, age: pd.Series) -> pd.Series:
        """Approximates eGFR using 2021 CKD-EPI formula (mL/min/1.73m²)."""
        scr = pd.to_numeric(serum_creatinine, errors="coerce").clip(lower=0.2)
        patient_age = pd.to_numeric(age, errors="coerce").clip(lower=18, upper=100)
        
        # Gender-neutral / average calibration baseline
        kappa = 0.8
        alpha = -0.27
        ratio = scr / kappa
        
        egfr = 142.0 * np.minimum(ratio, 1.0)**alpha * np.maximum(ratio, 1.0)**(-1.20) * (0.9938 ** patient_age)
        return egfr.round(1)

    @staticmethod
    def compute_bun_creatinine_ratio(blood_urea: pd.Series, serum_creatinine: pd.Series) -> pd.Series:
        """Calculates BUN-to-Creatinine ratio (pre-renal vs intrinsic renal failure indicator)."""
        bu = pd.to_numeric(blood_urea, errors="coerce").clip(lower=1.0)
        scr = pd.to_numeric(serum_creatinine, errors="coerce").clip(lower=0.2)
        return (bu / scr).round(2)

    @staticmethod
    def compute_electrolyte_ratio(sodium: pd.Series, potassium: pd.Series) -> pd.Series:
        """Calculates Serum Sodium-to-Potassium ratio."""
        na = pd.to_numeric(sodium, errors="coerce").clip(lower=50.0)
        k = pd.to_numeric(potassium, errors="coerce").clip(lower=1.0)
        return (na / k).round(2)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all clinical transformations and returns augmented DataFrame."""
        out = df.copy()

        if "serum_creatinine" in out.columns and "age" in out.columns:
            out["egfr_ckd_epi"] = self.estimate_egfr_ckd_epi(out["serum_creatinine"], out["age"])

        if "blood_urea" in out.columns and "serum_creatinine" in out.columns:
            out["bun_creatinine_ratio"] = self.compute_bun_creatinine_ratio(out["blood_urea"], out["serum_creatinine"])

        if "sodium" in out.columns and "potassium" in out.columns:
            out["electrolyte_ratio"] = self.compute_electrolyte_ratio(out["sodium"], out["potassium"])

        return out
