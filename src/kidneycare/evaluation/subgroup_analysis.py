"""Clinical demographic and comorbidity subgroup performance auditing."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from kidneycare.evaluation.metrics import calculate_clinical_metrics


class SubgroupAuditor:
    """Evaluates whether model discrimination degrades across clinical subgroups."""

    def __init__(self):
        pass

    def audit_subgroups(
        self,
        df_raw_test: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """Audit performance stratified by age groups, hypertension, and diabetes.

        Args:
            df_raw_test: Raw unnormalized test dataframe containing clinical metadata.
            y_true: True test labels.
            y_pred: Predicted test labels.
            y_prob: Predicted test probabilities.

        Returns:
            Dictionary of stratified subgroup performance metrics.
        """
        results: Dict[str, Any] = {}

        # 1. Stratify by Age (<50 vs >=50)
        if "age" in df_raw_test.columns:
            age_s = pd.to_numeric(df_raw_test["age"], errors="coerce")
            mask_young = (age_s < 50.0).values
            mask_senior = (age_s >= 50.0).values

            if np.sum(mask_young) > 5 and len(np.unique(y_true[mask_young])) > 1:
                results["Age < 50"] = calculate_clinical_metrics(
                    y_true[mask_young], y_pred[mask_young], y_prob[mask_young]
                )
            if np.sum(mask_senior) > 5 and len(np.unique(y_true[mask_senior])) > 1:
                results["Age >= 50"] = calculate_clinical_metrics(
                    y_true[mask_senior], y_pred[mask_senior], y_prob[mask_senior]
                )

        # 2. Stratify by Hypertension
        if "htn" in df_raw_test.columns:
            htn_s = df_raw_test["htn"].astype(str).str.strip().str.lower()
            mask_htn_yes = htn_s.isin(["yes", "1", "true"]).values
            mask_htn_no = htn_s.isin(["no", "0", "false"]).values

            if np.sum(mask_htn_yes) > 5 and len(np.unique(y_true[mask_htn_yes])) > 1:
                results["Hypertension: Yes"] = calculate_clinical_metrics(
                    y_true[mask_htn_yes], y_pred[mask_htn_yes], y_prob[mask_htn_yes]
                )
            if np.sum(mask_htn_no) > 5 and len(np.unique(y_true[mask_htn_no])) > 1:
                results["Hypertension: No"] = calculate_clinical_metrics(
                    y_true[mask_htn_no], y_pred[mask_htn_no], y_prob[mask_htn_no]
                )

        # 3. Stratify by Diabetes Mellitus
        if "dm" in df_raw_test.columns:
            dm_s = df_raw_test["dm"].astype(str).str.strip().str.lower()
            mask_dm_yes = dm_s.isin(["yes", "1", "true"]).values
            mask_dm_no = dm_s.isin(["no", "0", "false"]).values

            if np.sum(mask_dm_yes) > 5 and len(np.unique(y_true[mask_dm_yes])) > 1:
                results["Diabetes: Yes"] = calculate_clinical_metrics(
                    y_true[mask_dm_yes], y_pred[mask_dm_yes], y_prob[mask_dm_yes]
                )
            if np.sum(mask_dm_no) > 5 and len(np.unique(y_true[mask_dm_no])) > 1:
                results["Diabetes: No"] = calculate_clinical_metrics(
                    y_true[mask_dm_no], y_pred[mask_dm_no], y_prob[mask_dm_no]
                )

        return results
