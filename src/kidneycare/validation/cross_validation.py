"""Repeated Stratified Cross-Validation for robust statistical estimation."""

from typing import Dict, Any, List, Generator, Tuple
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

from kidneycare.evaluation.metrics import calculate_clinical_metrics


class RepeatedStratifiedCV:
    """Evaluates estimators across repeated stratified folds without data leakage."""

    def __init__(self, n_splits: int = 5, n_repeats: int = 2, random_state: int = 42):
        self.cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    def evaluate(self, estimator: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run repeated CV and aggregate metric distributions."""
        scores: Dict[str, List[float]] = {
            "accuracy": [],
            "f1_score": [],
            "roc_auc": [],
            "recall_sensitivity": [],
            "specificity": [],
        }

        for train_idx, val_idx in self.cv.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            clf = clone(estimator)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            probs = clf.predict_proba(X_va)[:, 1] if hasattr(clf, "predict_proba") else preds

            m = calculate_clinical_metrics(y_va, preds, probs)
            scores["accuracy"].append(m["accuracy"])
            scores["f1_score"].append(m["f1_score"])
            scores["roc_auc"].append(m["roc_auc"])
            scores["recall_sensitivity"].append(m["recall_sensitivity"])
            scores["specificity"].append(m["specificity"])

        summary = {}
        for k, v in scores.items():
            summary[k] = {
                "mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4),
                "min": round(float(np.min(v)), 4),
                "max": round(float(np.max(v)), 4),
            }

        return summary
