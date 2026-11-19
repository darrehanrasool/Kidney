"""Stratified cross-validation and out-of-fold risk assessment."""

from typing import Dict, List, Any
import numpy as np
from sklearn.model_selection import StratifiedKFold
from kidney_ai.evaluation.metrics import compute_clinical_metrics


def run_stratified_kfold(model_builder, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = model_builder()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        probs = model.predict_proba(X_va) if hasattr(model, "predict_proba") else None
        fold_metrics.append(compute_clinical_metrics(y_va, preds, probs))

    # Mean and standard deviation across folds
    metric_keys = fold_metrics[0].keys()
    summary = {}
    for k in metric_keys:
        vals = [fm[k] for fm in fold_metrics]
        summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        summary[f"{k}_std"] = round(float(np.std(vals)), 4)

    return summary
