"""Comprehensive clinical classification and calibration metrics."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    confusion_matrix
)


def compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Computes sensitivity, specificity, balanced accuracy, F1, AUC, and Brier score."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0) # Clinical Sensitivity
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    metrics = {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "sensitivity": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(prec), 4),
        "macro_f1": round(float(f1), 4),
        "mcc": round(float(mcc), 4),
    }

    if y_prob is not None:
        if y_prob.ndim == 2:
            prob_pos = y_prob[:, 1]
        else:
            prob_pos = y_prob
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, prob_pos)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, prob_pos)), 4)
            metrics["brier_score"] = round(float(brier_score_loss(y_true, prob_pos)), 4)
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
            metrics["brier_score"] = 0.0

    return metrics
