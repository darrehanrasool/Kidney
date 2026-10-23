"""Comprehensive clinical diagnostic evaluation metrics."""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)


def calculate_confusion_matrix_elements(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Calculate true negatives, false positives, false negatives, and true positives.

    Returns:
        Dict with keys 'tn', 'fp', 'fn', 'tp'.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def calculate_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / total_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return float(ece)


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate all standard clinical performance and calibration metrics.

    Args:
        y_true: True binary labels (0 = NotCKD, 1 = CKD).
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (CKD).
        threshold: Decision threshold applied.

    Returns:
        Dictionary containing all 12 requested metrics.
    """
    cm_counts = calculate_confusion_matrix_elements(y_true, y_pred)
    tn = cm_counts["tn"]
    fp = cm_counts["fp"]
    fn = cm_counts["fn"]
    tp = cm_counts["tp"]

    # Sensitivity / Recall
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Probability metrics
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(np.mean(y_true))
        brier = float(brier_score_loss(y_true, y_prob))
        ece = float(calculate_expected_calibration_error(y_true, y_prob))
    else:
        auc = 0.5
        pr_auc = float(np.mean(y_true))
        brier = 0.0
        ece = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "matthews_corrcoef": round(mcc, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "decision_threshold": round(threshold, 3),
        "confusion_matrix": cm_counts,
        "sample_size": len(y_true),
        "prevalence": round(float(np.mean(y_true)), 4),
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Any,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate empirical bootstrap confidence intervals for a metric.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_estimate = float(metric_func(y_true, y_pred))

    if n < 10:
        return point_estimate, point_estimate, point_estimate

    boot_scores = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        boot_scores.append(score)

    if not boot_scores:
        return point_estimate, point_estimate, point_estimate

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_scores, alpha * 100))
    upper = float(np.percentile(boot_scores, (1.0 - alpha) * 100))
    return point_estimate, lower, upper


class ClinicalMetricsResult:
    """Wrapper object with attribute and dict access for clinical metrics."""

    def __init__(self, metrics_dict: Dict[str, Any]):
        self._dict = metrics_dict
        self.accuracy = metrics_dict["accuracy"]
        self.precision = metrics_dict["precision"]
        self.recall = metrics_dict["recall_sensitivity"]
        self.sensitivity = metrics_dict["recall_sensitivity"]
        self.specificity = metrics_dict["specificity"]
        self.f1_macro = metrics_dict["f1_score"]
        self.roc_auc = metrics_dict["roc_auc"]
        self.pr_auc = metrics_dict["pr_auc"]
        self.balanced_accuracy = metrics_dict["balanced_accuracy"]
        self.brier_score = metrics_dict["brier_score"]
        self.expected_calibration_error = metrics_dict["expected_calibration_error"]
        self.confusion_matrix = metrics_dict["confusion_matrix"]

    def to_dict(self) -> Dict[str, Any]:
        return self._dict

    def __getitem__(self, item):
        return self._dict[item]


class ClinicalMetricsCalculator:
    """Unified calculator for clinical diagnostic metrics."""

    @staticmethod
    def compute_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> ClinicalMetricsResult:
        metrics_dict = calculate_clinical_metrics(y_true, y_pred, y_prob, threshold)
        return ClinicalMetricsResult(metrics_dict)
