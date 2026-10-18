"""Unit tests for clinical metrics, calibration, and statistical analysis."""

import numpy as np
import pytest

from kidneycare.evaluation.metrics import (
    calculate_clinical_metrics,
    calculate_confusion_matrix_elements,
    calculate_expected_calibration_error,
)
from kidneycare.evaluation.statistical_analysis import compare_models_statistical


def test_clinical_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 0]) # 1 false negative
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.4, 0.3])

    metrics = calculate_clinical_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall_sensitivity" in metrics
    assert "specificity" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "expected_calibration_error" in metrics
    assert metrics["accuracy"] == 0.875
    assert metrics["specificity"] == 1.0  # 4/4 true negatives
    assert metrics["confusion_matrix"]["fn"] == 1


def test_expected_calibration_error():
    y_true = np.array([1, 1, 0, 0])
    # Perfect probabilities
    y_prob_perf = np.array([1.0, 1.0, 0.0, 0.0])
    ece_perf = calculate_expected_calibration_error(y_true, y_prob_perf, n_bins=2)
    assert ece_perf == 0.0

    # Miscalibrated probabilities
    y_prob_bad = np.array([0.0, 0.0, 1.0, 1.0])
    ece_bad = calculate_expected_calibration_error(y_true, y_prob_bad, n_bins=2)
    assert ece_bad > 0.5


def test_mcnemar_statistical_comparison():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred_a = np.array([1, 1, 0, 0, 1, 0])
    y_pred_b = np.array([1, 0, 0, 1, 1, 0])

    res = compare_models_statistical(y_true, y_pred_a, y_pred_b)
    assert "p_value" in res
    assert "mcnemar_statistic" in res
    assert "significant_p_05" in res
