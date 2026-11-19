import numpy as np
from kidney_ai.evaluation.metrics import compute_clinical_metrics


def test_metrics_calculation():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4]])

    metrics = compute_clinical_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 0.8
    assert metrics["specificity"] == 1.0
