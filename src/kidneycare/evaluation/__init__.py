"""Comprehensive clinical evaluation metrics and model benchmarking."""

from kidneycare.evaluation.metrics import (
    calculate_clinical_metrics,
    calculate_confusion_matrix_elements,
    calculate_expected_calibration_error,
    bootstrap_confidence_interval,
)
from kidneycare.evaluation.evaluation import ModelEvaluator
from kidneycare.evaluation.model_comparison import ModelComparator
from kidneycare.evaluation.subgroup_analysis import SubgroupAuditor
from kidneycare.evaluation.statistical_analysis import compare_models_statistical

__all__ = [
    "calculate_clinical_metrics",
    "calculate_confusion_matrix_elements",
    "calculate_expected_calibration_error",
    "bootstrap_confidence_interval",
    "ModelEvaluator",
    "ModelComparator",
    "SubgroupAuditor",
    "compare_models_statistical",
]
