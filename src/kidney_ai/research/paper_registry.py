"""Paper registry and literature baseline metadata."""

from typing import List, Dict


class PaperRegistry:
    """Registry of published state-of-the-art benchmarks in renal AI."""

    PAPERS: List[Dict[str, str]] = [
        {
            "id": "ckd_ml_survey_2024",
            "title": "Machine Learning and Deep Learning in Chronic Kidney Disease Diagnosis: A Systematic Review",
            "venue": "Nature Digital Medicine",
            "reported_auc": "0.985"
        },
        {
            "id": "aki_deep_recurrent_2023",
            "title": "Deep Recurrent Neural Networks for Real-Time Acute Kidney Injury Prediction in Intensive Care",
            "venue": "Lancet Digital Health",
            "reported_auc": "0.942"
        }
    ]
