"""Registry of supported kidney disease datasets."""

from typing import Dict, Any, List, Optional
from kidneycare.core.constants import CLINICAL_BOUNDARY_DISCLAIMER


class DatasetRegistry:
    """Central catalog and metadata repository for kidney disease datasets."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {
            "UCI-CKD": {
                "id": "UCI-CKD",
                "name": "UCI Chronic Kidney Disease Dataset",
                "source": "UCI Machine Learning Repository",
                "uci_id": 330,
                "modality": "tabular_clinical",
                "records": 400,
                "features": 24,
                "target": "classification",
                "classes": ["ckd", "notckd"],
                "class_counts": {"ckd": 250, "notckd": 150},
                "source_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00330/chronic_kidney_disease.arff",
                "backup_url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/chronic-kidney-disease.csv",
                "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "citation": "Rubini, L., Eswaran, P. (2015). Chronic Kidney Disease. UCI Machine Learning Repository.",
                "status": "supported_primary",
            },
            "Synthetic-CKD-Benchmark": {
                "id": "Synthetic-CKD-Benchmark",
                "name": "Synthetic Clinical CKD Verification Cohort",
                "source": "KidneyCare-AI Internal Generator",
                "modality": "tabular_clinical",
                "records": 400,
                "features": 24,
                "target": "classification",
                "classes": ["ckd", "notckd"],
                "class_counts": {"ckd": 250, "notckd": 150},
                "license": "MIT",
                "citation": "KidneyCare-AI Benchmark Generator (2026)",
                "status": "supported_synthetic",
            },
            "Kaggle-CKD-Prediction": {
                "id": "Kaggle-CKD-Prediction",
                "name": "Kaggle Chronic Kidney Disease Prediction",
                "source": "Kaggle",
                "modality": "tabular_clinical",
                "target": "classification",
                "requires_auth": True,
                "license": "Open Data",
                "citation": "Kaggle CKD Community Dataset",
                "status": "optional_secondary",
            },
            "Kidney-Ultrasound-Benchmark": {
                "id": "Kidney-Ultrasound-Benchmark",
                "name": "Ultrasound Kidney Imaging Dataset (Optional)",
                "source": "Academic / Hannan & Pal (2023)",
                "modality": "imaging_ultrasound",
                "status": "optional_future_extension",
            },
        }

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve dataset metadata by identifier."""
        return self._registry.get(dataset_id)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all registered dataset configurations."""
        return list(self._registry.values())

    def is_supported(self, dataset_id: str) -> bool:
        """Check if dataset identifier is recognized."""
        return dataset_id in self._registry
