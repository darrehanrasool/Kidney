"""Registry of supported kidney disease datasets across clinical tabular and imaging modalities."""

from typing import Dict, Any, List, Optional
from kidneycare.core.constants import CLINICAL_BOUNDARY_DISCLAIMER


class DatasetRegistry:
    """Central catalog and metadata repository for kidney disease datasets."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {
            "UCI-CKD": {
                "id": "UCI-CKD",
                "name": "UCI Chronic Kidney Disease Dataset",
                "source": "UCI Machine Learning Repository / Apollo Hospitals",
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
            "NHANES-CKD-Large": {
                "id": "NHANES-CKD-Large",
                "name": "NHANES Large-Scale Epidemiological Kidney Biomarker Cohort",
                "source": "CDC National Health and Nutrition Examination Survey",
                "modality": "tabular_clinical",
                "records": 8250,
                "features": 28,
                "target": "ckd_stage_kdigo",
                "classes": ["G1", "G2", "G3a", "G3b", "G4", "G5"],
                "license": "Public Domain (CDC / US Gov)",
                "citation": "CDC/NCHS National Health and Nutrition Examination Survey (NHANES).",
                "status": "supported_large_scale",
            },
            "Kidney-Ultrasound-Benchmark": {
                "id": "Kidney-Ultrasound-Benchmark",
                "name": "Ultrasound B-Mode Renal Pathology Dataset",
                "source": "Academic / Hannan & Pal (2023) / Open Kidney US (Singla 2022)",
                "modality": "imaging_ultrasound",
                "records": 600,
                "classes": ["normal", "cyst", "stone", "tumour"],
                "format": "PNG / JPEG (150x150, 224x224)",
                "license": "Creative Commons Attribution",
                "citation": "Hannan & Pal (2023); Singla et al. (2022).",
                "status": "supported_imaging",
            },
            "Kidney-CT-Benchmark": {
                "id": "Kidney-CT-Benchmark",
                "name": "Computed Tomography (CT) Renal Pathology Slices",
                "source": "KiTS19 / DeepLesion NIH Subset",
                "modality": "imaging_ct",
                "records": 600,
                "classes": ["normal", "cyst", "stone", "tumour"],
                "format": "PNG / NIfTI (224x224, HU windowed)",
                "license": "Open Academic Challenge / NIH Box",
                "citation": "Heller et al. (2019) KiTS19 Challenge; Yan et al. (2018) DeepLesion.",
                "status": "supported_imaging",
            },
            "Multimodal-Kidney-Cohort": {
                "id": "Multimodal-Kidney-Cohort",
                "name": "NephroFusion Paired Multimodal Clinical Cohort",
                "source": "KidneyCare-AI Multi-Task Fusion Engine",
                "modality": "multimodal_paired",
                "records": 1000,
                "tasks": ["ckd_binary", "kdigo_stage", "albuminuria_stage", "lesion_type"],
                "license": "MIT Research",
                "citation": "KidneyCare-AI NephroFusionNet Protocol (2026).",
                "status": "supported_multimodal",
            },
            "KiTS19-Challenge": {
                "id": "KiTS19-Challenge",
                "name": "Kidney Tumor Segmentation Challenge (CT)",
                "source": "University of Minnesota / Grand Challenge",
                "modality": "imaging_ct",
                "records": 300,
                "license": "CC BY-NC-SA 4.0",
                "citation": "Heller, N. et al. (2019). The KiTS19 Challenge Data.",
                "status": "optional_external_download",
            },
            "DeepLesion-Kidney": {
                "id": "DeepLesion-Kidney",
                "name": "NIH DeepLesion CT Kidney Lesion Subset",
                "source": "NIH Clinical Center Box",
                "modality": "imaging_ct",
                "records": 1200,
                "license": "Open Government / CC0",
                "citation": "Yan, K. et al. (2018). DeepLesion. CVPR.",
                "status": "optional_external_download",
            },
            "MIMIC-IV-AKI": {
                "id": "MIMIC-IV-AKI",
                "name": "PhysioNet MIMIC-IV Acute Kidney Injury Cohort",
                "source": "MIT Laboratory for Computational Physiology",
                "modality": "ehr_time_series",
                "license": "PhysioNet Credentialed Health Data Use Agreement",
                "citation": "Johnson, A. et al. (2023). MIMIC-IV.",
                "status": "documented_credentialed_workflow",
            },
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        return list(self._registry.values())

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self._registry.get(dataset_id)

    def get_supported_ids(self) -> List[str]:
        return list(self._registry.keys())
