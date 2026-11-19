"""Registry of supported clinical and benchmark kidney cohorts."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetMeta:
    dataset_id: str
    display_name: str
    category: str
    task_type: str
    target_column: str
    feature_count: int
    source_url: str


class DatasetRegistry:
    """Central catalog for nephrology benchmark datasets."""

    DATASETS: Dict[str, DatasetMeta] = {
        "uci_ckd": DatasetMeta(
            dataset_id="uci_ckd",
            display_name="UCI Chronic Kidney Disease Benchmark",
            category="tabular_clinical",
            task_type="binary_classification",
            target_column="classification",
            feature_count=24,
            source_url="https://archive.ics.uci.edu/dataset/336/chronic_kidney_disease"
        ),
        "renal_biomarkers": DatasetMeta(
            dataset_id="renal_biomarkers",
            display_name="Multicenter Longitudinal Renal Biomarker Cohort",
            category="longitudinal_biomarkers",
            task_type="multiclass_staging",
            target_column="ckd_stage",
            feature_count=28,
            source_url="local_registry"
        )
    }

    @classmethod
    def get(cls, dataset_id: str) -> Optional[DatasetMeta]:
        return cls.DATASETS.get(dataset_id)

    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls.DATASETS.keys())
