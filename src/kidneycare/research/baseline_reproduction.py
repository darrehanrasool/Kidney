"""Reproduction audit comparing published baselines with local models."""

from typing import Dict, Any, List
import pandas as pd
from kidneycare.research.paper_registry import PaperRegistry


class BaselineReproducer:
    """Manages comparison between literature findings and local experimental reproductions."""

    def __init__(self, registry: Optional[PaperRegistry] = None):
        self.registry = registry or PaperRegistry()

    def compare_with_local(self, local_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare local model evaluation with published paper metrics."""
        rows = []

        # Add literature baselines
        poonia = self.registry.get_paper("poonia_healthcare_2022")
        if poonia:
            for m_name, m_vals in poonia.reported_metrics.items():
                rows.append({
                    "Source": f"Poonia et al. (2022)",
                    "Model": m_name,
                    "Dataset": "UCI-CKD (80/20 split)",
                    "Accuracy": m_vals.get("accuracy", 0.0),
                    "F1-Score": m_vals.get("f1", 0.0),
                    "Status": "Reported in Literature",
                })

        # Add local models
        for m_name, m_vals in local_metrics.items():
            rows.append({
                "Source": "KidneyCare-AI (Local)",
                "Model": m_name,
                "Dataset": "UCI-CKD (70/15/15 Stratified)",
                "Accuracy": m_vals.get("accuracy", 0.0),
                "F1-Score": m_vals.get("f1_score", m_vals.get("f1", 0.0)),
                "Status": "Independently Reproduced",
            })

        return pd.DataFrame(rows)
