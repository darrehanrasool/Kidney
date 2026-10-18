"""Comparative analysis and benchmarking of multiple clinical models."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from kidneycare.utils.file_utils import ensure_dir, atomic_write_text


class ModelComparator:
    """Aggregates and formats benchmark comparisons across all evaluated models."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        self.records: List[Dict[str, Any]] = []

    def add_model_result(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        runtime_seconds: float = 0.0,
        model_type: str = "classical",
    ) -> None:
        """Add evaluation output for a specific model."""
        record = {
            "Model": model_name,
            "Type": model_type,
            "Accuracy": metrics.get("accuracy", 0.0),
            "F1-Score": metrics.get("f1_score", 0.0),
            "Sensitivity (Recall)": metrics.get("recall_sensitivity", 0.0),
            "Specificity": metrics.get("specificity", 0.0),
            "ROC-AUC": metrics.get("roc_auc", 0.0),
            "PR-AUC": metrics.get("pr_auc", 0.0),
            "Balanced Acc": metrics.get("balanced_accuracy", 0.0),
            "MCC": metrics.get("matthews_corrcoef", 0.0),
            "Brier Score": metrics.get("brier_score", 0.0),
            "ECE": metrics.get("expected_calibration_error", 0.0),
            "Runtime (s)": round(runtime_seconds, 3),
        }
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert comparative records to sorted DataFrame."""
        df = pd.DataFrame(self.records)
        if not df.empty and "F1-Score" in df.columns:
            df = df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
        return df

    def save_reports(self, filename_prefix: str = "model_comparison") -> Tuple[Path, Path]:
        """Save benchmark comparison table as CSV and Markdown."""
        df = self.to_dataframe()

        csv_path = self.output_dir / f"{filename_prefix}.csv"
        df.to_csv(csv_path, index=False)

        md_path = self.output_dir / f"{filename_prefix}.md"
        md_text = f"# KidneyCare-AI Clinical Model Benchmark Comparison\n\n{df.to_markdown(index=False)}\n"
        atomic_write_text(md_path, md_text)

        return csv_path, md_path
