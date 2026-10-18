"""Clinical data quality auditing and report generation."""

from pathlib import Path
from typing import Dict, Any, Optional
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from kidneycare.core.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
from kidneycare.utils.file_utils import ensure_dir, atomic_write_json, atomic_write_text


class DataQualityAuditor:
    """Generates comprehensive data quality reports for clinical tabular datasets."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def audit(self, df: Any, dataset_name: str = "UCI-CKD") -> Dict[str, Any]:
        """Compute full statistical quality report."""
        total_rows = len(df)
        total_cols = len(df.columns)

        missing_summary = {}
        for col in df.columns:
            cnt = int(df[col].isna().sum())
            if cnt > 0:
                missing_summary[col] = {
                    "count": cnt,
                    "percentage": round((cnt / total_rows) * 100, 2),
                }

        # Numerical statistics
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        num_stats = {}
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) > 0:
                num_stats[c] = {
                    "mean": round(float(s.mean()), 3),
                    "std": round(float(s.std()), 3),
                    "min": round(float(s.min()), 3),
                    "median": round(float(s.median()), 3),
                    "max": round(float(s.max()), 3),
                    "skew": round(float(s.skew()), 3) if len(s) > 2 else 0.0,
                }

        # Categorical statistics
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        cat_stats = {}
        for c in cat_cols:
            s = df[c].dropna().astype(str).str.strip().str.lower()
            cat_stats[c] = {
                "unique_values": int(s.nunique()),
                "top_value": s.mode().iloc[0] if len(s) > 0 else "N/A",
                "frequencies": s.value_counts().to_dict(),
            }

        report = {
            "dataset_name": dataset_name,
            "total_records": total_rows,
            "total_features": total_cols,
            "missing_values_by_feature": missing_summary,
            "total_missing_cells": int(df.isna().sum().sum()),
            "overall_missing_rate_pct": round((df.isna().sum().sum() / (total_rows * total_cols)) * 100, 2),
            "numerical_summary": num_stats,
            "categorical_summary": cat_stats,
        }

        # Save JSON and Markdown
        json_path = self.output_dir / f"data_quality_{dataset_name.lower().replace('-', '_')}.json"
        atomic_write_json(json_path, report)

        md_content = self._format_markdown_report(report)
        md_path = self.output_dir / f"data_quality_{dataset_name.lower().replace('-', '_')}.md"
        atomic_write_text(md_path, md_content)

        return report

    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        lines = [
            f"# Clinical Data Quality Audit: {report['dataset_name']}",
            "",
            f"- **Total Patient Records**: {report['total_records']}",
            f"- **Total Features**: {report['total_features']}",
            f"- **Overall Missing Rate**: {report['overall_missing_rate_pct']}%",
            "",
            "## Missingness by Feature",
            "| Feature | Missing Count | Missing % |",
            "| :--- | :--- | :--- |",
        ]
        for f, data in report["missing_values_by_feature"].items():
            lines.append(f"| `{f}` | {data['count']} | {data['percentage']}% |")

        lines.extend([
            "",
            "## Numerical Biomarker Distribution",
            "| Biomarker | Mean | Std | Min | Median | Max | Skew |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for b, data in report["numerical_summary"].items():
            lines.append(
                f"| `{b}` | {data['mean']} | {data['std']} | {data['min']} | "
                f"{data['median']} | {data['max']} | {data['skew']} |"
            )

        return "\n".join(lines)
