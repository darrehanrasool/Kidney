"""Statistical exploration, distribution audits, and entropy analysis."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from kidney_ai.data.data_quality import DataQualityAuditor


class StatisticalAnalyzer:
    """Executes full diagnostic quality audit on raw clinical tables."""

    def __init__(self, data_path: Path = None):
        self.data_path = Path(data_path or "data/raw/clinical/chronic_kidney_disease.csv")

    def run_full_audit(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            return {"status": "NO_DATA"}

        df = pd.read_csv(self.data_path)
        summary = DataQualityAuditor.audit(df)

        report_dir = Path("reports/data_quality")
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "data_quality_report.md", "w") as f:
            f.write("# Clinical Data Quality Audit\n\n")
            f.write(f"- Total Patient Records: {summary['total_records']}\n")
            f.write(f"- Overall Completeness: {summary['overall_completeness_pct']}%\n")
            f.write(f"- Missing Cell Count: {summary['missing_cells']}\n")

        return summary
