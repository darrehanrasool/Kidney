"""Comparative research synthesis and gap analysis exporter."""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from kidneycare.research.paper_registry import PaperRegistry
from kidneycare.utils.file_utils import ensure_dir, atomic_write_text


class ResearchComparator:
    """Exports structured literature comparison tables and research gap summaries."""

    def __init__(self, registry: Optional[PaperRegistry] = None, output_dir: str = "reports"):
        self.registry = registry or PaperRegistry()
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def generate_literature_summary(self) -> Path:
        """Export comprehensive literature summary table."""
        papers = self.registry.list_papers()
        rows = []
        for p in papers:
            rows.append({
                "Study": f"{p.title[:35]}...",
                "Venue": p.venue,
                "Year": p.year,
                "Dataset": p.dataset,
                "Architecture": p.architecture,
                "Reported Accuracy": f"{p.best_accuracy:.2%}",
                "Reported F1": f"{p.best_f1:.2%}",
                "Reproducibility": p.reproducibility_status,
            })
        df = pd.DataFrame(rows)
        out_path = self.output_dir / "literature_benchmark_summary.md"
        content = f"# Kidney Disease Machine Learning Literature Matrix\n\n{df.to_markdown(index=False)}\n"
        atomic_write_text(out_path, content)
        return out_path
