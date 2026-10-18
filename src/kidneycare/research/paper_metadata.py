"""Paper metadata schema and loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from kidneycare.utils.file_utils import safe_load_json


@dataclass
class PaperMetadata:
    """Bibliographic and methodological metadata for a published study."""

    id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    dataset: str
    architecture: str
    best_accuracy: float
    best_f1: float
    reproducibility_status: str
    reported_limitations: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    pdf_path: Optional[str] = None
    reported_metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperMetadata":
        return cls(
            id=data["id"],
            title=data["title"],
            authors=data.get("authors", []),
            year=data.get("year", 2020),
            venue=data.get("venue", ""),
            dataset=data.get("dataset", ""),
            architecture=data.get("architecture", ""),
            best_accuracy=data.get("best_accuracy", 0.0),
            best_f1=data.get("best_f1", 0.0),
            reproducibility_status=data.get("reproducibility_status", "Unverified"),
            reported_limitations=data.get("reported_limitations", []),
            doi=data.get("doi"),
            pdf_path=data.get("pdf_path"),
            reported_metrics=data.get("reported_metrics", {}),
        )
