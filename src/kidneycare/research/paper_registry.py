"""Catalog and loader for research papers."""

from pathlib import Path
from typing import Dict, List, Optional
from kidneycare.research.paper_metadata import PaperMetadata
from kidneycare.utils.file_utils import safe_load_json


class PaperRegistry:
    """Catalog managing bibliographic records of peer-reviewed kidney disease literature."""

    def __init__(self, metadata_path: str = "papers/metadata/papers_metadata.json"):
        self.metadata_path = Path(metadata_path)
        self.papers: Dict[str, PaperMetadata] = {}
        self._load()

    def _load(self) -> None:
        if self.metadata_path.exists():
            data = safe_load_json(self.metadata_path)
            for p_dict in data.get("papers", []):
                pm = PaperMetadata.from_dict(p_dict)
                self.papers[pm.id] = pm

    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        return self.papers.get(paper_id)

    def list_papers(self) -> List[PaperMetadata]:
        return list(self.papers.values())
