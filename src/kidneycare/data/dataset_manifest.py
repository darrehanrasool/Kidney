"""Dataset manifest management and checksum auditing."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from kidneycare.utils.file_utils import ensure_dir, safe_load_json, atomic_write_json
from kidneycare.utils.hashing import calculate_sha256


class DatasetManifest:
    """Manages recording and verification of downloaded datasets."""

    def __init__(self, manifest_path: str = "data/manifests/dataset_manifest.json"):
        self.manifest_path = Path(manifest_path)
        ensure_dir(self.manifest_path.parent)
        self.entries: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                data = safe_load_json(self.manifest_path)
                return data.get("datasets", {})
            except Exception:
                return {}
        return {}

    def save(self) -> None:
        """Persist manifest state atomically to disk."""
        payload = {
            "version": "1.0.0",
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "datasets": self.entries,
        }
        atomic_write_json(self.manifest_path, payload)

    def register_dataset(
        self,
        dataset_id: str,
        name: str,
        file_path: str,
        source_url: str,
        license_str: str,
        modality: str,
        target_definition: str,
        feature_count: int,
        record_count: int,
        missing_count: int,
        class_distribution: Dict[str, int],
        citation: str,
        limitations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register a newly downloaded or generated dataset into the manifest."""
        p = Path(file_path)
        checksum = calculate_sha256(p) if p.exists() else ""
        file_size = p.stat().st_size if p.exists() else 0

        entry = {
            "dataset_id": dataset_id,
            "name": name,
            "file_name": p.name,
            "file_path": str(p),
            "file_size_bytes": file_size,
            "sha256_checksum": checksum,
            "download_date": time.strftime("%Y-%m-%d"),
            "source_url": source_url,
            "license": license_str,
            "modality": modality,
            "target_definition": target_definition,
            "feature_count": feature_count,
            "record_count": record_count,
            "missing_value_count": missing_count,
            "class_distribution": class_distribution,
            "required_citation": citation,
            "known_limitations": limitations or ["Single-center cohort", "Moderate sample size (n=400)"],
            "processing_status": "verified_raw",
        }
        self.entries[dataset_id] = entry
        self.save()
        return entry

    def get_entry(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve dataset entry by ID."""
        return self.entries.get(dataset_id)
