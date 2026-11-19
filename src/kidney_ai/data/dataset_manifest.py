"""Dataset manifest generation with SHA-256 cryptographic checksums."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any


class DatasetManifest:
    """Computes file hashes and registers dataset integrity manifests."""

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def generate_manifest(cls, filepath: Path, dataset_id: str) -> Dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "filename": filepath.name,
            "file_size_bytes": filepath.stat().st_size,
            "sha256": cls.compute_sha256(filepath),
        }
