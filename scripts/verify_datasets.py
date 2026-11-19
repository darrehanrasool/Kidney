#!/usr/bin/env python3
"""Dataset integrity, schema, and biological bounds auditor."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kidney_ai.data.data_validator import DataValidator
from kidney_ai.data.dataset_manifest import DatasetManifest


def main():
    raw_file = PROJECT_ROOT / "data" / "raw" / "clinical" / "chronic_kidney_disease.csv"
    if not raw_file.exists():
        print(f"File {raw_file} does not exist. Run 'make download-datasets' first.")
        return

    df = pd.read_csv(raw_file)
    validator = DataValidator()
    anomalies = validator.audit_bounds(df)

    print("=" * 70)
    print("             KIDNEY-AI CLINICAL DATASET AUDIT                   ")
    print("=" * 70)
    print(f"Audited File: {raw_file.name}")
    print(f"Total Cohort Rows: {len(df)}")
    print(f"Feature Columns: {len(df.columns)}")
    print(f"Physiological Anomalies Flagged: {len(anomalies)}")
    for feat, count in anomalies.items():
        print(f"  • {feat}: {count} records out of physiological range")

    # Generate Manifest
    manifest = DatasetManifest.generate_manifest(raw_file, "uci_ckd")
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(manifest_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"SHA-256 Checksum: {manifest['sha256']}")
    print("Manifest saved to data/manifests/dataset_manifest.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
