# KidneyCare-AI Data Directory Structure

This directory manages the end-to-end data lifecycle for the KidneyCare-AI platform, adhering to strict clinical data hygiene, integrity verification, privacy protection, and split isolation standards.

## Directory Structure

```
data/
├── README.md               # Data directory documentation
├── manifests/              # JSON checksum, provenance, and dictionary manifests
│   └── dataset_manifest.json
├── raw/                    # Untouched original raw clinical data downloads
│   ├── clinical/           # Tabular clinical records (UCI CKD ARFF/CSV)
│   └── imaging/            # Optional kidney ultrasound/CT DICOM/PNG files
├── interim/                # Imputed, cleaned, deduplicated intermediate clinical datasets
├── processed/              # Standardized, encoded, and feature-engineered datasets
├── splits/                 # Leakage-free partitioned subsets
│   ├── train/              # Training partition (70%)
│   ├── validation/         # Validation partition for hyperparameter tuning & threshold selection (15%)
│   └── test/               # Hold-out test partition for unbiased final reporting (15%)
└── external/               # KDIGO clinical guidelines, normal reference ranges, LOINC mappings
```

## Dataset Acquisition

To download the primary clinical benchmark (UCI Chronic Kidney Disease dataset) or generate the reproducible offline benchmark fixture:

```bash
# Offline high-fidelity synthetic benchmark (400 records, 24 clinical biomarkers)
python3 scripts/download_datasets.py --mode synthetic

# Live download from UCI ML Repository
python3 scripts/download_datasets.py --mode live
```

Verify integrity and schema conformance with:
```bash
python3 scripts/verify_datasets.py
```

Prepare and partition with strict leakage prevention:
```bash
python3 scripts/prepare_data.py
python3 scripts/create_splits.py
```
