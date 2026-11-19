# Kidney-AI Data Directory Hierarchy

This directory contains clinical datasets, raw laboratory records, de-identified biomarker registries, and processed train/val/test splits for the Kidney-AI research platform.

```
data/
├── README.md                 # Data hierarchy, HIPAA compliance, and schema documentation
├── external/                 # External clinical validation benchmarks and normative registries
├── interim/                  # Intermediate transformations and imputed matrices
├── manifests/                # Dataset integrity checksums (SHA-256) and record counts
├── processed/                # Normalized, scaled, and feature-engineered datasets
├── raw/                      # Raw clinical datasets (strictly de-identified, gitignored)
│   ├── clinical/             # Electronic Health Record tabular clinical sheets
│   ├── imaging/              # De-identified renal ultrasound / CT / MRI volumes
│   ├── lab_biomarkers/       # Serum creatinine, BUN, eGFR, urine albumin panels
│   └── temporal_ehr/         # ICU longitudinal streams for AKI staging (MIMIC/eICU)
└── splits/                   # Stratified splits for reproducible model benchmarking
    ├── test/                 # Holdout test partition (15%)
    ├── train/                # Training partition (70%)
    └── validation/           # Validation partition (15%)
```

## Clinical Data Compliance & De-identification
All datasets incorporated must be completely stripped of Protected Health Information (PHI) under HIPAA Safe Harbor rules and GDPR Article 9. No patient names, MRNs, dates of birth, or geographic identifiers are stored.
