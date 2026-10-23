# Model Card: Random Forest Clinical Screener (KidneyCare-AI)

## Model Details
- **Architecture**: Ensembled Breiman Random Forest Classifier (100 estimators, Gini criterion).
- **Modality**: Structured Tabular Electronic Health Records (24 clinical features + 2021 CKD-EPI derived features).
- **Target**: Chronic Kidney Disease (CKD) Binary Classification.
- **License**: Research & Educational Use Only.

## Clinical Intended Use
- **Primary Use**: Non-invasive algorithmic risk screening of ambulatory and inpatient lab panels.
- **Out-of-Scope**: Definitive clinical diagnosis, biopsy replacement, or standalone treatment decisions.

## Benchmark Performance (Holdout Test Set)
- **Accuracy**: 100.0%
- **Sensitivity / Recall**: 100.0%
- **Specificity**: 100.0%
- **F1-Score**: 1.0000
- **Brier Calibration Score**: 0.0022
- **Expected Calibration Error (ECE)**: 0.0257
