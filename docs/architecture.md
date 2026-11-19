# Kidney-AI System Architecture

## Overview

Kidney-AI is engineered as an end-to-end, multi-paradigm medical artificial intelligence framework designed for nephrology decision support, chronic kidney disease (CKD) staging, and acute kidney injury (AKI) risk prognosis.

```
                    ┌──────────────────────────────────────────────┐
                    │          Heterogeneous Clinical Data         │
                    │   (Tabular EHR, Lab Biomarkers, Imaging)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      Clinical Data Validation & Ingestion    │
                    │  - Strict PHI De-identification              │
                    │  - Biological Range & Missingness Auditing   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │        Medical Feature Engineering Engine    │
                    │  - eGFR (CKD-EPI 2021 formulation)           │
                    │  - BUN/Creatinine Ratio & Electrolyte Balance │
                    │  - Robust Scaler & Categorical Imputation    │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │         Multi-Paradigm Model Zoo             │
                    │  - Classical ML: RF, LightGBM, XGBoost, SVM  │
                    │  - Deep Learning: Tabular ResNet, MLP, LSTM  │
                    │  - Calibrated Stacking & Soft Voting Ensembles│
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      Clinical Explainability & Safety        │
                    │  - Global & Patient-Specific SHAP Analysis   │
                    │  - Risk Probability Calibration (Platt)      │
                    │  - High-Sensitivity Safety Decision Guardrails│
                    └──────────────────────────────────────────────┘
```

## Architectural Modules

1. **`kidney_ai.core`**: Centralized configuration management, clinical constants, domain exceptions, and seed reproducibility controllers.
2. **`kidney_ai.hardware`**: Compute acceleration detection (Apple Silicon MPS, NVIDIA CUDA, CPU) and RAM/VRAM resource budgeting.
3. **`kidney_ai.data`**: Clinical data validators, MICE/KNN imputation, automated stratified train/val/test splitting, and feature derivation.
4. **`kidney_ai.models`**: Modular model factory supporting classical statistical learning, deep tabular neural architectures, and ensembles.
5. **`kidney_ai.training`**: FocalLoss-regularized training loops, early stopping, and automatic checkpoint persistence.
6. **`kidney_ai.evaluation`**: Balanced accuracy, sensitivity/specificity, ROC-AUC, PR-AUC, and DeLong statistical tests.
7. **`kidney_ai.explainability`**: Patient-level SHAP explanations and biomarker attribution maps for transparent clinician review.
8. **`kidney_ai.pipeline`**: Unified CLI pipeline connecting raw clinical input to validated risk prediction.
