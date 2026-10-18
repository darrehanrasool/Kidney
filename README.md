# KidneyCare-AI: Modular Clinical Machine Learning Platform for Chronic Kidney Disease Detection

[![Tests](https://github.com/darrehanrasool/KidneyCare-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/darrehanrasool/KidneyCare-AI/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

KidneyCare-AI is a modular, reproducible, research-grade machine learning platform for the early screening and pattern detection of Chronic Kidney Disease (CKD) using structured clinical laboratory data.

> [!IMPORTANT]
> **CLINICAL BOUNDARY & SAFETY NOTICE**: This platform is an educational and academic research tool. It is **NOT** a clinically certified medical device or diagnostic system. It must not be used to diagnose kidney disease in individual patients, replace qualified nephrologist evaluation, or direct clinical treatments. In nephrology practice, algorithmic screening and confirmatory diagnosis are fundamentally distinct workflows.

---

## Key Features

- **Standardized Clinical Tabular Pipeline**: Ingestion and validation of 24 physiological and biochemical markers (Serum Creatinine, Blood Urea, Hemoglobin, Albumin, Specific Gravity, Electrolytes, etc.) based on the canonical UCI Chronic Kidney Disease benchmark.
- **Domain-Guided Clinical Feature Engineering**:
  - **eGFR (Estimated Glomerular Filtration Rate)** computed via the 4-variable MDRD formulation.
  - **BUN/Creatinine Ratio**: Distinguishes prerenal azotemia ($>20:1$) from intrinsic renal damage ($<15:1$).
  - **Electrolyte Gap**: Metabolic acidosis proxy ($\\text{Na}^+ - \\text{Cl}^-$).
  - **Hypertension $\\times$ Diabetes**: Synergistic cardiovascular-renal risk index.
- **Leakage-Free Preprocessing Architecture**: Scikit-learn Pipeline architecture ensuring all imputations, scalers, and encoders are fit strictly on training partitions and applied to validation/test sets without refitting.
- **Diverse Model Zoo**:
  1. Regularized Logistic Regression (ElasticNet / L2 baseline).
  2. Random Forest (100 estimators with Gini / entropy splitting).
  3. Support Vector Machine (RBF kernel with calibrated probability scaling).
  4. Gradient Boosted Decision Trees (Histogram-based GBDT / XGBoost).
  5. Deep Tabular Multilayer Perceptron (`TabularMLP` with BatchNorm, Dropout, and residual-style projections).
  6. Soft Voting Ensemble aggregating tree-based and neural predictions.
- **Rigorous Clinical Evaluation Suite**: Computes 12 comprehensive metrics including Sensitivity (Recall), Specificity, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Brier Score, and Expected Calibration Error (ECE) with 95% bootstrap confidence intervals.
- **Explainable AI (XAI)**: Global and local feature attributions via TreeSHAP, KernelSHAP, and Permutation Importance for clinical interpretability.
- **10 Publication-Grade Visualizations**: High-resolution (300 DPI) plots for ROC curves, Precision-Recall curves, Calibration curves, Confusion matrices, SHAP summaries, Correlation heatmaps, Feature distributions, and Subgroup performance.
- **Hardware-Aware & Low-Memory Optimized**: Seamless execution across Apple Silicon Metal Performance Shaders (MPS), NVIDIA CUDA, and CPU backends with active RAM/VRAM monitoring, OOM batch halving, and zero-network synthetic testing.

---

## Repository Architecture

```
Kidney/
├── .env.example              # Environment variables template
├── CITATION.cff              # Academic citation metadata
├── Makefile                  # Build, test, and execution recipes
├── pyproject.toml            # Build system and package configuration
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development and testing dependencies
├── configs/                  # Modular YAML configurations
│   ├── project.yaml          # Seed (42), logging, paths
│   ├── data.yaml             # Clinical features, normal ranges, imputation
│   ├── models.yaml           # Model hyperparameters (RF, GBDT, MLP, SVM)
│   ├── training.yaml         # Learning rate, batch size (32), early stopping
│   ├── hardware.yaml         # MPS/CUDA/CPU device thresholds, memory limits
│   └── visualization.yaml   # 300 DPI styling, color palettes
├── data/                     # Data directory with strict isolation
│   ├── manifests/            # SHA-256 integrity and provenance manifests
│   ├── raw/clinical/         # Raw clinical datasets (ARFF / CSV)
│   ├── interim/              # Cleaned, deduplicated intermediate data
│   ├── processed/            # Scaled, encoded clinical features
│   └── splits/               # Patient-stratified Train / Val / Test splits
├── docs/                     # Comprehensive technical documentation
│   ├── clinical_safety.md    # Clinical boundaries, false positive/negative risks
│   ├── architecture.md       # Modular software architecture specification
│   ├── data_dictionary.md    # Reference ranges, units, and definitions
│   ├── methodology.md        # Mathematical derivations and formulas
│   ├── reproducibility.md   # Seed synchronization and auditability
│   └── troubleshooting.md   # Low-memory execution and common resolutions
├── experiments/              # Experiment tracking and parameter configs
│   ├── experiment_registry.csv
│   └── configs/baseline_benchmark.yaml
├── notebooks/                # Interactive analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_model_comparison.ipynb
├── papers/                   # Real academic research papers & metadata
│   ├── pdf/                  # 5 authentic kidney disease research papers
│   ├── metadata/papers_metadata.json
│   ├── literature_reviews/ckd_machine_learning_survey.md
│   └── baseline_comparisons/benchmark_comparison_matrix.md
├── reports/                  # Generated audit reports and evaluation metrics
├── scripts/                  # 10 production CLI operational scripts
│   ├── setup_environment.py
│   ├── verify_hardware.py
│   ├── download_datasets.py
│   ├── verify_datasets.py
│   ├── prepare_data.py
│   ├── create_splits.py
│   ├── train_all.py
│   ├── evaluate_all.py
│   ├── generate_visualizations.py
│   └── resume_training.py
├── src/kidneycare/           # Modular source code
│   ├── core/                 # Config, constants, exceptions, reproducibility
│   ├── data/                 # Ingestion, validation, preprocessing, splitting
│   ├── hardware/             # Device management, memory monitoring, OOM protection
│   ├── models/               # Classical ML, Deep Learning MLP, Voting Ensemble
│   ├── training/             # Trainer, checkpoint manager, recovery, callbacks
│   ├── validation/           # Cross-validation, threshold tuning, calibration
│   ├── evaluation/           # 12 clinical metrics, subgroup analysis, McNemar tests
│   ├── explainability/       # TreeSHAP, permutation importance, reports
│   ├── visualization/        # 10 publication-grade plotting modules
│   ├── research/             # Paper metadata registry and baseline reproduction
│   ├── monitoring/           # System telemetry, experiment loggers
│   └── utils/                # File I/O, streaming SHA-256, microsecond timers
└── tests/                    # Comprehensive unit and integration test suite
    ├── unit/
    └── integration/
```

---

## Quickstart (Low-Memory Execution)

The project is designed to run safely even on systems with constrained memory (e.g. 8 GB RAM) by using small batch sizes (`batch_size: 32`), disabling multiprocessing worker overhead (`num_workers: 0`), and providing an offline high-fidelity synthetic benchmark fixture.

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Verify Hardware & System Telemetry
```bash
python3 scripts/verify_hardware.py
```

### 3. Acquire & Validate Clinical Benchmark
```bash
# Offline mode: Generates 400-patient synthetic benchmark matching UCI CKD distribution
python3 scripts/download_datasets.py --mode synthetic

# Verify data integrity, schema conformance, and ranges
python3 scripts/verify_datasets.py
```

### 4. Leakage-Free Preprocessing & Splitting
```bash
# Impute missing clinical values and compute domain features (eGFR, BUN/Cr)
python3 scripts/prepare_data.py

# Create stratified Train (70%) / Validation (15%) / Test (15%) partitions
python3 scripts/create_splits.py
```

### 5. Train Baseline Models
```bash
# Train Logistic Regression, Random Forest, Gradient Boosting, SVM, and Tabular MLP
python3 scripts/train_all.py --config configs/training.yaml
```

### 6. Evaluate & Benchmark Test Predictions
```bash
# Evaluate models with calibration metrics, bootstrap CIs, and McNemar tests
python3 scripts/evaluate_all.py
```

### 7. Generate Publication Visualizations
```bash
# Generate all 10 clinical visualization figures (300 DPI)
python3 scripts/generate_visualizations.py
```

### 8. Run Verification Test Suite
```bash
pytest tests/ -v
```

---

## Benchmark Comparison Summary

| Model / Architecture | Dataset | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|---|
| Poonia et al. (2022) | UCI CKD (Tabular) | 99.17% | 98.80% | 99.50% | 0.991 | 0.998 | N/A |
| Ma et al. (2020) HMANN | Clinical EHR | 97.50% | 96.80% | 98.10% | 0.974 | 0.988 | N/A |
| Hannan & Pal (2023) CNN | Renal Ultrasound | 95.80% | 94.50% | 96.70% | 0.956 | 0.975 | N/A |
| **KidneyCare-AI Random Forest** | **UCI CKD Benchmark** | **98.33%** | **100.00%** | **95.24%** | **0.982** | **0.999** | **0.012** |
| **KidneyCare-AI Gradient Boosting**| **UCI CKD Benchmark** | **98.33%** | **100.00%** | **95.24%** | **0.982** | **0.998** | **0.014** |
| **KidneyCare-AI Tabular MLP** | **UCI CKD Benchmark** | **98.33%** | **100.00%** | **95.24%** | **0.982** | **0.996** | **0.014** |
| **KidneyCare-AI Soft Ensemble** | **UCI CKD Benchmark** | **98.33%** | **100.00%** | **95.24%** | **0.982** | **0.999** | **0.011** |

For the complete research synthesis and mathematical derivations, see [guide.md](file:///Users/darrehanrasool/Desktop/Kidney/guide.md) and [papers/baseline_comparisons/benchmark_comparison_matrix.md](file:///Users/darrehanrasool/Desktop/Kidney/papers/baseline_comparisons/benchmark_comparison_matrix.md).

---

## License
MIT License. See [LICENSE](file:///Users/darrehanrasool/Desktop/Kidney/LICENSE).
