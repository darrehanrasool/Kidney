# Kidney-AI: Multi-Modal Deep Learning & Machine Learning Framework for Chronic Kidney Disease Staging, Acute Kidney Injury Prediction, and Renal Prognosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Tests: 10 Passed](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg?logo=pytest)](tests/)
[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-passing-brightgreen.svg?logo=github-actions)](https://github.com/darrehanrasool/Kidney/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI: Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.kidney--ai-blue.svg)](https://zenodo.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Project Lead](https://img.shields.io/badge/Project%20Lead-Dar%20Rehan%20Rasool-0284c7.svg)](https://github.com/darrehanrasool)
[![Copyright](https://img.shields.io/badge/Copyright-%40darrehanrasool-10b981.svg)](https://github.com/darrehanrasool)

**Kidney-AI** is an enterprise-grade clinical research platform and publication-grade benchmarking framework engineered for early screening, disease staging, and risk stratification of **Chronic Kidney Disease (CKD)** and **Acute Kidney Injury (AKI)**. Integrating clinical Electronic Health Records (EHR), serial laboratory biomarker panels (serum creatinine, blood urea nitrogen, urine albumin, electrolytes), and patient symptomatology, the platform introduces a **Multi-Paradigm Predictive Architecture**: combining fast, interpretable tree-based ensembles (Random Forest, Gradient Boosting, LightGBM) with deep residual tabular neural networks featuring Layer Normalization, skip connections, and Focal Loss regularization.

Calibrated via Platt scaling and paired with game-theoretic SHAP explainability, Kidney-AI delivers high diagnostic sensitivity ($\ge 99.1\%$) and calibrated decision support for clinical researchers, nephrologists, and healthcare systems.

---

## 📑 Table of Contents
1. [Research Team & Authorship](#-research-team--authorship)
2. [Complete System Architecture & Workflow](#-complete-system-architecture--workflow)
3. [Clinical Biomarker Engineering & Formulations](#-clinical-biomarker-engineering--formulations)
4. [Multi-Paradigm Model Zoo](#-multi-paradigm-model-zoo)
5. [Clinical Safety, Calibration & Decision Guardrails](#-clinical-safety-calibration--decision-guardrails)
6. [Supported Datasets & Benchmark Evaluation](#-supported-datasets--benchmark-evaluation)
7. [Hardware Acceleration (Apple Silicon MPS & NVIDIA CUDA)](#-hardware-acceleration-apple-silicon-mps--nvidia-cuda)
8. [Directory Structure](#-directory-structure)
9. [Quick Start & Command Reference](#-quick-start--command-reference)
10. [Testing & Quality Assurance](#-testing--quality-assurance)
11. [Citation & Intellectual Property](#-citation--intellectual-property)

---

## 👥 Research Team & Authorship

This research project and benchmark platform is collaboratively developed under faculty guidance and supervision:

| Role | Name | Designation & Affiliation | Research & Project Contribution |
| :--- | :--- | :--- | :--- |
| **Faculty Research Supervisor** | **Dr. Safia Gul\*** | Faculty Supervisor, Centre for Artificial Intelligence (CAI) | Project Conceptualization, Clinical Research Supervision & Strategic Architecture Direction |
| **Project Lead & Contributing Student** | [**Dar Rehan Rasool**](https://github.com/darrehanrasool) | M.Tech Student, Centre for Artificial Intelligence, IUST | Architecture Design, Tabular ResNet, Clinical Feature Engineering, Benchmarks & XAI |
| **Contributing Student** | [**Muneeb Syed Shah**](https://github.com/muneebsyed-oss) | M.Tech Student, Centre for Artificial Intelligence, IUST | Machine Learning Staging, Feature Harmonization & Cross-Cohort Evaluation |
| **Contributing Student** | [**Shafiya Bashir Pampori**](https://github.com/shafiyabashirpampori) | M.Tech Student, Centre for Artificial Intelligence, IUST | Clinical Missingness Auditing, Biomarker Quality Analysis & Data Integrity |

- **Academic Unit**: Centre for Artificial Intelligence (CAI)
- **Institution**: Islamic University of Science and Technology (IUST), Awantipora, Jammu & Kashmir, India
- **Repository**: [https://github.com/darrehanrasool/Kidney](https://github.com/darrehanrasool/Kidney)
- **Copyright & Handle**: `@darrehanrasool` • [GitHub: darrehanrasool](https://github.com/darrehanrasool)
- *\* Denotes corresponding/supervising faculty author.*

---

## 👥 Contributors

- [**Shafiya Bashir**](https://github.com/shafiyabashirpampori) (`@shafiyabashirpampori`) — Contributor
- [**Dar Rehan Rasool**](https://github.com/darrehanrasool) (`@darrehanrasool`) — Project Lead & Maintainer
- [**Muneeb Syed Shah**](https://github.com/muneebsyed-oss) (`@muneebsyed-oss`) — Contributor

---

## 🔬 Complete System Architecture & Workflow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              1. CLINICAL INGESTION & AUDIT                             │
│  - Patient Demographics: Age, Blood Pressure, Comorbidities (HTN, Diabetes, CAD)       │
│  - Laboratory Panels: Serum Creatinine, BUN, Electrolytes (Na, K), Hemoglobin, PCV     │
│  - Urinalysis: Specific Gravity, Urine Albumin, Urine Glucose, Pus Cells, Bacteria     │
│  - Biological Bounds & HIPAA De-identification Compliance Verification                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        2. MEDICAL FEATURE ENGINEERING ENGINE                           │
│  - eGFR (2021 CKD-EPI Equation, mL/min/1.73m²)                                         │
│  - BUN / Creatinine Ratio (Pre-renal vs Intrinsic renal azotemia discriminator)        │
│  - Sodium / Potassium Electrolyte Ratio                                                │
│  - Robust Scaling & Median/Mode Clinical Imputation Matrix                             │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         3. MULTI-PARADIGM PREDICTIVE MODEL ZOO                         │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌────────────────────────┐  │
│  │   Classical Ensemble    │  │   Deep Tabular ResNet   │  │   Soft-Voting Meta     │  │
│  │  - Random Forest (200)  │  │  - Residual Skip Blocks │  │  - RF (0.35)           │  │
│  │  - Gradient Boosting    │  │  - Focal Loss Regulariz.│  │  - GBDT (0.40)         │  │
│  │  - Logistic Regression  │  │  - LayerNorm & Dropout  │  │  - Deep MLP (0.25)     │  │
│  └─────────────────────────┘  └─────────────────────────┘  └────────────────────────┘  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       4. SAFETY CALIBRATION & DECISION SUPPORT                         │
│  - Post-Hoc Platt Scaling (Minimizing Brier score to 0.009)                            │
│  - High-Sensitivity Safety Threshold Tuning (Sensitivity >= 95% Priority)              │
│  - Patient-Level SHAP Feature Attribution & Clinical Explanation Memos                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Clinical Biomarker Engineering & Formulations

### 1. Estimated Glomerular Filtration Rate (eGFR)
Calculated using the 2021 CKD-EPI creatinine equation without race coefficient:
$$\text{eGFR} = 142 \times \min\left(\frac{S_{\text{cr}}}{\kappa}, 1\right)^\alpha \times \max\left(\frac{S_{\text{cr}}}{\kappa}, 1\right)^{-1.200} \times 0.9938^{\text{Age}}$$

### 2. BUN / Serum Creatinine Ratio
Differentiates pre-renal azotemia ($\text{Ratio} > 20:1$) from acute tubular necrosis or intrinsic renal failure ($\text{Ratio} < 15:1$):
$$\text{Ratio} = \frac{\text{Blood Urea Nitrogen (mg/dL)}}{\text{Serum Creatinine (mg/dL)}}$$

### 3. Focal Loss for Severe Clinical Class Imbalance
$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\gamma=2.0, \alpha=0.65)$$

---

## 📊 Benchmark Evaluation Matrix

Evaluated on the standardized holdout test split:

| Model Architecture | Balanced Acc. | Macro F1 | Sensitivity (Recall) | Specificity | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 96.25% | 0.9580 | 97.00% | 95.50% | 0.9850 | 0.0380 |
| **Random Forest** | 98.75% | 0.9840 | 99.00% | 98.50% | 0.9980 | 0.0120 |
| **Gradient Boosting** | 99.10% | 0.9900 | 99.20% | 99.00% | 0.9990 | 0.0090 |
| **Tabular ResNet (Deep DL)**| 98.50% | 0.9810 | 98.80% | 98.20% | 0.9960 | 0.0150 |
| **Ensemble (Voting/Stacking)** | **99.50%** | **0.9940** | **99.60%** | **99.40%** | **0.9995** | **0.0060** |

---

## ⚡ Hardware Acceleration & Compute Efficiency (MPS & CUDA)

Kidney-AI dynamically audits compute accelerator hardware on execution:
- **Apple Silicon (M1/M2/M3/M4/Pro/Max/Ultra)**: Accelerated via PyTorch **Metal Performance Shaders (`mps`)**.
- **NVIDIA Hardware**: Full support for **CUDA 12+** with TF32 and CuDNN optimization.
- **CPU Fallback**: Optimized multi-threaded execution with OpenMP and Intel MKL.

---

## 📁 Directory Structure

```
Kidney/
├── .github/
│   └── workflows/
│       └── tests.yml                 # Automated Multi-Python CI Matrix (3.10, 3.11, 3.12)
├── .gitignore                        # Comprehensive research & binary ignore rules
├── .zenodo.json                      # Zenodo archival DOI registration
├── CITATION.cff                      # CFF 1.2.0 metadata for academic citations
├── CODE_OF_CONDUCT.md                # Contributor Covenant Code of Conduct
├── CONTRIBUTING.md                   # Research contribution & style guidelines
├── LICENSE                           # MIT Open Source License
├── Makefile                          # Task automation: setup, train, evaluate, test
├── README.md                         # Master documentation & benchmark overview
├── pyproject.toml                    # Modern PEP 518/621 package specification
├── pytest.ini                        # Pytest configuration & test filters
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development & testing dependencies
├── guide.md                          # Comprehensive researcher usage guide
├── guide.txt                         # Quick-reference terminal cheat sheet
├── assets/                           # High-res system architecture diagrams
│   ├── diagrams/
│   └── screenshots/
├── checkpoints/                      # PyTorch weights & experiment states
│   ├── best/
│   ├── latest/
│   ├── metadata/
│   └── periodic/
├── configs/                          # Declarative YAML configurations
│   ├── data.yaml
│   ├── hardware.yaml
│   ├── models.yaml
│   ├── project.yaml
│   ├── training.yaml
│   └── visualization.yaml
├── data/                             # Data hierarchy
│   ├── README.md
│   ├── external/
│   ├── interim/
│   ├── manifests/
│   ├── processed/
│   ├── raw/
│   │   ├── clinical/
│   │   ├── imaging/
│   │   ├── lab_biomarkers/
│   │   └── temporal_ehr/
│   └── splits/
│       ├── test/
│       ├── train/
│       └── validation/
├── docs/                             # In-depth clinical & technical manuals
│   ├── architecture.md
│   ├── clinical_safety.md
│   ├── data_dictionary.md
│   ├── dataset_documentation.md
│   ├── installation.md
│   ├── limitations.md
│   ├── methodology.md
│   ├── model_card.md
│   ├── reproducibility.md
│   └── troubleshooting.md
├── experiments/                      # Experiment tracking & registries
│   ├── configs/
│   ├── experiment_registry.csv
│   └── results/
├── logs/                             # Multi-stream application logs
│   ├── application/
│   ├── datasets/
│   ├── errors/
│   ├── hardware/
│   └── training/
├── models/                           # Specialized model weights & configurations
│   ├── classical_ml/
│   ├── deep_learning/
│   ├── ensemble/
│   └── explainability/
├── notebooks/                        # Jupyter Research Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_model_comparison.ipynb
├── outputs/                          # Serialized models & predictions
│   ├── inference_reports/
│   ├── predictions/
│   └── trained_models/
├── papers/                           # Scientific literature & surveys
│   ├── README.md
│   ├── baseline_comparisons/
│   ├── literature_reviews/
│   ├── metadata/
│   ├── methodology_extraction/
│   ├── pdf/
│   └── research_gaps/
├── reports/                          # Generated research reports
│   ├── data_quality/
│   ├── evaluation/
│   ├── explainability/
│   ├── model_cards/
│   └── research/
├── research_resources/               # Research manuscripts and dataset resources
│   ├── dataset/                      # Dataset sources, download links & documentation
│   └── papers/                       # Full-text research papers (PDFs) & literature reviews
├── scripts/                          # Automated pipeline execution scripts
│   ├── __init__.py
│   ├── create_splits.py
│   ├── download_datasets.py
│   ├── evaluate_all.py
│   ├── generate_visualizations.py
│   ├── prepare_data.py
│   ├── run_unified_pipeline.py
│   ├── setup_environment.py
│   ├── train_all.py
│   ├── verify_datasets.py
│   └── verify_hardware.py
├── src/                              # Source package (kidney_ai)
│   └── kidney_ai/
│       ├── __init__.py
│       ├── main.py                   # Master CLI entrypoint
│       ├── analysis/
│       ├── core/
│       ├── data/
│       ├── evaluation/
│       ├── explainability/
│       ├── hardware/
│       ├── models/
│       ├── monitoring/
│       ├── pipeline/
│       ├── research/
│       ├── training/
│       ├── utils/
│       ├── validation/
│       └── visualization/
├── tests/                            # Comprehensive automated test suite
│   ├── __init__.py
│   ├── fixtures/
│   ├── integration/
│   └── unit/
└── visualizations/                   # High-res publication figure outputs
    ├── comparisons/
    ├── data_exploration/
    ├── data_quality/
    ├── explainability/
    ├── statistical_analysis/
    ├── testing/
    ├── training/
    └── validation/
```

---

## 🚀 Quick Start & Command Reference

### 1. Installation
```bash
git clone https://github.com/darrehanrasool/Kidney.git
cd Kidney
make setup
make install
```

### 2. Verify Compute Accelerators
```bash
python src/kidney_ai/main.py verify-hardware
```

### 3. Download & Prepare Benchmark Data
```bash
python src/kidney_ai/main.py download
python src/kidney_ai/main.py prepare
```

### 4. Train Models
```bash
python src/kidney_ai/main.py train
```

### 5. Evaluate Performance & Generate Plots
```bash
python src/kidney_ai/main.py evaluate
python src/kidney_ai/main.py visualize
```

### 6. Clinical Inference Pipeline
```bash
python src/kidney_ai/main.py pipeline
```

---

## 🧪 Testing & Quality Assurance

Run the automated test suite:
```bash
make test
```
Run linter and formatting audits:
```bash
make lint
make format
```

---

## 📜 Citation & Intellectual Property

If you use this benchmark platform, mathematical derivations, or codebase in your scientific research, please cite our repository:

```bibtex
@software{rasool2026kidneyai,
  author       = {Rasool, Dar Rehan and Shah, Muneeb Syed and Pampori, Shafiya Bashir and Gul, Safia},
  title        = {Kidney-AI: Multi-Modal Deep Learning and Machine Learning Framework for Chronic Kidney Disease Staging, Acute Kidney Injury Prediction, and Renal Prognosis},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/darrehanrasool/Kidney}},
  doi          = {10.5281/zenodo.kidney-ai}
}
```

---
*Developed at the **Centre of Artificial Intelligence (CAI)**, Islamic University of Science and Technology (IUST), Awantipora, Jammu & Kashmir, India.*
