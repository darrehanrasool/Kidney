# System Architecture: KidneyCare-AI Platform

## 1. High-Level System Architecture

KidneyCare-AI is engineered as a research-grade, modular clinical machine learning platform designed for structured tabular chronic kidney disease detection, model comparison, probability calibration, and explainability.

```mermaid
flowchart TD
    subgraph Data Layer ["Data Ingestion & Preprocessing Layer"]
        A1["Primary Benchmark: UCI Chronic Kidney Disease (n=400)"] --> B1["Data Validator & Quality Auditor"]
        A2["Synthetic Verification Cohort (Zero-Network)"] --> B1
        B1 --> B2["Deduplication & Data Hygiene"]
        B2 --> B3["Clinical Feature Engineering (eGFR MDRD, BUN/Cr, Anion Gap)"]
        B3 --> B4["Leakage-Safe Splitting (Stratified 70/15/15 Train/Val/Test)"]
    end

    subgraph Hardware Layer ["Hardware Acceleration & Telemetry"]
        H1["Hardware Detector"] --> H2["Device Selection (Apple MPS / NVIDIA CUDA / CPU)"]
        H2 --> H3["Memory Monitor & Resource Bounds"]
    end

    subgraph Model Layer ["Model Zoo & Clinical Inference Engine"]
        M1["Classical Tier (Random Forest, Gradient Boosting, SVM, Logistic Regression)"]
        M2["Deep Learning Tier (Tabular MLP with BatchNorm & Dropout)"]
        M3["Ensemble Tier (Calibrated Soft-Voting Classifier)"]
    end

    subgraph Validation Layer ["Clinical Validation & Calibration Engine"]
        V1["Threshold Tuning (Youden's J / F1 on Validation Data)"]
        V2["Metrics Suite (Accuracy, Sensitivity, Specificity, F1, ROC-AUC, PR-AUC, Brier)"]
        V3["Probability Calibration (Platt Sigmoid / Isotonic Regression + ECE Audit)"]
        V4["Subgroup Auditing (Stratified by Age, Hypertension, Diabetes)"]
        V5["Statistical Testing (McNemar & Bootstrap 95% CIs)"]
    end

    subgraph Explainability Layer ["Explainable AI & Clinical Reporting"]
        E1["Tree & Permutation Feature Importance"]
        E2["SHAP Biomarker Attribution"]
        E3["Automated Clinical Diagnostic Reports & 10+ Visualizations"]
    end

    B4 --> Model Layer
    Hardware Layer -.-> Model Layer
    Model Layer --> Validation Layer
    Model Layer --> Explainability Layer
```

## 2. Component Breakdown

### 2.1 Core Subsystem (`src/kidneycare/core/`)
- `config.py`: Hierarchical YAML configuration loader with validation of splits and hyperparameter bounds.
- `constants.py`: Feature names, clinical reference intervals, target mappings, and ethical boundary notices.
- `exceptions.py`: Custom typed exception taxonomy (`DatasetValidationError`, `DataLeakageError`, `ModelTrainingError`).
- `reproducibility.py`: Deterministic seed controllers for Python `random`, `numpy`, and `torch`.

### 2.2 Data Pipeline Subsystem (`src/kidneycare/data/`)
- `dataset_registry.py`: Central catalog of supported datasets (UCI-CKD, Synthetic-CKD-Benchmark).
- `dataset_downloader.py`: Resumable downloader with SHA-256 verification and offline synthetic benchmark generator.
- `data_validator.py`: Schema validation, checksum verification, and clinical range violation checks.
- `data_quality.py`: Generates missingness and biomarker distribution reports.
- `deduplication.py`: Removes exact and duplicate patient observations.
- `missing_values.py`: Fits median/mode imputers strictly on training data without test leakage.
- `feature_engineering.py`: Computes 4-variable MDRD eGFR, BUN/Cr ratio, and electrolyte imbalance scores.
- `preprocessing.py`: Scikit-learn compatible clinical transformer with scaling, imputation, and encoding.
- `data_splitting.py`: Mutual disjoint index verification ensuring zero train/val/test overlap.

### 2.3 Model Architecture Subsystem (`src/kidneycare/models/`)
- `classical/`: Logistic Regression, Random Forest, Support Vector Classifier, and Gradient Boosting.
- `deep_learning/`: Tabular Multilayer Perceptron (`TabularMLP`) with Batch Normalization, Dropout, and Softmax output.
- `ensemble/`: Soft-voting ensemble combining diverse classifiers.
- `model_factory.py`: Instantiates models by name and configuration.

### 2.4 Training & Validation (`src/kidneycare/training/` & `src/kidneycare/validation/`)
- Full checkpointing with atomic file writes, early stopping monitoring validation F1, and learning rate scheduling.
- Probability calibration (Platt Scaling and Isotonic Regression) ensuring reliable risk probabilities.
- Optimal threshold selection using validation data alone.
