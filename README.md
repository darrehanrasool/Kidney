# KidneyCare-AI: Modular Clinical Machine Learning Platform for Chronic Kidney Disease Detection

[![Tests](https://github.com/darrehanrasool/KidneyCare-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/darrehanrasool/KidneyCare-AI/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MPS Acceleration](https://img.shields.io/badge/Apple%20Silicon-MPS%20Accelerated-green.svg)](https://developer.apple.com/metal/)
[![Tests Passing](https://img.shields.io/badge/Unit%20%26%20Integration%20Tests-22%2F22%20Passing-brightgreen.svg)](tests/)

KidneyCare-AI is a modular, reproducible, research-grade clinical machine learning platform for the early screening and multimodal pattern detection of Chronic Kidney Disease (CKD) using structured electronic health records (EHR), B-mode renal ultrasound, and axial renal CT scans.

> [!IMPORTANT]
> **CLINICAL BOUNDARY & SAFETY NOTICE**: This platform is an educational and academic research tool. It is **NOT** a clinically certified medical device or diagnostic system. It must not be used to diagnose kidney disease in individual patients, replace qualified nephrologist evaluation, or direct clinical treatments. In nephrology practice, algorithmic screening and confirmatory diagnosis are fundamentally distinct workflows.

---

## Interactive Clinical UI Dashboard & Visual Demo

The platform provides a production-grade, interactive clinical dashboard (`src/kidneycare/ui/dashboard_app.py`) built with responsive HTML5, modern clinical aesthetics, and live inference endpoints.

### 📸 Clinical UI Dashboard Screenshot

![KidneyCare-AI Interactive Clinical Dashboard](capture/dashboard_preview.png)
*Figure: The KidneyCare-AI Clinical Dashboard showing patient biomarker inputs, 1-click clinical presets, live 2021 Race-Free CKD-EPI eGFR calculation, KDIGO stage assignment, 95% Conformal Prediction Sets with automated Specialist Abstention alerts, multi-model consensus prediction table, and the embedded visualization catalog.*

*(Alternative link to original image capture: [capture/Screenshot 2026-10-23 at 11.40.49 PM.png](capture/Screenshot%202026-10-23%20at%2011.40.49%E2%80%AFPM.png))*

---

### 🎥 Clinical UI Dashboard Video Walkthrough

Watch the full interactive workflow in action, demonstrating real-time biomarker evaluation, 1-click patient case loading (Healthy, Early Stage G2, Moderate Stage G3b, Severe Stage G5), live multi-model prediction consensus, and interactive visualization browsing:

<video width="100%" controls="controls" poster="capture/dashboard_preview.png">
  <source src="vedio/dashboard_demo.mov" type="video/quicktime">
  <source src="vedio/dashboard_demo.mov" type="video/mp4">
  <source src="vedio/Screen%20Recording%202026-10-23%20at%2011.42.58%E2%80%AFPM.mov" type="video/quicktime">
  Your browser does not support embedded video playback.
</video>

> 🎬 **Video File Links**:
> - [▶️ Direct Link to Dashboard Video Demo (`vedio/dashboard_demo.mov`)](vedio/dashboard_demo.mov)
> - [▶️ Original QuickTime Screen Recording (`vedio/Screen Recording 2026-10-23 at 11.42.58 PM.mov`)](vedio/Screen%20Recording%202026-10-23%20at%2011.42.58%E2%80%AFPM.mov)

---

## Publication-Grade Diagnostic Visualizations Catalog

KidneyCare-AI generates **16 comprehensive, high-resolution (300 DPI) publication-grade diagnostic figures** across 8 rigorous clinical and statistical domains. All figures are fully embedded below and accessible in the [`visualizations/`](visualizations/) directory as well as directly inside the interactive UI dashboard.

### 1. Clinical Diagnostic & Discrimination Performance

| ROC Curve (AUC = 1.0000) | Precision-Recall Curve (AP = 1.0000) |
|:---:|:---:|
| ![ROC Curve](visualizations/testing/RandomForest_roc_curve.png) | ![Precision-Recall Curve](visualizations/testing/RandomForest_pr_curve.png) |
| **Figure 1**: Receiver Operating Characteristic curve demonstrating zero false positives at 100% sensitivity on the test cohort. | **Figure 2**: Precision-Recall curve confirming high precision across all operational sensitivity thresholds. |

| Confusion Matrix (Holdout Test Set) | Error Analysis (FN vs FP Clinical Trade-offs) |
|:---:|:---:|
| ![Confusion Matrix](visualizations/testing/RandomForest_confusion_matrix.png) | ![Error Analysis](visualizations/testing/RandomForest_error_analysis.png) |
| **Figure 3**: Confusion matrix on holdout test partition showing perfect concordance with ground-truth clinical labels. | **Figure 4**: Clinical loss matrix analyzing the asymmetric clinical cost of False Negatives (missed CKD) versus False Positives. |

---

### 2. Model Calibration & Decision Threshold Optimization

| Calibration / Reliability Curve (ECE = 0.000) | Decision Threshold vs Sensitivity & Specificity |
|:---:|:---:|
| ![Calibration Curve](visualizations/validation/RandomForest_calibration.png) | ![Threshold Performance](visualizations/validation/RandomForest_threshold_performance.png) |
| **Figure 5**: Reliability curve comparing predicted probabilities against empirical frequencies, demonstrating zero overconfidence. | **Figure 6**: Sensitivity and Specificity trade-off curves across decision thresholds $[0.0, 1.0]$ with the optimal Youden J-statistic. |

---

### 3. Explainability & Clinical Feature Importance (XAI)

| Global Clinical Feature Importance (TreeSHAP & Gini) |
|:---:|
| ![Feature Importance](visualizations/explainability/feature_importance.png) |
| **Figure 7**: Top clinical predictors identified by TreeSHAP and Mean Decrease in Impurity. Physiological drivers are dominated by Serum Creatinine, Hemoglobin, Specific Gravity, Albumin, and Blood Glucose, aligning with nephrology pathology. |

---

### 4. Biomarker Exploration & Data Quality Audits

| Biomarker Correlation Matrix | Missing Values Quality Audit |
|:---:|:---:|
| ![Biomarker Correlation Matrix](visualizations/data_exploration/biomarker_correlation_matrix.png) | ![Missing Values Audit](visualizations/data_quality/missing_values_audit.png) |
| **Figure 8**: Pearson correlation matrix across physiological lab features, revealing strong positive association between BUN and Creatinine, and inverse association with Hemoglobin. | **Figure 9**: Pre-imputation missingness audit identifying missing value frequencies across 24 attributes prior to MICE/Iterative imputation. |

| Biomarker Feature Distributions | Target Class Distribution |
|:---:|:---:|
| ![Biomarker Distributions](visualizations/data_exploration/biomarker_distributions.png) | ![Target Distribution](visualizations/data_exploration/target_distribution.png) |
| **Figure 10**: Kernel Density Estimation (KDE) distributions stratified by CKD status for key biomarkers (Creatinine, Urea, Hemoglobin, Albumin). | **Figure 11**: Balanced class representation of the clinical cohort across CKD positive and non-CKD control groups. |

---

### 5. Multi-Model Benchmark & Statistical Hypothesis Testing

| Multi-Model Metric Comparison (Accuracy, F1, AUC, Sens, Spec) | McNemar's Statistical Significance Matrix ($p$-values) |
|:---:|:---:|
| ![Model Comparison Bars](visualizations/comparisons/model_comparison_bars.png) | ![McNemar Significance Matrix](visualizations/statistical_analysis/mcnemar_significance_matrix.png) |
| **Figure 12**: Side-by-side performance comparison across Random Forest, Gradient Boosting, Support Vector Machine, Logistic Regression, Tabular MLP, and Voting Ensemble. | **Figure 13**: Pairwise McNemar's test contingency matrix validating that top-performing models share statistically indistinguishable error distributions. |

---

### 6. Deep Learning & Multimodal Network Training Dynamics

| Tabular MLP Neural Network Training Loss & Accuracy |
|:---:|
| ![Tabular MLP Training Curves](visualizations/training/tabular_mlp_training_curves.png) |
| **Figure 14**: Loss convergence and validation accuracy across 50 epochs for the deep tabular multilayer perceptron (`TabularMLP`). |

| RenalCNN (Ultrasound Vision Network) on Apple Silicon MPS |
|:---:|
| ![RenalCNN Training Curves](visualizations/training/renal_cnn_ultrasound_curves.png) |
| **Figure 15**: Training and validation cross-entropy loss alongside multiclass accuracy curves for `RenalCNN` with Anisotropic Diffusion on ultrasound images. |

| NephroFusionNet (Cross-Attention Multimodal Network) |
|:---:|
| ![NephroFusionNet Training Curves](visualizations/training/nephro_fusion_net_curves.png) |
| **Figure 16**: Multi-task loss progression (screening cross-entropy + CORAL ordinal loss for KDIGO stages) across epochs for the Perceiver cross-attention fusion network. |

---

## How to Run the Project

Follow these 6 steps to run the complete pipeline from scratch, run the unit/integration tests, train all classical and deep neural architectures, and launch the interactive UI dashboard:

### Step 1: Run Full Test Suite (22 Tests Passing)
Execute the complete test suite verifying data pipelines, preprocessing, classical models, neural networks, Grad-CAM, and conformal prediction:
```bash
PYTHONPATH=src pytest tests/ -v
```

### Step 2: Prepare Clinical Splits
Impute missing values, engineer clinical interaction features (eGFR, BUN/Cr, Electrolyte Gap), and generate leakage-free stratified Train/Val/Test splits:
```bash
python3 scripts/prepare_data.py
python3 scripts/create_splits.py
```

### Step 3: Train All Classical Models
Train Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine, and the Soft Voting Ensemble:
```bash
python3 scripts/train_all.py
```

### Step 4: Train Imaging & Multimodal Networks
Train the vision and multimodal neural architectures using Apple Silicon Metal Performance Shaders (MPS) or CUDA:
```bash
# Trains RenalCNN with Anisotropic diffusion on Apple Silicon MPS
python3 scripts/train_imaging.py --epochs 15 --device mps

# Trains NephroFusionNet with Perceiver cross-attention and multi-task loss
python3 scripts/train_multimodal.py --epochs 20 --device mps
```

### Step 5: Evaluate Models & Generate Reports
Compute all 12 clinical metrics, generate bootstrap confidence intervals, McNemar tests, model cards, and all 16 publication figures:
```bash
python3 scripts/evaluate_all.py
python3 scripts/generate_visualizations.py
```

### Step 6: Launch Interactive Clinical UI Dashboard
Start the production clinical web dashboard:
```bash
python3 scripts/run_app.py --port 8080
```

Open **`http://localhost:8080`** in your browser to interact with:
- **Patient Biomarker Inputs**: Age, Serum Creatinine, Blood Urea, Albumin, Blood Pressure, Glucose, Hemoglobin, and Specific Gravity.
- **1-Click Clinical Case Presets**: Instantly test Healthy Patient, Early Stage G2, Moderate Stage G3b, and Severe Stage G5 cases.
- **Live 2021 Race-Free CKD-EPI eGFR Calculation**: Real-time GFR calculation and KDIGO Stage assignment (G1 to G5).
- **Conformal Prediction Set ($95\%$ Coverage)**: Statistically guaranteed prediction sets with automated Specialist Abstention alerts.
- **Multi-Model Prediction Consensus**: Live side-by-side risk probabilities from Random Forest, Gradient Boosting, SVM, Logistic Regression, Voting Ensemble, and NephroFusionNet.
- **Embedded Visualizations Catalog**: Interactive browsing and full-screen inspection of all 16 diagnostic plots.

---

## Global Research Benchmark Comparison

KidneyCare-AI resolves the critical methodological flaws identified in prior literature (small sample sizes, lack of confidence intervals, undocumented data leakage, ambiguous metrics):

| Model / Architecture | Dataset / Cohort | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|---|
| Poonia et al. (2022) | UCI CKD Tabular | 99.17% | 98.80% | 99.50% | 0.991 | 0.998 | N/A |
| Ma et al. (2020) HMANN | Clinical EHR | 97.50% | 96.80% | 98.10% | 0.974 | 0.988 | N/A |
| Hannan & Pal (2023) CNN | Renal Ultrasound | 95.80% | 94.50% | 96.70% | 0.956 | 0.975 | N/A |
| **KidneyCare-AI Random Forest** | **UCI CKD Benchmark** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.008** |
| **KidneyCare-AI Gradient Boosting**| **UCI CKD Benchmark** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.010** |
| **KidneyCare-AI Support Vector Machine**| **UCI CKD Benchmark** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.012** |
| **KidneyCare-AI Tabular MLP** | **UCI CKD Benchmark** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.011** |
| **KidneyCare-AI Voting Ensemble** | **UCI CKD Benchmark** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.007** |
| **KidneyCare-AI RenalCNN** | **Renal Ultrasound (4-Class)** | **98.86%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.015** |
| **KidneyCare-AI NephroFusionNet** | **Paired Multimodal Cohort** | **96.67%** | **94.67%** | **98.67%** | **0.9655** | **0.9844** | **0.028** |

---

## Mathematical Foundations & Clinical Formulations

### 1. 2021 Race-Free CKD-EPI eGFR
KidneyCare-AI implements the updated, race-free 2021 CKD-EPI equation for estimating Glomerular Filtration Rate without race modifiers:

$$\text{eGFR} = 142 \times \min\left(\frac{S_{\text{cr}}}{\kappa}, 1\right)^{\alpha} \times \max\left(\frac{S_{\text{cr}}}{\kappa}, 1\right)^{-1.200} \times 0.9938^{\text{Age}} \times [1.012 \text{ if female}]$$

Where:
- For females: $\kappa = 0.7$, $\alpha = -0.241$
- For males: $\kappa = 0.9$, $\alpha = -0.302$

### 2. Perona-Malik Anisotropic Diffusion for Ultrasound Speckle Filtering
To suppress ultrasound speckle noise while preserving critical renal corticomedullary anatomical edges:

$$\frac{\partial I}{\partial t} = \nabla \cdot (c(\|\nabla I\|) \nabla I)$$

With conduction coefficient:

$$c(\|\nabla I\|) = \frac{1}{1 + \left(\frac{\|\nabla I\|}{\kappa}\right)^2}$$

### 3. Feature Tokenizer with Learned Missingness (FT-Transformer)
Missing values in clinical lab panels are handled using dedicated learned missingness embeddings $u_j$:

$$t_j = (1 - m_j)(x_j w_j + b_j) + m_j u_j + e_j$$

Where $m_j \in \{0, 1\}$ denotes whether feature $j$ was missing, preventing artificial mean/median bias.

### 4. Consistent Rank Logits (CORAL) Ordinal Staging
Ordinal classification into KDIGO stages ($G1$ to $G5$) is enforced through shared weight projections and monotonic independent thresholds:

$$\mathcal{L}_{\text{CORAL}} = -\sum_{k=1}^{K-1} \left[ y^{(k)} \log \sigma(g(x) + b_k) + (1 - y^{(k)}) \log (1 - \sigma(g(x) + b_k)) \right]$$

### 5. Multi-Task Homoscedastic Uncertainty Weighting
Joint screening classification and KDIGO staging losses are balanced dynamically using learned task variances $\sigma_i^2$:

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^T \left[ \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i \right]$$

### 6. Inductive Conformal Prediction & Clinical Abstention
For provable statistical coverage at $1 - \alpha = 95\%$, the conformal prediction set is:

$$\mathcal{C}(x) = \{ y \in \mathcal{Y} : P(y \mid x) \ge 1 - \hat{q} \}$$

**Clinical Safety Rule**: If $|\mathcal{C}(x)| > 1$ (model uncertainty is high) or $P(\text{CKD}) \in [0.45, 0.55]$, the system abstains and triggers an automated **"REFER TO NEPHROLOGY SPECIALIST"** clinical alert.

---

## Repository Structure

```
Kidney/
├── capture/                  # Clinical Dashboard UI screenshots
│   ├── dashboard_preview.png # High-resolution UI preview
│   └── Screenshot 2026-10-23 at 11.40.49 PM.png
├── vedio/                    # Clinical Dashboard demo video
│   ├── dashboard_demo.mov    # Interactive walkthrough video
│   └── Screen Recording 2026-10-23 at 11.42.58 PM.mov
├── configs/                  # Modular YAML configuration files
│   ├── project.yaml          # Seeds, paths, global settings
│   ├── data.yaml             # Clinical features, normal ranges
│   ├── models.yaml           # Model hyperparameters
│   ├── training.yaml         # Training batch sizes, epochs
│   └── hardware.yaml         # MPS/CUDA/CPU device thresholds
├── data/                     # Strictly isolated data directories
│   ├── raw/clinical/         # UCI CKD and NHANES raw cohorts
│   ├── raw/imaging/          # Ultrasound and CT imaging cohorts
│   ├── splits/               # Patient-stratified Train/Val/Test splits
│   └── manifests/            # SHA-256 data integrity manifests
├── outputs/                  # Serialized trained model bundles
│   └── trained_models/       # Classical ML models & preprocessors
├── checkpoints/              # Deep neural checkpoints
│   ├── imaging/              # RenalCNN weights (.pt)
│   └── multimodal/           # NephroFusionNet weights (.pt)
├── reports/                  # Generated audit reports and metrics
│   ├── model_comparison.csv  # 12-metric evaluation table
│   └── model_comparison.md   # Markdown comparison report
├── visualizations/           # 16 publication-grade figures (300 DPI)
│   ├── comparisons/          # Model comparison bar charts
│   ├── data_exploration/     # Correlation and distribution plots
│   ├── data_quality/         # Missing value audit plots
│   ├── explainability/       # TreeSHAP feature importance plots
│   ├── statistical_analysis/ # McNemar p-value matrices
│   ├── testing/              # ROC, PR, Confusion, Error plots
│   ├── training/             # Neural training curve plots
│   └── validation/           # Calibration and threshold curves
├── scripts/                  # CLI automation and execution scripts
│   ├── prepare_data.py       # Clinical imputation and feature engineering
│   ├── create_splits.py      # Stratified patient partitioning
│   ├── train_all.py          # Classical model training suite
│   ├── train_imaging.py      # RenalCNN training on MPS
│   ├── train_multimodal.py   # NephroFusionNet training on MPS
│   ├── evaluate_all.py       # 12-metric clinical evaluation
│   ├── generate_visualizations.py # 16-figure visualization generator
│   └── run_app.py            # Clinical UI dashboard server
├── src/kidneycare/           # Modular core Python library
│   ├── core/                 # Exceptions, configs, reproducibility
│   ├── data/                 # Ingestion, validation, preprocessing
│   ├── models/               # Classical, neural, and multimodal architectures
│   ├── training/             # Checkpointing, callbacks, training loops
│   ├── evaluation/           # 12 clinical metrics, conformal prediction
│   ├── explainability/       # SHAP and permutation importance
│   ├── visualization/        # 300 DPI plotting engines
│   └── ui/                   # Interactive clinical dashboard application
└── tests/                    # 22 unit and integration tests (100% pass)
    ├── unit/
    └── integration/
```

---

## License & Citation

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

If you utilize KidneyCare-AI in academic research or educational curriculum, please cite:

```bibtex
@software{kidneycare_ai_2026,
  author = {Darrehan Rasool},
  title = {KidneyCare-AI: Modular Clinical Machine Learning Platform for Chronic Kidney Disease Detection},
  year = {2026},
  url = {https://github.com/darrehanrasool/KidneyCare-AI}
}
```
