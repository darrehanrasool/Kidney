# Literature Review: Machine Learning for Chronic Kidney Disease Detection

## Executive Summary

Chronic Kidney Disease (CKD) affects approximately 10–13% of the global adult population, leading to progressive renal parenchyma destruction, electrolyte dysregulation, and eventual end-stage renal disease (ESRD). Early detection through clinical screening algorithms offers significant opportunities to arrest disease progression before irreversible renal fibrosis occurs.

This systematic literature review examines published methodologies, algorithmic architectures, biomarker choices, and validation protocols based on primary research papers in this domain:
- **Poonia et al. (2022)**, *Healthcare* 10(2): 371.
- **Ma, Sun, & Jing (2020)**, *Future Generation Computer Systems* 111: 17–26.
- **Hannan & Pal (2023)**, *Journal of Neurology and Neurorehabilitation Research*.
- **Kidney AI Research Atlas (2026)**, comprehensive comparative synthesis.

---

## 1. Landmark Studies in Chronic Kidney Disease Detection

### 1.1 Poonia et al. (2022) — Clinical Tabular Benchmark
- **Title**: *Intelligent Diagnostic Prediction and Classification Models for Detection of Kidney Disease*
- **Focus**: Tabular classification using the standard UCI Chronic Kidney Disease benchmark dataset (Apollo Hospitals, India; $n=400$ instances, 24 attributes).
- **Techniques**:
  - Investigated three distinct feature selection techniques: Chi-Square ($\chi^2$), ExtraTrees importance, and Recursive Feature Elimination (RFE).
  - Evaluated classical machine learning models: Random Forest (RF), Support Vector Machine (SVM), Logistic Regression (LR), Naive Bayes (NB), Decision Trees (DT), and K-Nearest Neighbors (KNN).
- **Reported Performance**:
  - Random Forest achieved an accuracy of 99.25% ($F_1 = 0.9925$) on an 80/20 train/test partition.
  - SVM achieved 97.50% accuracy.
  - Logistic Regression achieved 96.25% accuracy.
- **Key Clinical Finding**: Serum Creatinine (`sc`), Packed Cell Volume (`pcv`), Hemoglobin (`hemo`), Specific Gravity (`sg`), and Albuminuria (`al`) constituted the most discriminative biomarkers across all feature selection filters.

### 1.2 Ma et al. (2020) — Heterogeneous Modified ANN (HMANN)
- **Title**: *Detection and diagnosis of chronic kidney disease using deep learning-based heterogeneous modified artificial neural network*
- **Focus**: Multimodal detection operating across clinical internet of medical things (IoMT) platforms, integrating ultrasound imaging and clinical attributes.
- **Methodology**:
  - Preprocessing using Level Set segmentation, Gabor filtering, and Gray-Level Co-occurrence Matrix (GLCM) texture extraction.
  - Classification using a hybrid network: a modified Multilayer Perceptron with backpropagation (MLP-BP) coupled with an SVM decision boundary.
- **Reported Performance**:
  - Diagnostic accuracy of 96.8% with an $F_1$-score of 96.5% across 640 patient records.
- **Limitations**:
  - The ultrasound database is proprietary and unreleased.
  - Fusion between imaging textures and tabular laboratory biomarkers was performed without separate unimodal ablations.

### 1.3 Hannan & Pal (2023) — Ultrasonography 2D CNN
- **Title**: *Detection and classification of kidney disease using convolution neural network*
- **Focus**: Deep convolutional networks trained directly on B-mode renal ultrasonography.
- **Reported Performance**:
  - CNN classification accuracy of 87.4% on ultrasound images.
- **Limitations**:
  - Ultrasound image interpretation suffers from high operator variability, acoustic shadowing, and cannot assess glomerular filtration directly without laboratory serum biomarkers.

---

## 2. Key Methodological Comparisons

| Dimension | Poonia et al. (2022) | Ma et al. (2020) | Hannan & Pal (2023) | KidneyCare-AI (This Work) |
| :--- | :--- | :--- | :--- | :--- |
| **Data Modality** | Tabular Clinical ($n=400$) | Tabular + Ultrasound ($n=640$) | Ultrasound Imaging ($n=1200$) | Tabular Clinical + Modular Fusion |
| **Validation Strategy** | Single 80/20 train-test split | Hospital cohort holdout | 70/30 train-test split | Leakage-safe 70/15/15 stratified + Repeated 5-fold CV |
| **Models Evaluated** | RF, SVM, LR, NB, DT, KNN | HMANN (MLP-BP + SVM) | 2D CNN | RF, GB, SVM, LR, Tabular MLP, Voting Ensemble |
| **Probability Calibration** | Not assessed | Not assessed | Not assessed | Platt scaling + Isotonic regression + ECE audit |
| **Uncertainty Quantification** | None | None | None | Bootstrap 95% Confidence Intervals |
| **Subgroup Performance** | Not reported | Not reported | Not reported | Stratified across Age, Hypertension, Diabetes |
| **Code & Model Availability** | Proprietary script snippets | Closed source | Closed source | Open Source, fully reproducible test suite |

---

## 3. Clinical Research Gaps Identified in Existing Literature

1. **Test-Set Threshold Optimization (Data Leakage Risk)**: Multiple previous studies selected optimal classification thresholds or feature sets using the entire dataset or the test partition directly, inflating reported scores.
2. **Neglect of Probability Calibration**: In clinical decision support, calibrated probabilities are critical. A model that reports 99% accuracy but outputs overconfident, uncalibrated risk probabilities poses safety hazards.
3. **Absence of Uncertainty Bounds**: Point estimates of accuracy or sensitivity without confidence intervals conceal high sampling variance inherent to moderate sample cohorts ($n=400$).
4. **Lack of Demographic Subgroup Auditing**: Prior papers do not report whether diagnostic sensitivity degrades in elderly patients ($\ge 65$) or in patients with comorbid diabetic nephropathy.
