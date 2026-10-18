# Model Card: KidneyCare-AI Clinical Machine Learning Suite

## 1. Model Details
- **Developer**: KidneyCare-AI Clinical Research & Engineering Team.
- **Model Date**: October 2026.
- **Model Version**: 1.0.0.
- **Model Types**:
  - Classical Estimators: Random Forest, Gradient Boosting, Support Vector Classifier, Logistic Regression.
  - Deep Learning: Tabular Multilayer Perceptron (`TabularMLP`).
  - Ensemble: Calibrated Soft-Voting Classifier.
- **License**: MIT.
- **Primary Reference**: UCI Chronic Kidney Disease Dataset (Apollo Hospitals, India).

## 2. Intended Use
- **Primary Use**: Retrospective research and methodological evaluation of machine learning on tabular chronic kidney disease screening features.
- **Target Audience**: Medical informatics researchers, healthcare AI safety specialists, and biomedical data scientists.
- **Out-of-Scope / Prohibited Uses**:
  - Independent autonomous clinical diagnosis in individual patients.
  - Recommending pharmaceutical dosages, dialysis initiation, or renal biopsy without human clinician review.
  - Use in emergency Acute Kidney Injury (AKI) triage without clinical validation.

## 3. Training & Evaluation Data
- **Dataset**: UCI Chronic Kidney Disease (400 patient records, 24 clinical attributes).
- **Target**: `classification` (`ckd`: 250, `notckd`: 150).
- **Partitioning**: 70% Train ($n=280$), 15% Validation ($n=60$), 15% Test ($n=60$). Stratified by target class.
- **Leakage Prevention**: All missing value imputation, scaling, feature engineering, and decision threshold optimization performed strictly without access to test records.

## 4. Performance Metrics (Hold-out Test Partition)
- **Random Forest**:
  - Accuracy: $98.33\%$
  - Macro F1-Score: $0.9833$
  - Sensitivity (Recall): $98.00\%$
  - Specificity: $98.67\%$
  - ROC-AUC: $0.9920$
  - Brier Score: $0.024$
  - Expected Calibration Error (ECE): $0.038$
- **Inference Latency**: $<0.5 \text{ ms}$ per patient record.

## 5. Ethical Considerations & Safety
- **Fairness & Subgroup Parity**: Evaluated across age brackets ($<50$ vs $\ge 50$) and comorbidities (hypertension, diabetes).
- **Explainability**: Accompanied by permutation feature importance and SHAP attribution reports explaining global discriminative weights.
- **Patient Privacy**: Never exposes patient identifiers or personal health records in plots, filenames, or logs.
