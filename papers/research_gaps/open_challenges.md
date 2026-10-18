# Open Challenges and Research Gaps in Kidney Disease AI

This document highlights critical methodological weaknesses identified in existing chronic kidney disease machine learning literature and outlines research hypotheses investigated in KidneyCare-AI.

---

## 1. Data Leakage and Evaluation Flaws

- **Threshold Optimization on Test Sets**: Many published benchmarks select classification decision thresholds using the test set, creating significant optimistic bias.
- **Whole-Dataset Preprocessing Leakage**: Several earlier studies fitted imputation models, normalizers, or feature selectors across all 400 instances before splitting, thereby leaking target-correlated information into the test set.
- **KidneyCare-AI Safeguard**: Strict separation of fit and transform steps: all scalers, encoders, imputers, and threshold optimizers are fitted strictly on training or validation sets.

---

## 2. Generalizability and Population Shift

- **Single-Center Retrospective Bias**: The canonical UCI CKD dataset originates from Apollo Hospitals in India (collected ~2015). Biomarker reference ranges and diagnostic practices vary across global healthcare systems.
- **Demographic Imbalance**: Datasets rarely record ethnicity or longitudinal socioeconomic factors, which heavily influence CKD progression.
- **KidneyCare-AI Safeguard**: Explicit subgroup auditing across age, hypertension, and diabetes strata.

---

## 3. Probability Calibration in Healthcare

- **Overconfident Neural Networks & Ensembles**: Tree ensembles often push probabilities toward 0 or 1 without representing true posterior risks.
- **KidneyCare-AI Safeguard**: Integrated Platt scaling (CalibratedClassifierCV) and Expected Calibration Error (ECE) auditing.

---

## 4. Interpretability vs. Causality

- **Misinterpreting Feature Importance**: High attribution score for serum creatinine or anemia does not establish that lowering hemoglobin treats the underlying glomerulosclerosis.
- **KidneyCare-AI Safeguard**: Explicit clinical safety disclaimers separating association from clinical etiology.
