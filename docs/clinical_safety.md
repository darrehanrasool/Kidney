# Clinical Safety & Risk Governance

## Clinical AI Safety Principles

Kidney-AI is developed as an **investigational decision-support platform** to assist qualified nephrologists and medical practitioners, **not** as an autonomous diagnostic device.

### 1. High-Sensitivity Threshold Optimization
- In renal failure and acute kidney injury prognostics, false negatives (failing to identify impending disease progression) carry far higher clinical risk than false positives.
- Decision thresholds are calibrated using Youden's J statistic with a high-sensitivity bias ($\ge 95\%$ sensitivity priority) to ensure early clinical alerting.

### 2. Probability Calibration
- Uncalibrated neural networks frequently output overconfident probabilities.
- All model probabilities undergo post-hoc calibration (Platt scaling / Isotonic regression) to ensure predicted probabilities accurately mirror empirical incidence rates.

### 3. Clinician-in-the-Loop Transparency
- Every prediction is paired with SHAP-based biomarker attributions detailing exactly which laboratory features (e.g., elevated serum creatinine, severe proteinuria) drove the risk score.
- Unreliable predictions flagged by out-of-distribution (OOD) biomarker detectors require mandatory manual review.
