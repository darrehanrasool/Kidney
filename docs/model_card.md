# Model Card: Kidney-AI Predictor Zoo

## Model Details
- **Architecture Family**: Hybrid Ensemble (XGBoost + Random Forest + Tabular Deep ResNet)
- **Primary Input**: Clinical laboratory panel (serum creatinine, BUN, electrolytes, albumin, blood pressure, etc.)
- **Output**: Calibrated risk probability ($0.0 - 1.0$), CKD binary classification, and SHAP attribution vector.
- **License**: MIT License
- **Version**: 1.0.0

## Intended Use
- **Primary Use**: Investigational risk scoring and clinical decision support for early identification of renal impairment.
- **Out-of-Scope Use**: Autonomous diagnostic decision-making without clinician oversight.

## Evaluation Summary
- **Balanced Accuracy**: $99.1\%$
- **Macro F1-Score**: $0.990$
- **ROC-AUC**: $0.999$
- **PR-AUC**: $0.998$
- **Brier Calibration Score**: $0.009$
