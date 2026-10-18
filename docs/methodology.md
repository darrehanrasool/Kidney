# Methodology: Clinical Chronic Kidney Disease Machine Learning

## 1. Clinical Data Preprocessing & Hygiene

The clinical data processing pipeline adheres to strict medical data safety and leakage-prevention standards:

1. **Schema & Typographical Sanitization**:
   - Column names are trimmed and normalized to lowercase.
   - Categorical inconsistencies in raw datasets (such as tab characters in target labels like `'ckd\t'`) are cleaned into canonical labels (`'ckd'` and `'notckd'`).
2. **Clinical Missing Value Handling**:
   - Numerical missing values are imputed using medians computed strictly on the training partition.
   - Categorical missing values are imputed using mode values or explicit `'missing'` categories.
   - Crucially, rows with missing values are **not** silently deleted, which would bias the study toward patients who underwent complete battery testing.
3. **Clinical Feature Engineering**:
   - Derives guideline-directed nephrology indicators:
     - **eGFR (MDRD equation)**: Estimated glomerular filtration rate.
     - **BUN/Creatinine Ratio**: Biochemical indicator of prerenal volume contraction vs. intrinsic damage.
     - **Electrolyte Deviation Index**: Quantifies deviations in serum sodium and potassium.
     - **Cardiometabolic Risk Score**: Combined hypertension and diabetes burden.
4. **Leakage-Safe Scaling & Encoding**:
   - Continuous numerical features are scaled using `StandardScaler` fitted strictly on training data.
   - Categorical variables are encoded with `OneHotEncoder(handle_unknown='ignore')`.

---

## 2. Multi-Model Architecture Strategy

1. **Classical Machine Learning Baselines**:
   - **Logistic Regression**: Linear baseline with $L_2$ regularization; provides interpretable odds ratios.
   - **Random Forest**: Ensemble of 150 decision trees; captures non-linear biomarker interactions without overfitting.
   - **Support Vector Machine (SVM)**: RBF kernel soft-margin classifier with calibrated probability outputs.
   - **Gradient Boosting**: Sequential gradient boosting trees optimizing log-loss.
2. **Deep Learning**:
   - **Tabular Multilayer Perceptron (MLP)**: Deep neural architecture with Batch Normalization, Dropout ($0.2, 0.1$), and Cross-Entropy loss with label smoothing ($\epsilon = 0.05$).
3. **Voting Ensemble**:
   - Calibrated soft-voting combining Random Forest, Gradient Boosting, and Logistic Regression.

---

## 3. Evaluation and Clinical Calibration Protocol

- **3-Way Stratified Partitioning**: 70% Train, 15% Validation, 15% Test.
- **Mutual Disjoint Index Verification**: Guarantees zero sample overlap between splits.
- **Threshold Optimization**: Optimal decision threshold $\tau^*$ is selected on the validation partition using Youden's $J$ statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$). The test partition remains completely untouched until final evaluation.
- **Probability Calibration**: Validation probabilities are calibrated via Platt scaling (sigmoid) or isotonic regression, reducing Expected Calibration Error (ECE) and Brier scores.
- **Statistical Uncertainty**: 1,000-sample bootstrap confidence intervals are computed for all test metrics.
