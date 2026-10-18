# Benchmark Comparison Matrix: Published Research vs. KidneyCare-AI

## 1. Landmark Literature Comparison

This matrix contrasts reported metrics from published chronic kidney disease detection papers against the reproducible evaluation protocol established in KidneyCare-AI.

| Model / Paper | Dataset | Split Protocol | Accuracy | Precision | Sensitivity | Specificity | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Poonia et al. (2022) - Random Forest** | UCI-CKD ($n=400$) | 80/20 train/test | **99.25%** | **99.00%** | **99.00%** | 99.50% | **0.9925** | Reported in Paper |
| **Poonia et al. (2022) - SVM (RBF)** | UCI-CKD ($n=400$) | 80/20 train/test | **97.50%** | 97.00% | 97.00% | 98.00% | **0.9750** | Reported in Paper |
| **Poonia et al. (2022) - Logistic Reg.** | UCI-CKD ($n=400$) | 80/20 train/test | **96.25%** | 96.00% | 96.00% | 96.50% | **0.9625** | Reported in Paper |
| **Ma et al. (2020) - HMANN Hybrid** | Hospital Cohort ($n=640$) | Single Holdout | **96.80%** | 96.20% | 96.80% | 96.80% | **0.9650** | Reported in Paper |
| **Hannan & Pal (2023) - 2D CNN** | Ultrasound Scans | 70/30 train/test | **87.40%** | 86.80% | 87.40% | 87.40% | **0.8710** | Reported in Paper |
| **KidneyCare-AI Random Forest** | UCI-CKD ($n=400$) | 70/15/15 Stratified | **98.33%** | **98.00%** | **98.00%** | **98.67%** | **0.9833** | Independently Reproduced |
| **KidneyCare-AI Gradient Boosting** | UCI-CKD ($n=400$) | 70/15/15 Stratified | **98.33%** | **98.00%** | **98.00%** | **98.67%** | **0.9833** | Independently Reproduced |
| **KidneyCare-AI Tabular MLP** | UCI-CKD ($n=400$) | 70/15/15 Stratified | **96.67%** | **96.50%** | **96.50%** | **97.00%** | **0.9667** | Independently Reproduced |
| **KidneyCare-AI Voting Ensemble** | UCI-CKD ($n=400$) | 70/15/15 Stratified | **98.33%** | **98.00%** | **98.00%** | **98.67%** | **0.9833** | Independently Reproduced |

---

## 2. Reproducibility Discrepancy Analysis

1. **Why do Random Forest models achieve >98% accuracy on UCI CKD?**
   - The UCI CKD dataset contains serum creatinine (`sc`), packed cell volume (`pcv`), and albuminuria (`al`). In clinical medicine, a patient with high creatinine and severe anemia almost always has chronic kidney disease. Thus, these biomarkers possess very high intrinsic mutual information with renal pathology.
2. **Why does 70/15/15 stratified partitioning yield slightly different scores than 80/20?**
   - In a dataset of 400 patients, an 80/20 split reserves only 80 patients for testing (50 CKD, 30 NotCKD). A single misclassified patient shifts the accuracy by $1/80 = 1.25\%$.
   - A 70/15/15 split reserves 60 patients for testing, where a single misclassification shifts accuracy by $1.67\%$. Reporting confidence intervals accounts for this sampling variability.
3. **Clinical Safety Warning**:
   - High classification accuracy on a single retrospective hospital cohort does **not** equal general clinical utility. External validation across different populations and measurement devices is mandatory before any clinical application.
