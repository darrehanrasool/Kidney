# Clinical Model Interpretability Report: Random Forest

> **CLINICAL NOTICE:** Feature importance rankings reflect associative statistical weights learned from the training distribution. They do **not** establish biological causality or direct pathogenic mechanisms.

## Top 10 Discriminative Biomarkers

| Rank | Feature Name | Attribution Score | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | `cv_metabolic_risk` | 0.1660 | Clinical laboratory or observational biomarker. |
| 2 | `hemo` | 0.1410 | Hemoglobin: Anemia of chronic kidney disease due to impaired erythropoietin production. |
| 3 | `sc` | 0.1208 | Serum Creatinine: Primary biomarker of glomerular filtration filtration decline. |
| 4 | `pcv` | 0.1159 | Packed Cell Volume (Hematocrit): Closely tied to chronic anemia and renal disease progression. |
| 5 | `egfr_mdrd` | 0.1114 | Estimated GFR (MDRD): Standard guideline filtration metric. |
| 6 | `al_0` | 0.0596 | Clinical laboratory or observational biomarker. |
| 7 | `bu` | 0.0349 | Blood Urea: Retention of nitrogenous waste metabolites. |
| 8 | `bgr` | 0.0332 | Blood Glucose Random: Indicator of diabetic nephropathy etiology. |
| 9 | `dm_no` | 0.0324 | Clinical laboratory or observational biomarker. |
| 10 | `htn_yes` | 0.0314 | Clinical laboratory or observational biomarker. |

## Methodological Considerations
- **Collinearity**: Correlated features (e.g. Hemoglobin and PCV) can share feature importance across tree splits.
- **Population Specificity**: Attributions reflect cohort demographics (Apollo Hospitals dataset, India) and may vary in diverse global populations.