# Benchmark Datasets & Cohort Documentation

## Supported Cohorts

### 1. UCI Chronic Kidney Disease Dataset
- **Records**: 400 patient instances (250 CKD, 150 non-CKD)
- **Features**: 24 clinical attributes (14 numerical lab measurements, 10 categorical signs & symptoms)
- **Target**: Binary classification (`ckd` vs. `notckd`)
- **Missing Data**: Handled via iterative multivariate chained imputation (MICE) and clinical median substitution.

### 2. Multi-Center Renal Function & Longitudinal Cohort
- **Focus**: Multi-stage classification according to KDIGO guidelines (Stage 1 to Stage 5 kidney failure).
- **Temporal Signals**: Longitudinal serial serum creatinine, BUN, and urine output streams.

## Integrity Verification
All datasets downloaded or staged in `data/raw/` are automatically validated against cryptographic SHA-256 hashes registered in `data/manifests/`.
