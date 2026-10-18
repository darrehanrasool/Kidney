# Dataset Documentation: Kidney Disease Datasets & Modalities

## 1. Primary Benchmark: UCI Chronic Kidney Disease Dataset

- **Identifier**: `UCI-CKD` (UCI Repository ID: 330)
- **Institution**: Apollo Hospitals, Tamil Nadu, India.
- **Curators**: L. Rubini & Dr. P. Eswaran.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Target Variable**: `classification` (`ckd`: 250 patients, `notckd`: 150 patients).
- **Cohort Size**: 400 patient records.
- **Feature Dimensionality**: 24 clinical attributes (11 continuous, 13 categorical/ordinal).
- **Modality**: Tabular clinical chemistry, urinalysis, hematology, and clinical history.
- **Missingness Profile**: Contains realistic clinical missing values (~5–10% missing cells across various tests such as serum potassium, white blood cell count, and urinalysis pus cells).
- **Known Limitations**:
  - Single-center geographical collection.
  - Lack of longitudinal follow-up (cross-sectional snapshot).
  - Absence of patient identifiers (preventing multi-visit patient clustering).

---

## 2. Synthetic Clinical Verification Cohort (Zero-Network Mode)

- **Identifier**: `Synthetic-CKD-Benchmark`
- **Origin**: KidneyCare-AI Synthetic Benchmark Generator (`scripts/download_datasets.py --mode synthetic`).
- **Purpose**: Enables immediate offline training, hardware benchmarking, and continuous integration testing on resource-constrained systems without requiring network connectivity.
- **Statistical Fidelity**: Modeled from the empirical covariance matrix, physiological ranges, and missingness rates of the canonical UCI CKD dataset.

---

## 3. Secondary & Optional Datasets

### 3.1 Kaggle Chronic Kidney Disease Prediction Dataset
- **Modality**: Tabular clinical records (25 columns).
- **Access Conditions**: Requires Kaggle authentication and API key setup.

### 3.2 Kidney Ultrasound Benchmark (Optional Future Extension)
- **Modality**: B-mode renal ultrasonography imaging.
- **Sources**: Hannan & Pal (2023), Ma et al. (2020).
- **Status**: Optional multimodal extension. Does not block or constrain tabular clinical screening baselines.

---

## 4. Multi-Dataset Combination & Incompatibility Protocol

> **CRITICAL CLINICAL DATA PROTOCOL:**
> Different kidney datasets frequently use divergent:
> 1. Target definitions (eGFR $<60$ vs. ICD-10 codes vs. serum creatinine thresholds).
> 2. Laboratory measurement units (e.g., Creatinine in $\text{mg/dL}$ vs. $\mu\text{mol/L}$).
> 3. Urinalysis dipstick grading scales ($0-5$ vs. semi-quantitative trace/plus).
> 4. Patient populations (ambulatory outpatients vs. hospitalized intensive care patients).
>
> **Rules for Combining Datasets:**
> - **Never blindly concatenate** different datasets without explicit schema mapping, unit harmonization, and population compatibility verification.
> - When datasets lack compatible definitions, models must be trained and evaluated as **independent separate experiments** or evaluated exclusively for external transfer generalization.
