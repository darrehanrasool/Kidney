# Known Limitations and Clinical Safety Boundaries

This document provides transparent, documented limitations regarding dataset constraints, algorithmic boundaries, and clinical translation hurdles.

---

## 1. Dataset Limitations

### 1.1 Retrospective Single-Center Cohort
- The primary benchmark dataset (UCI CKD) was collected retrospectively from Apollo Hospitals in Tamil Nadu, India (~2015).
- Local population demographics, dietary patterns (e.g. vegetarianism influencing baseline serum creatinine), and regional clinical referral practices may not reflect diverse international populations.
- External validation on multi-center hospital systems (e.g. UK Biobank, eICU, or MIMIC-IV) is essential before any clinical translational claims can be considered.

### 1.2 Modest Sample Size ($n=400$)
- While canonical in biomedical machine learning literature, 400 patient records provide limited statistical power for rare subgroup analyses or subtle non-linear interactions.
- Sampling variance in test sets ($n=60$) means that single misclassifications shift accuracy by $\sim 1.67\%$. Confidence intervals must always accompany point metrics.

### 1.3 Cross-Sectional Nature (No Longitudinal Follow-Up)
- The dataset captures a single clinical snapshot in time.
- Clinical nephrology guidelines (KDIGO) require persistent renal functional impairment for $\ge 3\text{ months}$ to formally diagnose Chronic Kidney Disease. A single elevated serum creatinine value cannot definitively distinguish pre-existing CKD from transient Acute Kidney Injury (AKI) or temporary dehydration.

### 1.4 Unspecified Laboratory Assay Standardization
- The original dataset records serum creatinine in $\text{mg/dL}$ but does not document whether assays were isotope dilution mass spectrometry (IDMS) traceable or used the older colorimetric Jaffe reaction. This uncertainty introduces slight variation into absolute eGFR equations.

---

## 2. Algorithmic and Modeling Boundaries

### 2.1 Tabular vs. Imaging Modalities
- The initial baseline system operates strictly on structured laboratory and clinical records.
- Although imaging support is designed modularly, the initial tabular pipeline does not inspect renal ultrasound images, renal Doppler resistive indices, or kidney biopsies.
- Tabular models cannot detect anatomical anomalies such as renal cysts, hydronephrosis, or nephrolithiasis unless reflected in laboratory biomarkers.

### 2.2 Correlation Does Not Imply Biological Causality
- High feature attribution scores (e.g. for serum creatinine, hemoglobin, or albuminuria) reflect empirical associations learned from training records.
- Modifying a feature value in a patient record (e.g. artificially raising hemoglobin via transfusion) does not directly treat the underlying glomerular or tubulointerstitial pathology.

### 2.3 Subgroup Performance Variations
- Performance metrics may vary between patients with isolated diabetic nephropathy versus hypertensive nephrosclerosis. Subgroup audits are provided in `reports/` and must be reviewed.
