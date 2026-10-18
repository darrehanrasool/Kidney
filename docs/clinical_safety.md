# Clinical AI Safety and Regulatory Boundary Document

## 1. Ethical & Regulatory Boundary Notice

> **IMPORTANT CLINICAL NOTICE:**
> KidneyCare-AI is strictly a research and educational platform for investigating machine learning architectures on tabular chronic kidney disease datasets.
>
> **Under no circumstances should KidneyCare-AI be interpreted or deployed as an FDA-, CE-, or MHRA-cleared clinical diagnostic device.**
> - It does **not** provide definitive medical diagnoses for individual patients.
> - It must **never** be used to recommend, initiate, adjust, or withdraw medical treatments or pharmacological therapies.
> - Model predictions represent statistical pattern associations, not clinical diagnoses.

---

## 2. Distinction Between Clinical Screening and Diagnosis

1. **Screening vs. Diagnosis**:
   - **Screening** identifies asymptomatic individuals in a population who may have an increased probability of having a disease.
   - **Diagnosis** is a definitive determination of disease etiology, requiring comprehensive evaluation by a licensed physician, clinical nephrologist, and formal laboratory tests (e.g., measured GFR, 24-hour urine protein, renal ultrasound, or renal biopsy).
2. **Distinct Clinical Entities**:
   - **Chronic Kidney Disease (CKD)**: Progressive abnormalities of kidney structure or function present for $>3$ months.
   - **Acute Kidney Injury (AKI)**: Sudden decline in renal function over hours to days.
   - **Kidney Image Classification**: Visual structural pathology (cysts, stones, tumors) evaluated via ultrasound or CT.
   - KidneyCare-AI addresses tabular screening patterns associated with CKD and must never be conflated with AKI emergency detection or oncological tumor grading.

---

## 3. Clinical Risk Analysis of Model Errors

In nephrological decision support, predictive errors incur asymmetrical clinical costs:

| Error Type | Clinical Reality | Immediate Risk | Downstream Harm |
| :--- | :--- | :--- | :--- |
| **False Negative (FN)** | Patient with active CKD predicted as "NotCKD" | Missed early intervention; lack of ACEi/ARB or SGLT2i renoprotective therapy | Accelerated decline to end-stage renal disease (ESRD); unmanaged uremic complications |
| **False Positive (FP)** | Healthy patient predicted as "CKD" | Patient anxiety; unnecessary repeat blood draws | Inappropriate specialist referrals; potential complications from invasive diagnostic procedures |

**KidneyCare-AI Mitigation Strategy**:
- Prioritizes high Sensitivity (Recall $\ge 98\%$) to minimize False Negatives while reporting Specificity, Positive Predictive Value (Precision), and Brier calibration scores.
- Incorporates probability calibration so clinicians receive calibrated uncertainty intervals rather than binary predictions.

---

## 4. Human-in-the-Loop Protocol

- All algorithmic outputs must be reviewed by qualified medical personnel.
- Predictions must be correlated with patient clinical history, physical examination, and standardized serum chemistry.
