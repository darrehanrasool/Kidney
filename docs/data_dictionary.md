# Kidney-AI Clinical Data Dictionary

| Variable Identifier | Clinical Description | Data Type | Units / Valid Range | Normal Physiological Range |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Patient chronological age | Numerical | $1 - 100$ years | N/A |
| `blood_pressure` | Resting diastolic blood pressure | Numerical | $50 - 180$ mm/Hg | $60 - 80$ mm/Hg |
| `specific_gravity` | Urine specific gravity | Categorical/Num | $1.005 - 1.030$ | $1.010 - 1.025$ |
| `albumin` | Urine protein/albumin level | Ordinal | $0 - 5$ scale | $0$ (Negative) |
| `sugar` | Urine glucose level | Ordinal | $0 - 5$ scale | $0$ (Negative) |
| `blood_glucose_random` | Random serum glucose | Numerical | $50 - 500$ mg/dL | $70 - 140$ mg/dL |
| `blood_urea` | Blood urea nitrogen (BUN) | Numerical | $5 - 200$ mg/dL | $7 - 20$ mg/dL |
| `serum_creatinine` | Serum creatinine concentration | Numerical | $0.4 - 15.0$ mg/dL | $0.6 - 1.2$ mg/dL |
| `sodium` | Serum sodium electrolyte | Numerical | $100 - 170$ mEq/L | $135 - 145$ mEq/L |
| `potassium` | Serum potassium electrolyte | Numerical | $2.5 - 8.0$ mEq/L | $3.5 - 5.0$ mEq/L |
| `hemoglobin` | Whole blood hemoglobin | Numerical | $3.0 - 18.0$ g/dL | $12.0 - 17.5$ g/dL |
| `packed_cell_volume` | Hematocrit (PCV) | Numerical | $15 - 55$ % | $36 - 50$ % |
| `white_blood_cell_count` | Total leukocyte count | Numerical | $2,000 - 30,000$ /$\mu$L | $4,500 - 11,000$ /$\mu$L |
| `red_blood_cell_count` | Total erythrocyte count | Numerical | $2.0 - 7.0$ M/$\mu$L | $4.2 - 5.9$ M/$\mu$L |
| `hypertension` | History of systemic hypertension | Categorical | `yes` / `no` | `no` |
| `diabetes_mellitus` | History of diabetes mellitus | Categorical | `yes` / `no` | `no` |
| `coronary_artery_disease`| Coronary vascular disease | Categorical | `yes` / `no` | `no` |
| `appetite` | Appetite assessment | Categorical | `good` / `poor` | `good` |
| `pedal_edema` | Peripheral lower limb edema | Categorical | `yes` / `no` | `no` |
| `anemia` | Presence of clinical anemia | Categorical | `yes` / `no` | `no` |
| `classification` | Chronic Kidney Disease Status | Binary Target | `ckd` (1), `notckd` (0) | `notckd` |
