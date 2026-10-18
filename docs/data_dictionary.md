# Clinical Data Dictionary: Chronic Kidney Disease Biomarkers

This data dictionary defines the 24 primary clinical attributes and 4 clinically engineered biomarkers utilized across KidneyCare-AI models, including measurement units, normal physiological reference intervals, and pathophysiological significance.

---

## 1. Primary Clinical Attributes (UCI CKD Standard)

| Attribute Code | Attribute Name | Type | Measurement Unit | Clinical Reference Interval | Pathophysiological Significance in CKD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | Patient Age | Numerical | Years | 1 – 120 | Renal functional reserve declines progressively with aging; incidence rises sharply in seniors ($\ge 60$). |
| `bp` | Blood Pressure | Numerical | mm/Hg | 60 – 120 | Sustained hypertension accelerates glomerular capillary hypertension and glomerulosclerosis. |
| `sg` | Urine Specific Gravity | Categorical / Ordinal | Ratio | 1.010 – 1.025 | Measures tubular concentrating ability; fixated low gravity (1.010, isosthenuria) marks end-stage tubular atrophy. |
| `al` | Urine Albumin | Categorical / Ordinal | Grades 0 to 5 | 0 (Negative) | Indicator of disrupted podocyte foot processes and glomerular filtration barrier breakdown. |
| `su` | Urine Sugar (Glucosuria) | Categorical / Ordinal | Grades 0 to 5 | 0 (Negative) | Reflects uncontrolled diabetic hyperglycemia or proximal renal tubular reabsorptive dysfunction. |
| `rbc` | Red Blood Cells in Urine | Categorical | normal / abnormal | normal | Microscopic hematuria suggests active glomerulonephritis or parenchymal trauma. |
| `pc` | Pus Cell in Urine (Pyuria) | Categorical | normal / abnormal | normal | Indicates active tubulointerstitial inflammation, pyelonephritis, or urinary tract infection. |
| `pcc` | Pus Cell Clumps | Categorical | present / notpresent | notpresent | Severe active parenchymal renal infection or interstitial nephritis. |
| `ba` | Bacteria in Urine | Categorical | present / notpresent | notpresent | Urinary bacterial colonization accelerating acute-on-chronic renal deterioration. |
| `bgr` | Blood Glucose Random | Numerical | mg/dL | 70 – 140 | Chronic hyperglycemia causes advanced glycation end-products and diabetic nephropathy. |
| `bu` | Blood Urea Nitrogen | Numerical | mg/dL | 7 – 20 | Accumulation of nitrogenous waste products from protein catabolism due to declining filtration. |
| `sc` | Serum Creatinine | Numerical | mg/dL | 0.6 – 1.2 | Hallmark endogenous marker of glomerular filtration rate; inverse exponential relationship to clearance. |
| `sod` | Serum Sodium | Numerical | mEq/L | 135 – 145 | Dysregulation indicates advanced disruption of renal medullary countercurrent concentration. |
| `pot` | Serum Potassium | Numerical | mEq/L | 3.5 – 5.0 | Severe hyperkalemia ($>5.5$) is a life-threatening consequence of impaired distal tubular potassium excretion. |
| `hemo` | Hemoglobin | Numerical | g/dL | 13.5 – 17.5 (M), 12.0 – 15.5 (F) | Normocytic normochromic anemia develops due to deficiency of peritubular capillary erythropoietin. |
| `pcv` | Packed Cell Volume (Hematocrit) | Numerical | % | 38 – 50 | Tracks red cell volume; drops in parallel with hemoglobin in renal anemia. |
| `wc` | White Blood Cell Count | Numerical | cells/$\mu$L | 4,000 – 11,000 | Marker of systemic chronic inflammation or intercurrent infectious decompensation. |
| `rc` | Red Blood Cell Count | Numerical | million/$\mu$L | 4.2 – 5.9 | Decreased production from erythropoietin deficiency. |
| `htn` | Hypertension Diagnosis | Categorical | yes / no | no | Etiology and exacerbating factor causing nephrosclerosis. |
| `dm` | Diabetes Mellitus | Categorical | yes / no | no | Leading worldwide cause of chronic kidney disease and end-stage renal disease. |
| `cad` | Coronary Artery Disease | Categorical | yes / no | no | Cardiorenal syndrome; high cardiovascular mortality in CKD patients. |
| `appet` | Appetite | Categorical | good / poor | good | Uremic anorexia develops from toxic retention of middle molecules. |
| `pe` | Pedal Edema | Categorical | yes / no | no | Sodium and water retention manifesting as peripheral dependent edema. |
| `ane` | Clinical Anemia | Categorical | yes / no | no | Overt clinical symptom of advanced renal failure. |
| `classification` | Diagnostic Target | Categorical | `ckd` / `notckd` | `notckd` | Ground truth diagnosis verified by clinical and laboratory evaluation. |

---

## 2. Clinically Derived Biomarkers

| Feature Name | Type | Unit | Formula | Clinical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `egfr_mdrd` | Float | mL/min/1.73 $\text{m}^2$ | $175 \times \text{sc}^{-1.154} \times \text{age}^{-0.203}$ | Standard clinical guideline metric for KDIGO staging ($<60$ indicates CKD). |
| `bun_cr_ratio` | Float | Ratio | $\text{bu} / \max(\text{sc}, 0.1)$ | Ratios $>20$ indicate prerenal volume contraction; $10-15$ indicates intrinsic parenchymal loss. |
| `anion_gap` | Float | Score | $|\text{sod} - 140| + 8 \times |\text{pot} - 4.5|$ | Distance from electrolyte homeostatic balance. |
| `cv_metabolic_risk` | Integer | Count (0-2) | $\mathbf{1}(\text{htn}) + \mathbf{1}(\text{dm})$ | Count of dual metabolic-cardiovascular drivers. |
