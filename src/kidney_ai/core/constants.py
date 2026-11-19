"""Clinical reference ranges, target definitions, and project constants."""

# Clinical Target Constants
LABEL_CKD = 1
LABEL_NOT_CKD = 0

TARGET_MAPPING = {
    "ckd": LABEL_CKD,
    "ckd\t": LABEL_CKD,
    "notckd": LABEL_NOT_CKD,
    "notckd\t": LABEL_NOT_CKD,
    "1": LABEL_CKD,
    "0": LABEL_NOT_CKD,
    1: LABEL_CKD,
    0: LABEL_NOT_CKD
}

# Physiological Plausibility Ranges (Minimum, Maximum)
PHYSIOLOGICAL_BOUNDS = {
    "age": (1.0, 115.0),
    "blood_pressure": (40.0, 220.0),
    "specific_gravity": (1.000, 1.040),
    "blood_glucose_random": (30.0, 600.0),
    "blood_urea": (2.0, 300.0),
    "serum_creatinine": (0.2, 25.0),
    "sodium": (90.0, 180.0),
    "potassium": (1.5, 9.5),
    "hemoglobin": (2.0, 22.0),
    "packed_cell_volume": (10.0, 65.0),
    "white_blood_cell_count": (1000.0, 45000.0),
    "red_blood_cell_count": (1.5, 8.5),
}

# CKD Staging based on eGFR (KDIGO guidelines, mL/min/1.73m²)
CKD_STAGES = {
    "Stage 1": (90.0, float("inf")),
    "Stage 2": (60.0, 89.9),
    "Stage 3a": (45.0, 59.9),
    "Stage 3b": (30.0, 44.9),
    "Stage 4": (15.0, 29.9),
    "Stage 5": (0.0, 14.9),
}
