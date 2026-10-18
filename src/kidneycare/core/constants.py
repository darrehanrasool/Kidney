"""KidneyCare-AI Clinical & System Constants.

Defines feature names, clinical reference intervals, target mappings,
and mandatory ethical/safety boundaries.
"""

from typing import Dict, List, Any

# Clinical Boundary Disclaimer
CLINICAL_BOUNDARY_DISCLAIMER: str = (
    "RESEARCH & EDUCATIONAL USE ONLY: KidneyCare-AI is designed exclusively for "
    "methodological evaluation and research in machine learning. It is not cleared "
    "for clinical diagnostic use, screening, or treatment guidance. Clinical decisions "
    "must always be made by licensed medical practitioners following established guidelines."
)

# Target Definition
TARGET_COLUMN: str = "classification"
POSITIVE_CLASS_LABEL: str = "ckd"
NEGATIVE_CLASS_LABEL: str = "notckd"

CLASS_MAPPING: Dict[str, int] = {
    "notckd": 0,
    "ckd": 1,
}

INVERSE_CLASS_MAPPING: Dict[int, str] = {v: k for k, v in CLASS_MAPPING.items()}

# Primary Clinical Features (UCI CKD 24 attributes)
NUMERICAL_FEATURES: List[str] = [
    "age",   # Age in years
    "bp",    # Blood Pressure in mm/Hg
    "bgr",   # Blood Glucose Random in mgs/dl
    "bu",    # Blood Urea in mgs/dl
    "sc",    # Serum Creatinine in mgs/dl
    "sod",   # Sodium in mEq/L
    "pot",   # Potassium in mEq/L
    "hemo",  # Hemoglobin in gms
    "pcv",   # Packed Cell Volume (%)
    "wc",    # White Blood Cell Count in cells/cumm
    "rc",    # Red Blood Cell Count in millions/cmm
]

CATEGORICAL_FEATURES: List[str] = [
    "sg",    # Specific Gravity: 1.005, 1.010, 1.015, 1.020, 1.025
    "al",    # Albumin: 0, 1, 2, 3, 4, 5
    "su",    # Sugar: 0, 1, 2, 3, 4, 5
    "rbc",   # Red Blood Cells: normal, abnormal
    "pc",    # Pus Cell: normal, abnormal
    "pcc",   # Pus Cell Clumps: present, notpresent
    "ba",    # Bacteria: present, notpresent
    "htn",   # Hypertension: yes, no
    "dm",    # Diabetes Mellitus: yes, no
    "cad",   # Coronary Artery Disease: yes, no
    "appet", # Appetite: good, poor
    "pe",    # Pedal Edema: yes, no
    "ane",   # Anemia: yes, no
]

ALL_BASE_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Engineered Clinical Biomarkers
ENGINEERED_FEATURES: List[str] = [
    "egfr_mdrd",     # Estimated Glomerular Filtration Rate (MDRD equation)
    "bun_cr_ratio",  # Blood Urea Nitrogen / Serum Creatinine ratio
    "anion_gap",     # Electrolyte balance surrogate
]

# Physiologically Plausible Reference Ranges
CLINICAL_RANGES: Dict[str, Dict[str, Any]] = {
    "age": {"min": 1.0, "max": 120.0, "unit": "years"},
    "bp": {"min": 40.0, "max": 240.0, "unit": "mm/Hg"},
    "bgr": {"min": 20.0, "max": 600.0, "unit": "mgs/dl"},
    "bu": {"min": 1.0, "max": 450.0, "unit": "mgs/dl"},
    "sc": {"min": 0.1, "max": 35.0, "unit": "mgs/dl"},
    "sod": {"min": 90.0, "max": 180.0, "unit": "mEq/L"},
    "pot": {"min": 1.5, "max": 12.0, "unit": "mEq/L"},
    "hemo": {"min": 2.0, "max": 25.0, "unit": "gms"},
    "pcv": {"min": 10.0, "max": 70.0, "unit": "%"},
    "wc": {"min": 1000.0, "max": 35000.0, "unit": "cells/cumm"},
    "rc": {"min": 1.0, "max": 10.0, "unit": "millions/cmm"},
}

# KDIGO eGFR Staging Thresholds (ml/min/1.73m2)
KDIGO_STAGES: Dict[str, Dict[str, Any]] = {
    "G1": {"name": "Normal or high", "min": 90.0, "max": float("inf")},
    "G2": {"name": "Mildly decreased", "min": 60.0, "max": 89.9},
    "G3a": {"name": "Mildly to moderately decreased", "min": 45.0, "max": 59.9},
    "G3b": {"name": "Moderately to severely decreased", "min": 30.0, "max": 44.9},
    "G4": {"name": "Severely decreased", "min": 15.0, "max": 29.9},
    "G5": {"name": "Kidney failure", "min": 0.0, "max": 14.9},
}

# Checkpoint Structure
CHECKPOINT_LATEST_DIR = "checkpoints/latest"
CHECKPOINT_BEST_DIR = "checkpoints/best"
CHECKPOINT_PERIODIC_DIR = "checkpoints/periodic"
CHECKPOINT_METADATA_DIR = "checkpoints/metadata"
