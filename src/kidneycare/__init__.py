"""KidneyCare-AI: Modular, Research-Grade Clinical Machine Learning Platform for Chronic Kidney Disease Detection.

Clinical Boundary Notice:
This package is strictly for research and educational purposes. It does not provide
clinical diagnoses or medical treatment advice.
"""

__version__ = "1.0.0"
__author__ = "KidneyCare-AI Research Team"

from kidneycare.core.config import Config
from kidneycare.core.constants import CLINICAL_BOUNDARY_DISCLAIMER

__all__ = ["Config", "CLINICAL_BOUNDARY_DISCLAIMER", "__version__"]
