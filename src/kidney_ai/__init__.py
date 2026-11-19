"""Kidney-AI: Advanced Deep Learning & Machine Learning Framework for Nephrology Research."""

__version__ = "1.0.0"
__author__ = "Dar Rehan Rasool"
__affiliation__ = "Centre of Artificial Intelligence (CAI), IUST"

from kidney_ai.core.config import load_config
from kidney_ai.core.reproducibility import set_seed

__all__ = ["load_config", "set_seed", "__version__"]
