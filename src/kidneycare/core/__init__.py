"""Core package containing configuration, constants, exceptions, and reproducibility utilities."""

from kidneycare.core.config import Config, load_config
from kidneycare.core.constants import (
    CLINICAL_BOUNDARY_DISCLAIMER,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    CLASS_MAPPING,
)
from kidneycare.core.exceptions import (
    KidneyCareException,
    ConfigurationError,
    DatasetValidationError,
    DataLeakageError,
    ModelTrainingError,
)
from kidneycare.core.reproducibility import set_seed, get_environment_info

__all__ = [
    "Config",
    "load_config",
    "CLINICAL_BOUNDARY_DISCLAIMER",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET_COLUMN",
    "CLASS_MAPPING",
    "KidneyCareException",
    "ConfigurationError",
    "DatasetValidationError",
    "DataLeakageError",
    "ModelTrainingError",
    "set_seed",
    "get_environment_info",
]
