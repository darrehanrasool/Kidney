"""Custom exceptions for KidneyCare-AI platform."""


class KidneyCareException(Exception):
    """Base exception for all KidneyCare-AI errors."""
    pass


class ConfigurationError(KidneyCareException):
    """Raised when configuration is invalid, incomplete, or missing."""
    pass


class DatasetValidationError(KidneyCareException):
    """Raised when dataset integrity, checksum, or schema validation fails."""
    pass


class DataLeakageError(KidneyCareException):
    """Raised when target leakage or train/test overlap is detected."""
    pass


class HardwareDetectionError(KidneyCareException):
    """Raised when hardware detection fails or device cannot be initialized."""
    pass


class ModelTrainingError(KidneyCareException):
    """Raised when model fitting, forward pass, or gradient step fails."""
    pass


class CheckpointCorruptError(KidneyCareException):
    """Raised when checkpoint file is corrupted, truncated, or incompatible."""
    pass


class MetricCalculationError(KidneyCareException):
    """Raised when an evaluation metric cannot be computed."""
    pass


class ClinicalBoundaryViolationError(KidneyCareException):
    """Raised when an unapproved clinical diagnostic claim is attempted."""
    pass
