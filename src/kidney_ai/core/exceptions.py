"""Domain exceptions for Kidney-AI clinical research pipeline."""


class KidneyAIException(Exception):
    """Base exception for all Kidney-AI platform errors."""
    pass


class ClinicalDataValidationError(KidneyAIException):
    """Raised when clinical data violates physiological constraints or schema."""
    pass


class ModelNotFoundError(KidneyAIException):
    """Raised when a requested model architecture is unknown or uninitialized."""
    pass


class HardwareAccelerationError(KidneyAIException):
    """Raised when hardware accelerator (CUDA/MPS) fails to initialize."""
    pass


class PipelineExecutionError(KidneyAIException):
    """Raised when the end-to-end execution pipeline encounters an unrecoverable state."""
    pass
