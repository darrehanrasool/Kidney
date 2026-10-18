"""Execution device manager with automated fallback."""

import logging
from typing import Optional

from kidneycare.hardware.hardware_detector import HardwareDetector
from kidneycare.monitoring.logging_config import get_logger

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = get_logger("kidneycare.hardware")


class DeviceManager:
    """Manages PyTorch execution device assignment with CPU fallback safety."""

    def __init__(self, preferred: str = "auto", detector: Optional[HardwareDetector] = None):
        self.detector = detector or HardwareDetector()
        self.preferred = preferred.lower()
        self._device: Optional["torch.device"] = None

    def get_device(self) -> "torch.device":
        """Resolve and return appropriate PyTorch device.

        Returns:
            torch.device instance.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for device management.")

        if self._device is not None:
            return self._device

        profile = self.detector.detect()

        if self.preferred == "cuda":
            if profile["cuda_available"]:
                self._device = torch.device("cuda")
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self._device = torch.device("cpu")
        elif self.preferred == "mps":
            if profile["mps_available"]:
                self._device = torch.device("mps")
            else:
                logger.warning("MPS requested but not available. Falling back to CPU.")
                self._device = torch.device("cpu")
        elif self.preferred == "cpu":
            self._device = torch.device("cpu")
        else: # auto
            if profile["cuda_available"]:
                self._device = torch.device("cuda")
            elif profile["mps_available"]:
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

        logger.info(f"Initialized execution device: {self._device.type.upper()}")
        return self._device

    def to_device(self, tensor_or_model: Any) -> Any:
        """Safely move tensor or module to active device."""
        dev = self.get_device()
        try:
            return tensor_or_model.to(dev)
        except Exception as e:
            logger.warning(f"Failed to move object to {dev}: {e}. Falling back to CPU.")
            self._device = torch.device("cpu")
            return tensor_or_model.to(self._device)
