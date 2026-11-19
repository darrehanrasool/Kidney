"""Hardware acceleration and device management utilities."""

from kidney_ai.hardware.hardware_detector import HardwareProfile, detect_hardware
from kidney_ai.hardware.device_manager import get_torch_device

__all__ = ["HardwareProfile", "detect_hardware", "get_torch_device"]
