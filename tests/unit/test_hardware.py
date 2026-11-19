from kidney_ai.hardware.hardware_detector import detect_hardware
from kidney_ai.hardware.device_manager import get_torch_device
import torch


def test_hardware_detector():
    profile = detect_hardware()
    assert profile.os_name != ""
    assert profile.cpu_count_logical >= 1
    assert profile.recommended_device in ["mps", "cuda", "cpu"]


def test_get_torch_device():
    dev = get_torch_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"
