"""Device allocation manager for PyTorch tensors and modules."""

import torch
from kidney_ai.hardware.hardware_detector import detect_hardware


def get_torch_device(preference: str = "auto") -> torch.device:
    """Resolves optimal torch.device given user preference ('auto', 'mps', 'cuda', 'cpu')."""
    if preference == "auto":
        profile = detect_hardware()
        return torch.device(profile.recommended_device)

    if preference == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device("cpu")
