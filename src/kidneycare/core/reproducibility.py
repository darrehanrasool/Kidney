"""Reproducibility utilities for KidneyCare-AI.

Ensures deterministic execution, random seed synchronization across backends,
and environment telemetry collection.
"""

import os
import random
import platform
import sys
from typing import Dict, Any, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed across all libraries for deterministic execution.

    Args:
        seed: Integer seed value.
        deterministic: If True, configure PyTorch algorithms to be deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Seed MPS if supported
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass


def get_environment_info() -> Dict[str, Any]:
    """Gather complete environment runtime telemetry.

    Returns:
        Dictionary containing OS, Python, and package version details.
    """
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }

    if HAS_TORCH:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        info["mps_available"] = mps_available
    else:
        info["torch_version"] = "not_installed"
        info["cuda_available"] = False
        info["mps_available"] = False

    return info
