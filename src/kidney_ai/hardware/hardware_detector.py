"""Automated detection of compute hardware (Apple Silicon MPS, NVIDIA CUDA, CPU)."""

import os
import platform
from dataclasses import dataclass

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    cpu_architecture: str
    cpu_count_physical: int
    cpu_count_logical: int
    total_ram_gb: float
    available_ram_gb: float
    has_cuda: bool
    cuda_device_name: str
    has_mps: bool
    recommended_device: str


def detect_hardware() -> HardwareProfile:
    """Audits system architecture, memory, and GPU acceleration capabilities."""
    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        total_ram_gb = round(vm.total / (1024**3), 2)
        available_ram_gb = round(vm.available / (1024**3), 2)
        physical_cpus = psutil.cpu_count(logical=False) or 1
        logical_cpus = psutil.cpu_count(logical=True) or 1
    else:
        total_ram_gb = 16.0
        available_ram_gb = 8.0
        logical_cpus = os.cpu_count() or 4
        physical_cpus = logical_cpus // 2 or 1

    has_cuda = HAS_TORCH and torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if has_cuda else "None"
    
    # Check Apple Silicon MPS
    has_mps = HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if has_cuda:
        recommended = "cuda"
    elif has_mps:
        recommended = "mps"
    else:
        recommended = "cpu"

    return HardwareProfile(
        os_name=platform.system(),
        os_version=platform.release(),
        cpu_architecture=platform.machine(),
        cpu_count_physical=physical_cpus,
        cpu_count_logical=logical_cpus,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        has_cuda=has_cuda,
        cuda_device_name=cuda_device_name,
        has_mps=has_mps,
        recommended_device=recommended,
    )
