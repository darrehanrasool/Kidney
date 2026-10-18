"""Hardware and accelerator capability detection."""

import platform
import sys
from typing import Dict, Any, Optional

import os
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class HardwareDetector:
    """Detects available computing hardware, CPU, CUDA, and Apple Silicon MPS."""

    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None

    def detect(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Detect runtime hardware environment and capabilities.

        Args:
            force_refresh: If True, bypass cached results.

        Returns:
            Dictionary with complete hardware profile.
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        os_name = platform.system()
        if psutil is not None:
            cpu_physical = psutil.cpu_count(logical=False) or 1
            cpu_logical = psutil.cpu_count(logical=True) or 1
            ram_bytes = psutil.virtual_memory().total
        else:
            cpu_physical = os.cpu_count() or 1
            cpu_logical = os.cpu_count() or 1
            try:
                ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except (AttributeError, ValueError):
                ram_bytes = 8 * (1024**3)

        ram_gb = round(ram_bytes / (1024**3), 2)

        cuda_available = False
        cuda_devices = []
        mps_available = False
        selected_backend = "cpu"
        warnings = []

        if HAS_TORCH:
            # Check CUDA
            if torch.cuda.is_available():
                try:
                    # Run small capability check
                    test_tensor = torch.zeros((2, 2), device="cuda")
                    _ = test_tensor + 1.0
                    cuda_available = True
                    selected_backend = "cuda"
                    for i in range(torch.cuda.device_count()):
                        prop = torch.cuda.get_device_properties(i)
                        cuda_devices.append({
                            "index": i,
                            "name": prop.name,
                            "total_memory_mb": round(prop.total_memory / (1024**2), 2),
                            "major": prop.major,
                            "minor": prop.minor,
                        })
                except Exception as e:
                    warnings.append(f"CUDA reported available but capability check failed: {e}")
                    cuda_available = False

            # Check MPS on macOS
            if not cuda_available and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    test_tensor = torch.zeros((2, 2), device="mps")
                    _ = test_tensor + 1.0
                    mps_available = True
                    selected_backend = "mps"
                except Exception as e:
                    warnings.append(f"Apple Silicon MPS available but capability check failed: {e}")
                    mps_available = False
        else:
            warnings.append("PyTorch is not installed; running in CPU-only classical ML mode.")

        profile = {
            "os": os_name,
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
            "framework": "PyTorch" if HAS_TORCH else "None",
            "framework_version": torch.__version__ if HAS_TORCH else "N/A",
            "cpu_physical_cores": cpu_physical,
            "cpu_logical_cores": cpu_logical,
            "system_ram_gb": ram_gb,
            "cuda_available": cuda_available,
            "cuda_devices": cuda_devices,
            "mps_available": mps_available,
            "selected_backend": selected_backend,
            "warnings": warnings,
        }

        self._cache = profile
        return profile

    def generate_startup_report(self) -> str:
        """Format detection results into a clear readable startup report."""
        p = self.detect()
        lines = [
            "=" * 60,
            "  KidneyCare-AI Hardware & Runtime Telemetry",
            "=" * 60,
            f"  Operating System  : {p['os']} ({p['machine']}, {p['os_release']})",
            f"  Python Version    : {p['python_version']}",
            f"  Framework         : {p['framework']} {p['framework_version']}",
            f"  CPU Cores         : {p['cpu_physical_cores']} physical, {p['cpu_logical_cores']} logical",
            f"  System RAM        : {p['system_ram_gb']} GB",
            f"  NVIDIA CUDA       : {'Available' if p['cuda_available'] else 'Not available'}",
        ]
        if p["cuda_devices"]:
            for d in p["cuda_devices"]:
                lines.append(f"    - GPU {d['index']}: {d['name']} ({d['total_memory_mb']} MB)")
        lines.append(f"  Apple Silicon MPS : {'Available' if p['mps_available'] else 'Not available'}")
        lines.append(f"  Execution Backend : {p['selected_backend'].upper()}")
        if p["warnings"]:
            lines.append("  Warnings:")
            for w in p["warnings"]:
                lines.append(f"    * {w}")
        lines.append("=" * 60)
        return "\n".join(lines)
