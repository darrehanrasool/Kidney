"""Memory monitoring and garbage collection triggers."""

import gc
from typing import Dict, Any

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


class MemoryMonitor:
    """Monitors RAM and VRAM, providing cleanup methods."""

    def __init__(self, max_ram_fraction: float = 0.85):
        self.max_ram_fraction = max_ram_fraction

    def check_memory_headroom(self) -> bool:
        """Returns True if memory usage is within safe limits."""
        if psutil is not None:
            mem = psutil.virtual_memory()
            usage_fraction = mem.used / mem.total
            return usage_fraction < self.max_ram_fraction
        return True

    def clean_cache(self) -> None:
        """Trigger Python garbage collection and framework cache clearing."""
        gc.collect()
        if HAS_TORCH:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    def get_summary(self) -> Dict[str, Any]:
        """Summary of current memory usage."""
        if psutil is not None:
            mem = psutil.virtual_memory()
            return {
                "used_ram_percent": mem.percent,
                "available_ram_gb": round(mem.available / (1024**3), 2),
                "is_safe": self.check_memory_headroom(),
            }
        return {
            "used_ram_percent": 0.0,
            "available_ram_gb": 8.0,
            "is_safe": True,
        }
