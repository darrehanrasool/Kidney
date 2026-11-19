"""Memory usage tracker and automated cache reclamation."""

import gc

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


class MemoryMonitor:
    """Monitors physical RAM and PyTorch accelerator memory."""

    @staticmethod
    def get_ram_usage_mb() -> float:
        if HAS_PSUTIL:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        return 0.0

    @staticmethod
    def reclaim_memory() -> None:
        gc.collect()
        if HAS_TORCH:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
