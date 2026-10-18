"""Resource limit enforcement and OOM adaptation."""

import os
from typing import Optional
try:
    import psutil
except ImportError:
    psutil = None

from kidneycare.monitoring.logging_config import get_logger

logger = get_logger("kidneycare.hardware")


class ResourceLimits:
    """Enforces conservative bounds on worker processes, batch size, and timeouts."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_batch_size: int = 128,
        min_batch_size: int = 4,
    ):
        system_cores = (psutil.cpu_count(logical=True) if psutil is not None else os.cpu_count()) or 1
        # Conservative default: max cores / 2, minimum 1
        default_workers = max(1, system_cores // 2)
        self.max_workers = max_workers if max_workers is not None else min(default_workers, 4)
        self.current_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

    def adapt_batch_size_on_oom(self) -> int:
        """Halve batch size upon out-of-memory condition.

        Returns:
            Reduced batch size, or raises RuntimeError if minimum reached.
        """
        new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        if new_batch_size == self.current_batch_size:
            raise RuntimeError(
                f"Cannot reduce batch size further; already at minimum ({self.min_batch_size})"
            )
        logger.warning(
            f"OOM detected: Reducing batch size from {self.current_batch_size} to {new_batch_size}"
        )
        self.current_batch_size = new_batch_size
        return new_batch_size

    def get_safe_num_workers(self) -> int:
        """Return safe number of DataLoader workers."""
        # On macOS or Windows, 0 is safest to avoid fork issues
        import platform
        if platform.system() in ["Darwin", "Windows"]:
            return 0
        return self.max_workers
