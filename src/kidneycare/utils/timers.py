"""Execution timer utilities."""

import time
from typing import Optional


class Timer:
    """Context manager and stopwatch for measuring execution latency."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - (self.start_time or self.end_time)

    def elapsed_seconds(self) -> float:
        """Return elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        if self.end_time is not None:
            return self.elapsed
        return time.perf_counter() - self.start_time
