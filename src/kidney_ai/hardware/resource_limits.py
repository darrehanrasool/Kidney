"""Enforcement of CPU threads and memory limits."""

import os
import torch


def configure_resource_limits(num_threads: int = 4) -> None:
    """Configures thread concurrency for PyTorch and underlying BLAS libraries."""
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
