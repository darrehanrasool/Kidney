"""Global seed and determinism management for rigorous reproducibility."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
