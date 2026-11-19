"""Learning rate schedulers."""

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR


def build_scheduler(optimizer: torch.optim.Optimizer, name: str = "CosineAnnealingWarmRestarts", t_0: int = 10, t_mult: int = 2, eta_min: float = 1e-6):
    if name.lower() == "cosineannealingwarmrestarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)
    elif name.lower() == "steplr":
        return StepLR(optimizer, step_size=15, gamma=0.5)
    return CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)
