"""Learning rate scheduler factory."""

from typing import Dict, Any, Optional
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR, ReduceLROnPlateau


def get_scheduler(
    optimizer: optim.Optimizer,
    name: str = "CosineAnnealingLR",
    epochs: int = 60,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build and configure learning rate scheduler.

    Args:
        optimizer: Configured optimizer.
        name: Scheduler name.
        epochs: Maximum training epochs.
        params: Additional parameters.

    Returns:
        Configured PyTorch scheduler.
    """
    sched_name = name.lower()
    extra_params = params or {}

    if sched_name in ["cosineannealinglr", "cosine"]:
        t_max = extra_params.get("T_max", epochs)
        eta_min = extra_params.get("eta_min", 1e-5)
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif sched_name == "steplr":
        step_size = extra_params.get("step_size", 20)
        gamma = extra_params.get("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "reducelronplateau":
        mode = extra_params.get("mode", "max")
        patience = extra_params.get("patience", 5)
        factor = extra_params.get("factor", 0.5)
        return ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)
    else:
        # Fallback to constant LR
        return StepLR(optimizer, step_size=epochs * 2, gamma=1.0)
