"""Optimizer factory for neural network training."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(
    model: nn.Module,
    name: str = "AdamW",
    lr: float = 0.001,
    weight_decay: float = 0.001,
    params: Optional[Dict[str, Any]] = None,
) -> optim.Optimizer:
    """Instantiate and configure PyTorch optimizer.

    Args:
        model: PyTorch module.
        name: Optimizer name ('AdamW', 'Adam', 'SGD').
        lr: Learning rate.
        weight_decay: L2 regularization penalty.
        params: Additional keyword arguments.

    Returns:
        Configured optimizer.
    """
    opt_name = name.lower()
    extra_params = params or {}

    if opt_name == "adamw":
        betas = extra_params.get("betas", (0.9, 0.999))
        eps = extra_params.get("eps", 1e-8)
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif opt_name == "adam":
        betas = extra_params.get("betas", (0.9, 0.999))
        eps = extra_params.get("eps", 1e-8)
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif opt_name == "sgd":
        momentum = extra_params.get("momentum", 0.9)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer '{name}'. Choose from ['AdamW', 'Adam', 'SGD']")
