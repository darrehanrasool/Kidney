"""Loss functions and optimizer builders."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict


class FocalLoss(nn.Module):
    """Focal Loss down-weighting well-classified clinical examples."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.65, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_optimizer(model: torch.nn.Module, name: str = "AdamW", lr: float = 0.001, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    if name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr)
