"""Multi-Layer Perceptron for tabular biomarker classification."""

import torch
import torch.nn as nn
from typing import List


class TabularMLP(nn.Module):
    """Deep Multi-Layer Perceptron with Batch Normalization and Dropout."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, output_dim: int = 2, dropout: float = 0.25):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
