"""Tabular ResNet architecture with skip connections for deep biomarker representations."""

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Residual Neural Network optimized for high-dimensional clinical tabular data."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 3, output_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.relu(self.norm(h))
        return self.head(h)
