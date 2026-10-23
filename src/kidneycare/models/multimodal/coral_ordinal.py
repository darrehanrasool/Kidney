"""Consistent Rank Logits (CORAL) for ordinal clinical staging (KDIGO G1-G5, Albuminuria A1-A3).

Formulation:
    P(y > k | x) = sigma(f(x) + b_k) for k = 0, ..., K-2
    L_CORAL = - sum_{k=0}^{K-2} [ I(y > k) * log P_k + I(y <= k) * log(1 - P_k) ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CoralOrdinalHead(nn.Module):
    """Ordinal classification head using rank logits."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, 1, bias=False)
        # Learn K-1 independent threshold biases
        self.threshold_biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ordinal logits of shape (batch_size, num_classes - 1)."""
        score = self.linear(x) # (B, 1)
        logits = score + self.threshold_biases # (B, K-1)
        return logits

    def predict_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert ordinal rank logits into discrete class probabilities (B, K)."""
        sigmoids = torch.sigmoid(logits) # P(y > k)
        # Probabilities P(y = k)
        probs = []
        # P(y = 0) = 1 - P(y > 0)
        probs.append(1.0 - sigmoids[:, 0:1])
        # P(y = k) = P(y > k-1) - P(y > k)
        for k in range(1, self.num_classes - 1):
            probs.append(sigmoids[:, k - 1 : k] - sigmoids[:, k : k + 1])
        # P(y = K-1) = P(y > K-2)
        probs.append(sigmoids[:, -1:])

        probs = torch.cat(probs, dim=1)
        # Numerical clamp and re-normalize
        probs = torch.clamp(probs, min=0.0, max=1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def predict_labels(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict expected ordinal stage by counting binary cumulative indicators."""
        sigmoids = torch.sigmoid(logits)
        predicted_stages = torch.sum(sigmoids > 0.5, dim=1)
        return predicted_stages


class CoralLoss(nn.Module):
    """Loss function for Consistent Rank Logits (CORAL)."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary cross entropy across all K-1 binary rank thresholds.

        Args:
            logits: (B, K-1) tensor.
            targets: (B,) integer tensor with labels in [0, K-1].
        """
        batch_size = targets.size(0)
        num_thresholds = logits.size(1)

        # Build binary indicator matrix: target_indicators[i, k] = 1 if targets[i] > k else 0
        thresholds = torch.arange(num_thresholds, device=targets.device).unsqueeze(0).expand(batch_size, -1)
        binary_targets = (targets.unsqueeze(1) > thresholds).float()

        loss = F.binary_cross_entropy_with_logits(logits, binary_targets)
        return loss
