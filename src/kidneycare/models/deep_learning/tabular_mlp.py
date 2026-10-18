"""Deep Tabular Multilayer Perceptron (MLP) architecture."""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    """Deep Tabular Multilayer Perceptron with Batch Normalization and Dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 2,
        dropout_rates: Optional[List[float]] = None,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout_rates = dropout_rates or [0.2, 0.1]
        self.use_residual = use_residual

        # Match dropouts length to hidden dims
        if len(self.dropout_rates) < len(self.hidden_dims):
            self.dropout_rates.extend([0.1] * (len(self.hidden_dims) - len(self.dropout_rates)))

        layers = []
        prev_dim = input_dim

        for idx, h_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation.lower() == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            if self.dropout_rates[idx] > 0.0:
                layers.append(nn.Dropout(self.dropout_rates[idx]))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier_head = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        features = self.feature_extractor(x)
        logits = self.classifier_head(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized probabilities via Softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
