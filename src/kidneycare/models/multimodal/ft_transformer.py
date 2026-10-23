"""Feature Tokenizer Transformer (FT-Transformer) with learned missing-value tokens.

Mathematical Formulation:
    t_j = (1 - m_j) * (x_j * w_j + b_j) + m_j * u_j + e_j
where:
    - m_j in {0, 1} is the missingness indicator mask.
    - u_j in R^d is a learned vector representing unmeasured/missing lab values.
    - e_j in R^d is the feature identity position embedding.
"""

import torch
import torch.nn as nn
from typing import Optional


class FeatureTokenizer(nn.Module):
    """Tokenizes continuous and categorical clinical features with missingness tokens."""

    def __init__(self, num_features: int, d_token: int = 64):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token

        # Continuous feature projection weights & biases
        self.weight = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

        # Learned missing value tokens: u_j
        self.missing_tokens = nn.Parameter(torch.randn(num_features, d_token) * 0.02)

        # Feature identity / positional embeddings: e_j
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, d_token) * 0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Tokenize tabular input matrix.

        Args:
            x: (B, N) tensor of numerical values (missing values marked as 0.0 or NaN).
            mask: (B, N) binary tensor where 1 indicates missing value, 0 indicates observed.

        Returns:
            Tokens tensor of shape (B, N, d_token).
        """
        B, N = x.shape
        if mask is None:
            # Detect NaNs as missing
            mask = torch.isnan(x).float()
            x = torch.nan_to_num(x, nan=0.0)

        # Reshape for broadcasting: (B, N, 1) * (1, N, d_token) -> (B, N, d_token)
        x_expanded = x.unsqueeze(-1)
        w_expanded = self.weight.unsqueeze(0)
        b_expanded = self.bias.unsqueeze(0)

        # Observed token: x_j * w_j + b_j
        observed_tokens = x_expanded * w_expanded + b_expanded

        # Missing token: u_j
        missing_expanded = self.missing_tokens.unsqueeze(0).expand(B, -1, -1)
        mask_expanded = mask.unsqueeze(-1)

        # Blended token with missingness signal
        tokens = (1.0 - mask_expanded) * observed_tokens + mask_expanded * missing_expanded
        tokens = tokens + self.feature_embeddings.unsqueeze(0)
        return tokens


class FTTransformer(nn.Module):
    """Tabular Transformer operating on tokenized clinical lab values."""

    def __init__(
        self,
        num_features: int,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.out_dim = d_token

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract dense tabular representation vector.

        Returns:
            Representation vector (B, d_token).
        """
        tokens = self.tokenizer(x, mask) # (B, N, d)
        B = tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d)

        seq = torch.cat([cls_tokens, tokens], dim=1) # (B, N+1, d)
        out = self.transformer(seq)
        cls_out = self.norm(out[:, 0, :]) # Take CLS token representation
        return cls_out
