"""Perceiver Cross-Attention Fusion with Modality Dropout.

Formulation:
    z = Perceiver({delta_k * h_k}_{k=1}^K), delta_k ~ Bern(1 - p)
Randomly masks whole modalities during training so the network seamlessly handles
tabular-only, ultrasound/CT-only, or paired multimodal patient encounters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class PerceiverCrossAttentionBlock(nn.Module):
    """Cross-attention block projecting variable input tokens into fixed latent queries."""

    def __init__(self, d_latent: int = 128, d_input: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_latent, kdim=d_input, vdim=d_input, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm_latent = nn.LayerNorm(d_latent)
        self.norm_input = nn.LayerNorm(d_input)

        # Self-attention over latent tokens
        self.self_attn = nn.MultiheadAttention(embed_dim=d_latent, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm_self = nn.LayerNorm(d_latent)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 2, d_latent),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_latent)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-attend latents to inputs, then self-attend."""
        # 1. Cross attention
        q = self.norm_latent(latents)
        k = self.norm_input(inputs)
        v = k
        cross_out, _ = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask)
        latents = latents + cross_out

        # 2. Self attention
        q_self = self.norm_self(latents)
        self_out, _ = self.self_attn(q_self, q_self, q_self)
        latents = latents + self_out

        # 3. Feedforward
        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


class PerceiverFusionEngine(nn.Module):
    """Fuses heterogeneous modalities (Tabular, Ultrasound, CT) into unified patient representation."""

    def __init__(
        self,
        d_latent: int = 128,
        num_latents: int = 16,
        modality_dropout: float = 0.3,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.num_latents = num_latents
        self.modality_dropout = modality_dropout

        # Learnable latent query tokens: Z in R^{num_latents x d_latent}
        self.latents = nn.Parameter(torch.randn(num_latents, d_latent) * 0.02)

        # Perceiver attention blocks
        self.blocks = nn.ModuleList([
            PerceiverCrossAttentionBlock(d_latent=d_latent, d_input=d_latent, n_heads=4)
            for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_latent)
        self.out_dim = d_latent

    def forward(
        self,
        tabular_tokens: Optional[torch.Tensor] = None, # (B, 1, d)
        image_tokens: Optional[torch.Tensor] = None,   # (B, 1, d)
        force_modality: Optional[str] = None,          # 'tabular', 'image', or 'both'
    ) -> torch.Tensor:
        """Execute multimodal fusion with stochastic modality dropout.

        Args:
            tabular_tokens: Projected tabular embeddings (B, 1, d).
            image_tokens: Projected vision embeddings (B, 1, d).
            force_modality: Optional evaluation constraint.

        Returns:
            Fused patient latent vector (B, d_latent).
        """
        B = tabular_tokens.size(0) if tabular_tokens is not None else image_tokens.size(0)
        device = tabular_tokens.device if tabular_tokens is not None else image_tokens.device

        # Modality dropout during training
        keep_tab = True
        keep_img = True

        if self.training and force_modality is None:
            if tabular_tokens is not None and image_tokens is not None:
                p = torch.rand(1).item()
                if p < self.modality_dropout / 2:
                    keep_img = False  # Tabular only
                elif p < self.modality_dropout:
                    keep_tab = False  # Image only
        elif force_modality == "tabular":
            keep_img = False
        elif force_modality == "image":
            keep_tab = False

        input_list = []
        if tabular_tokens is not None and keep_tab:
            input_list.append(tabular_tokens)
        if image_tokens is not None and keep_img:
            input_list.append(image_tokens)

        # If both dropped by extreme accident, restore at least one
        if not input_list:
            if tabular_tokens is not None:
                input_list.append(tabular_tokens)
            else:
                input_list.append(image_tokens)

        # Concatenate tokens along sequence length
        inputs = torch.cat(input_list, dim=1) # (B, seq_len, d_latent)

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1) # (B, num_latents, d_latent)

        for block in self.blocks:
            latents = block(latents, inputs)

        latents = self.final_norm(latents)
        # Pool latents to single vector: mean over latent queries
        pooled = torch.mean(latents, dim=1)
        return pooled
