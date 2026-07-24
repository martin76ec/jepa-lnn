"""Cross-attention image decoder for projected LeWM CLS latents."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DecoderArchitecture:
    """Immutable architecture specification for the post-hoc image decoder."""

    latent_dim: int = 192
    hidden_dim: int = 192
    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "channels": self.channels,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
        }
        if any(isinstance(value, bool) or value <= 0 for value in integer_fields.values()):
            raise ValueError("decoder architecture integer fields must be positive")
        if self.channels != 3:
            raise ValueError("decoder output must have three ImageNet RGB channels")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

    @property
    def num_patches(self) -> int:
        """Return the number of learned output patch queries."""
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side


class _CrossAttentionBlock(nn.Module):
    def __init__(self, architecture: DecoderArchitecture) -> None:
        super().__init__()
        hidden_dim = architecture.hidden_dim
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            architecture.num_heads,
            dropout=architecture.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(architecture.dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * architecture.mlp_ratio),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(hidden_dim * architecture.mlp_ratio, hidden_dim),
            nn.Dropout(architecture.dropout),
        )

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        """Apply one latent-conditioned cross-attention and residual MLP block."""
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        queries = queries + self.attention_dropout(attended)
        return queries + cast(Tensor, self.mlp(self.mlp_norm(queries)))


class CrossAttentionDecoder(nn.Module):
    """Decode one projected CLS latent per image into ImageNet-normalized pixels."""

    def __init__(self, architecture: DecoderArchitecture | None = None) -> None:
        super().__init__()
        self.architecture = architecture or DecoderArchitecture()
        architecture = self.architecture
        self.latent_projection = nn.Linear(architecture.latent_dim, architecture.hidden_dim)
        self.patch_queries = nn.Parameter(
            torch.empty(1, architecture.num_patches, architecture.hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [_CrossAttentionBlock(architecture) for _ in range(architecture.num_layers)]
        )
        self.final_norm = nn.LayerNorm(architecture.hidden_dim)
        self.patch_head = nn.Linear(
            architecture.hidden_dim,
            architecture.channels * architecture.patch_size * architecture.patch_size,
        )
        nn.init.normal_(self.patch_queries, std=0.02)

    def forward(self, latent: Tensor) -> Tensor:
        """Decode projected CLS latents of shape ``(batch, latent_dim)``."""
        architecture = self.architecture
        if latent.ndim != 2 or latent.shape[-1] != architecture.latent_dim:
            raise ValueError(f"latent must have shape (batch, {architecture.latent_dim})")
        context = self.latent_projection(latent).unsqueeze(1)
        queries = self.patch_queries.expand(latent.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        patches = self.patch_head(self.final_norm(queries))
        return self._unpatchify(patches)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        architecture = self.architecture
        batch_size = patches.shape[0]
        grid_size = architecture.image_size // architecture.patch_size
        pixels = patches.reshape(
            batch_size,
            grid_size,
            grid_size,
            architecture.channels,
            architecture.patch_size,
            architecture.patch_size,
        )
        return pixels.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size,
            architecture.channels,
            architecture.image_size,
            architecture.image_size,
        )
