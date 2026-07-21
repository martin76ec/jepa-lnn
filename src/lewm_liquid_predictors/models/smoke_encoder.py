"""Lightweight stand-in encoders for pipeline smoke tests only.

These are NOT the LeWM baseline encoder. They exist to validate the full
train/evaluate/data/provenance pipeline without requiring the upstream
ViT and stable-pretraining dependencies. The real experiment must replace
these with the pinned upstream encoder, projector, and action encoder.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from .encoder import LeWMEncoder


class _SmokeVisionBackbone(nn.Module):
    """Flatten pixels and apply a single linear projection to latent_dim."""

    def __init__(self, channels: int, height: int, width: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.projection = nn.Linear(channels * height * width, latent_dim)

    def forward(self, pixels: Tensor, *, interpolate_pos_encoding: bool = True) -> object:
        del interpolate_pos_encoding
        flattened = pixels.flatten(start_dim=1)
        projected = cast(Tensor, self.projection(flattened))
        cls_token = projected.unsqueeze(1)
        return _SmokeOutput(last_hidden_state=cls_token)


class _SmokeOutput:
    def __init__(self, last_hidden_state: Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class SmokeActionEncoder(nn.Module):
    """Linear action encoder matching the upstream action embedding dimension."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, actions: Tensor) -> Tensor:
        return cast(Tensor, self.linear(actions.float()))


def build_smoke_encoder(
    latent_dim: int,
    channels: int = 3,
    height: int = 224,
    width: int = 224,
) -> LeWMEncoder:
    """Build a lightweight encoder wrapper for smoke tests only."""
    backbone = _SmokeVisionBackbone(channels, height, width, latent_dim)
    projector = nn.Identity()
    return LeWMEncoder(backbone, projector)
