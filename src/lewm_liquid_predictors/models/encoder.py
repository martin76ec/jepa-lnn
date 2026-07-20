"""Shared LeWM pixel-to-latent encoder wrapper."""

from __future__ import annotations

from typing import Protocol, cast

from torch import Tensor, nn


class VisionEncoderOutput(Protocol):
    """The Hugging Face ViT output surface used by LeWM."""

    last_hidden_state: Tensor


class VisionEncoder(Protocol):
    """Vision encoder interface used by the upstream LeWM model."""

    def __call__(self, pixels: Tensor, *, interpolate_pos_encoding: bool) -> VisionEncoderOutput:
        """Encode flattened images into token embeddings."""


class LeWMEncoder(nn.Module):
    """Apply LeWM's existing ViT and projector to batched image sequences."""

    def __init__(self, encoder: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, observations: Tensor) -> Tensor:
        """Return one projected latent vector for every input observation.

        This matches the pixel branch of upstream ``JEPA.encode`` exactly:
        flatten batch/time, select the ViT CLS token, project, and restore the
        original batch/time axes.
        """
        if observations.ndim != 5:
            raise ValueError("observations must have shape (batch, time, channels, height, width)")
        batch_size, timesteps = observations.shape[:2]
        pixels = observations.float().flatten(end_dim=1)
        encoder = cast(VisionEncoder, self.encoder)
        output = encoder(pixels, interpolate_pos_encoding=True)
        projected = self.projector(output.last_hidden_state[:, 0])
        if not isinstance(projected, Tensor) or projected.ndim != 2:
            raise ValueError("projector must return a rank-2 latent tensor")
        return projected.reshape(batch_size, timesteps, -1)
