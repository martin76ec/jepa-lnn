"""Shared encoder, action encoder, and dynamics predictor composition."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from lewm_liquid_predictors.data import ObservationTrajectoryBatch, TrajectoryBatch

from .encoder import LeWMEncoder


class PredictorSystem(nn.Module):
    """Compose shared LeWM encoders with one interchangeable latent predictor."""

    def __init__(
        self,
        encoder: LeWMEncoder,
        action_encoder: nn.Module,
        predictor: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self._shared_modules_frozen = False

    def freeze_shared_modules(self) -> None:
        """Freeze the encoder and action encoder as a fixed latent representation."""
        self._shared_modules_frozen = True
        self.encoder.requires_grad_(False).eval()
        self.action_encoder.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> PredictorSystem:
        """Set training mode while preserving frozen shared modules in evaluation mode."""
        super().train(mode)
        if self._shared_modules_frozen:
            self.encoder.eval()
            self.action_encoder.eval()
        return self

    def encode_batch(self, batch: ObservationTrajectoryBatch) -> TrajectoryBatch:
        """Encode valid pixels/actions without letting padding affect shared modules."""
        state_mask = torch.cat(
            (
                torch.ones(
                    (batch.transition_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=batch.transition_mask.device,
                ),
                batch.transition_mask,
            ),
            dim=1,
        )
        with torch.set_grad_enabled(not self._shared_modules_frozen):
            latents = self._scatter_encoded_states(batch.observations, state_mask)
            action_embeddings = self._scatter_encoded_actions(batch.actions, batch.transition_mask)
        return TrajectoryBatch(
            episode_ids=batch.episode_ids,
            latents=latents,
            actions=action_embeddings,
            transition_mask=batch.transition_mask,
        )

    def _scatter_encoded_states(self, observations: Tensor, mask: Tensor) -> Tensor:
        encoded = self.encoder(observations[mask].unsqueeze(0)).squeeze(0)
        latents = cast(Tensor, encoded.new_zeros((*mask.shape, encoded.shape[-1])))
        latents[mask] = encoded
        return latents

    def _scatter_encoded_actions(self, actions: Tensor, mask: Tensor) -> Tensor:
        encoded = cast(Tensor, self.action_encoder(actions[mask].unsqueeze(0)))
        if not isinstance(encoded, Tensor) or encoded.ndim != 3 or encoded.shape[0] != 1:
            raise ValueError("action_encoder must return shape (batch, time, action_embedding_dim)")
        action_embeddings = encoded.new_zeros((*mask.shape, encoded.shape[-1]))
        action_embeddings[mask] = encoded.squeeze(0)
        return action_embeddings
