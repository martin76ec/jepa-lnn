"""The shared interface for stateful latent dynamics predictors."""

from __future__ import annotations

from typing import Protocol, TypeAlias

from torch import Tensor

PredictorState: TypeAlias = object | None


class DynamicsPredictor(Protocol):
    """Predict next latent states while carrying model-specific recurrent state."""

    def init_state(self, batch_size: int, device: str) -> PredictorState:
        """Create an episode-local state for a batch."""

    def step(
        self,
        latent: Tensor,
        action: Tensor,
        state: PredictorState,
        dt: Tensor | None,
    ) -> tuple[Tensor, PredictorState]:
        """Predict one transition and return the updated state."""

    def rollout(
        self,
        initial_latent: Tensor,
        actions: Tensor,
        state: PredictorState = None,
        dt: Tensor | None = None,
    ) -> tuple[Tensor, PredictorState]:
        """Autoregressively predict a sequence of latent states."""
