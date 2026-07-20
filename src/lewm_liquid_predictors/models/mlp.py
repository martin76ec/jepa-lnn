"""Stateless MLP dynamics predictor."""

from __future__ import annotations

from torch import Tensor, cat, nn, stack

from .protocol import PredictorState


class PredictorMLP(nn.Module):
    """Predict the next latent state from the current latent state and action."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if latent_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim, action_dim, and hidden_dim must be positive")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def init_state(self, batch_size: int, device: str) -> PredictorState:
        """Return the stateless predictor state after validating the batch size."""
        del device
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return None

    def step(
        self,
        latent: Tensor,
        action: Tensor,
        state: PredictorState,
        dt: Tensor | None,
    ) -> tuple[Tensor, PredictorState]:
        """Predict one transition; ``state`` and ``dt`` are unused by this model."""
        del state, dt
        self._validate_inputs(latent, action)
        return self.network(cat((latent, action), dim=-1)), None

    def rollout(
        self,
        initial_latent: Tensor,
        actions: Tensor,
        state: PredictorState = None,
        dt: Tensor | None = None,
    ) -> tuple[Tensor, PredictorState]:
        """Autoregressively predict one latent state for every action."""
        if initial_latent.ndim != 2:
            raise ValueError("initial_latent must have shape (batch, latent_dim)")
        if actions.ndim != 3:
            raise ValueError("actions must have shape (batch, time, action_dim)")
        if initial_latent.shape[0] != actions.shape[0]:
            raise ValueError("initial_latent and actions must have the same batch size")
        if actions.shape[1] == 0:
            raise ValueError("actions must contain at least one timestep")

        latent = initial_latent
        predictions: list[Tensor] = []
        for action in actions.unbind(dim=1):
            latent, state = self.step(latent, action, state, dt)
            predictions.append(latent)
        return stack(predictions, dim=1), state

    def _validate_inputs(self, latent: Tensor, action: Tensor) -> None:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent must have shape (batch, latent_dim)")
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError("action must have shape (batch, action_dim)")
        if latent.shape[0] != action.shape[0]:
            raise ValueError("latent and action must have the same batch size")
