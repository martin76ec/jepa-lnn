"""Causal context Transformer dynamics predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from .protocol import PredictorState


@dataclass(frozen=True)
class TransformerState:
    """Bounded sequence of past latent/action features for one episode batch."""

    features: Tensor


class PredictorTransformer(nn.Module):
    """Predict next latents using a bounded causal Transformer context."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        context_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(latent_dim, action_dim, hidden_dim, num_heads, num_layers, context_length) <= 0:
            raise ValueError(
                "Transformer dimensions, layers, heads, and context_length must be positive"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.input_projection = nn.Linear(latent_dim + action_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def init_state(self, batch_size: int, device: str) -> PredictorState:
        """Create an empty episode-local causal context."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return TransformerState(
            features=torch.empty((batch_size, 0, self.latent_dim + self.action_dim), device=device)
        )

    def step(
        self,
        latent: Tensor,
        action: Tensor,
        state: PredictorState,
        dt: Tensor | None,
    ) -> tuple[Tensor, PredictorState]:
        """Predict one transition from the current and preceding context only."""
        del dt
        self._validate_inputs(latent, action)
        context = self._validate_or_initialize_state(state, latent)
        current_feature = torch.cat((latent, action), dim=-1).unsqueeze(1)
        features = torch.cat((context.features, current_feature), dim=1)
        features = features[:, -self.context_length :]
        hidden = self.input_projection(features)
        causal_mask = torch.triu(
            torch.full((hidden.shape[1], hidden.shape[1]), float("-inf"), device=hidden.device),
            diagonal=1,
        )
        prediction = self.output_projection(self.transformer(hidden, mask=causal_mask)[:, -1])
        return prediction, TransformerState(features=features)

    def rollout(
        self,
        initial_latent: Tensor,
        actions: Tensor,
        state: PredictorState = None,
        dt: Tensor | None = None,
    ) -> tuple[Tensor, PredictorState]:
        """Autoregressively predict one latent state for every action."""
        if initial_latent.ndim != 2 or initial_latent.shape[-1] != self.latent_dim:
            raise ValueError("initial_latent must have shape (batch, latent_dim)")
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError("actions must have shape (batch, time, action_dim)")
        if initial_latent.shape[0] != actions.shape[0] or actions.shape[1] == 0:
            raise ValueError("initial_latent and non-empty actions must share a batch size")
        current = initial_latent
        predictions: list[Tensor] = []
        current_state = (
            self.init_state(initial_latent.shape[0], str(initial_latent.device))
            if state is None
            else state
        )
        for action in actions.unbind(dim=1):
            current, current_state = self.step(current, action, current_state, dt)
            predictions.append(current)
        return torch.stack(predictions, dim=1), current_state

    def _validate_inputs(self, latent: Tensor, action: Tensor) -> None:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent must have shape (batch, latent_dim)")
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError("action must have shape (batch, action_dim)")
        if latent.shape[0] != action.shape[0]:
            raise ValueError("latent and action must share a batch size")

    def _validate_or_initialize_state(
        self, state: PredictorState, latent: Tensor
    ) -> TransformerState:
        if state is None:
            return cast(TransformerState, self.init_state(latent.shape[0], str(latent.device)))
        if not isinstance(state, TransformerState):
            raise TypeError("state must be TransformerState or None")
        expected_shape = (latent.shape[0], self.latent_dim + self.action_dim)
        if state.features.ndim != 3 or state.features.shape[0] != expected_shape[0]:
            raise ValueError("state batch size does not match latent")
        if state.features.shape[-1] != expected_shape[1] or state.features.device != latent.device:
            raise ValueError("state feature dimension or device does not match latent")
        return state
