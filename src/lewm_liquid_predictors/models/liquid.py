"""CfC and LTC dynamics predictors backed by the maintained ncps package."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import torch
from torch import Tensor, nn

from .protocol import PredictorState


class _LiquidPredictor(nn.Module):
    """Shared one-step adapter for ncps recurrent liquid models."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, model_name: str) -> None:
        super().__init__()
        if latent_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim, action_dim, and hidden_dim must be positive")
        ncps_torch: Any = import_module("ncps.torch")
        model_class: Any = getattr(ncps_torch, model_name)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn = cast(
            nn.Module,
            model_class(
                latent_dim + action_dim,
                hidden_dim,
                return_sequences=True,
                batch_first=True,
            ),
        )
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def init_state(self, batch_size: int, device: str) -> PredictorState:
        """Create a zero recurrent state for a new batch of episodes."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros((batch_size, self.hidden_dim), device=device)

    def step(
        self,
        latent: Tensor,
        action: Tensor,
        state: PredictorState,
        dt: Tensor | None,
    ) -> tuple[Tensor, PredictorState]:
        """Advance the liquid recurrent state by one latent/action transition."""
        self._validate_inputs(latent, action)
        recurrent_state = self._validate_or_initialize_state(state, latent)
        inputs = torch.cat((latent, action), dim=-1).unsqueeze(1)
        timespans = self._timespans(dt, latent.shape[0], latent.device)
        output, next_state = self._run_rnn(inputs, recurrent_state, timespans)
        if not isinstance(output, Tensor) or not isinstance(next_state, Tensor):
            raise TypeError("ncps liquid models must return tensor outputs and states")
        return self.output_projection(output[:, -1]), next_state

    def _run_rnn(
        self, inputs: Tensor, state: Tensor, timespans: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if timespans is None:
            output, next_state = self.rnn(inputs, state)
            return self._validate_rnn_result(output, next_state)

        # ncps currently squeezes batched timespans internally. Processing
        # one sample at a time retains each sample's explicit integration dt.
        outputs: list[Tensor] = []
        next_states: list[Tensor] = []
        for index in range(inputs.shape[0]):
            output, next_state = self.rnn(
                inputs[index : index + 1],
                state[index : index + 1],
                timespans=timespans[index : index + 1],
            )
            validated_output, validated_state = self._validate_rnn_result(output, next_state)
            outputs.append(validated_output)
            next_states.append(validated_state)
        return torch.cat(outputs, dim=0), torch.cat(next_states, dim=0)

    def _validate_rnn_result(self, output: object, next_state: object) -> tuple[Tensor, Tensor]:
        if not isinstance(output, Tensor) or not isinstance(next_state, Tensor):
            raise TypeError("ncps liquid models must return tensor outputs and states")
        return output, next_state

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
        current_state = (
            self.init_state(initial_latent.shape[0], str(initial_latent.device))
            if state is None
            else state
        )
        current = initial_latent
        predictions: list[Tensor] = []
        for index, action in enumerate(actions.unbind(dim=1)):
            current, current_state = self.step(
                current, action, current_state, self._step_dt(dt, index)
            )
            predictions.append(current)
        return torch.stack(predictions, dim=1), current_state

    def _validate_inputs(self, latent: Tensor, action: Tensor) -> None:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent must have shape (batch, latent_dim)")
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError("action must have shape (batch, action_dim)")
        if latent.shape[0] != action.shape[0]:
            raise ValueError("latent and action must share a batch size")

    def _validate_or_initialize_state(self, state: PredictorState, latent: Tensor) -> Tensor:
        if state is None:
            return cast(Tensor, self.init_state(latent.shape[0], str(latent.device)))
        if not isinstance(state, Tensor):
            raise TypeError("state must be a tensor or None")
        if state.shape != (latent.shape[0], self.hidden_dim) or state.device != latent.device:
            raise ValueError("state shape or device does not match latent")
        return state

    def _timespans(self, dt: Tensor | None, batch_size: int, device: torch.device) -> Tensor | None:
        if dt is None:
            return None
        if dt.ndim == 0:
            return dt.to(device).expand(batch_size, 1)
        if dt.ndim == 1 and dt.shape[0] == batch_size:
            return dt.to(device).unsqueeze(1)
        if dt.ndim == 2 and dt.shape == (batch_size, 1):
            return dt.to(device)
        raise ValueError("dt must be scalar, shape (batch,), or shape (batch, 1)")

    def _step_dt(self, dt: Tensor | None, index: int) -> Tensor | None:
        if dt is None or dt.ndim < 2:
            return dt
        return dt[:, index]


class PredictorCfC(_LiquidPredictor):
    """Closed-form continuous-time predictor using ``ncps.torch.CfC``."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__(latent_dim, action_dim, hidden_dim, model_name="CfC")


class PredictorLTC(_LiquidPredictor):
    """Liquid time-constant predictor using ``ncps.torch.LTC``."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__(latent_dim, action_dim, hidden_dim, model_name="LTC")
