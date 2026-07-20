"""Shared teacher-forced rollout helpers for dynamics predictors."""

from __future__ import annotations

from torch import Tensor, stack

from .protocol import DynamicsPredictor


def teacher_forced_rollout(
    predictor: DynamicsPredictor,
    latents: Tensor,
    actions: Tensor,
    dt: Tensor | None = None,
) -> Tensor:
    """Predict each next latent while feeding the observed previous latent.

    This preserves the upstream next-embedding training objective. Closed-loop
    rollout is reserved for evaluation rather than changing the optimization
    target for predictor variants.
    """
    if latents.ndim != 3:
        raise ValueError("latents must have shape (batch, time + 1, latent_dim)")
    if actions.ndim != 3:
        raise ValueError("actions must have shape (batch, time, action_dim)")
    if latents.shape[0] != actions.shape[0] or latents.shape[1] != actions.shape[1] + 1:
        raise ValueError("latents and actions must describe the same transitions")
    if actions.shape[1] == 0:
        raise ValueError("actions must contain at least one timestep")

    state = predictor.init_state(latents.shape[0], str(latents.device))
    predictions: list[Tensor] = []
    for index, action in enumerate(actions.unbind(dim=1)):
        prediction, state = predictor.step(latents[:, index], action, state, _step_dt(dt, index))
        predictions.append(prediction)
    return stack(predictions, dim=1)


def _step_dt(dt: Tensor | None, index: int) -> Tensor | None:
    if dt is None or dt.ndim < 2:
        return dt
    return dt[:, index]
