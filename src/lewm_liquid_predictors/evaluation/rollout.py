"""Closed-loop latent rollout evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lewm_liquid_predictors.data import TrajectoryBatch
from lewm_liquid_predictors.models.protocol import DynamicsPredictor
from lewm_liquid_predictors.models.rollout import teacher_forced_rollout


@dataclass(frozen=True)
class RolloutMetrics:
    """Teacher-forced and closed-loop errors for one padded trajectory batch."""

    one_step_normalized_rmse: Tensor
    rollout_normalized_rmse: dict[int, Tensor]
    divergence_time: Tensor
    divergence_rate: Tensor


@torch.no_grad()
def evaluate_rollouts(
    predictor: DynamicsPredictor,
    batch: TrajectoryBatch,
    horizons: tuple[int, ...],
    divergence_threshold: float,
) -> RolloutMetrics:
    """Evaluate teacher-forced and autoregressive predictions without padded tails."""
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive values")
    if max(horizons) > batch.actions.shape[1]:
        raise ValueError("requested horizon exceeds the batch trajectory length")
    if divergence_threshold <= 0:
        raise ValueError("divergence_threshold must be positive")

    targets = batch.latents[:, 1:]
    teacher_forced = teacher_forced_rollout(predictor, batch.latents, batch.actions)
    rollout, _ = predictor.rollout(batch.latents[:, 0], batch.actions)
    horizon_errors = {
        horizon: normalized_rmse(
            rollout[:, horizon - 1 : horizon],
            targets[:, horizon - 1 : horizon],
            batch.transition_mask[:, horizon - 1 : horizon],
        )
        for horizon in horizons
    }
    divergence_time = divergence_times(
        rollout,
        targets,
        batch.transition_mask,
        divergence_threshold,
    )
    divergence_rate = (divergence_time >= 0).to(rollout.dtype).mean()
    return RolloutMetrics(
        one_step_normalized_rmse=normalized_rmse(teacher_forced, targets, batch.transition_mask),
        rollout_normalized_rmse=horizon_errors,
        divergence_time=divergence_time,
        divergence_rate=divergence_rate,
    )


def normalized_rmse(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Compute latent RMSE normalized by masked target RMS."""
    _validate_evaluation_tensors(predictions, targets, mask)
    expanded_mask = mask.unsqueeze(-1)
    values = expanded_mask.sum() * predictions.shape[-1]
    if values.item() == 0:
        raise ValueError("mask must contain at least one valid transition")
    rmse = (((predictions - targets).square() * expanded_mask).sum() / values).sqrt()
    target_rms = ((targets.square() * expanded_mask).sum() / values).sqrt()
    return rmse / target_rms.clamp_min(torch.finfo(targets.dtype).eps)


def divergence_times(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    threshold: float,
) -> Tensor:
    """Return the first one-indexed divergent horizon, or ``-1`` if stable."""
    _validate_evaluation_tensors(predictions, targets, mask)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    per_step_rmse = (predictions - targets).square().mean(dim=-1).sqrt()
    target_rms = targets.square().mean(dim=-1).sqrt().clamp_min(torch.finfo(targets.dtype).eps)
    is_divergent = (per_step_rmse / target_rms > threshold) | ~torch.isfinite(predictions).all(
        dim=-1
    )
    is_divergent &= mask
    time_indices = torch.arange(1, mask.shape[1] + 1, device=mask.device)
    no_divergence = torch.full_like(time_indices, mask.shape[1] + 1)
    first_times = torch.where(is_divergent, time_indices, no_divergence).min(dim=1).values
    return torch.where(first_times <= mask.shape[1], first_times, -1)


def _validate_evaluation_tensors(predictions: Tensor, targets: Tensor, mask: Tensor) -> None:
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("predictions and targets must share shape (batch, time, latent_dim)")
    if mask.shape != predictions.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape (batch, time)")
