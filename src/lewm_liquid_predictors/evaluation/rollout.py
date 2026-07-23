"""Closed-loop latent rollout evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import median

import torch
from torch import Tensor

from lewm_liquid_predictors.data import TrajectoryBatch
from lewm_liquid_predictors.models.protocol import DynamicsPredictor
from lewm_liquid_predictors.models.rollout import teacher_forced_rollout


@dataclass(frozen=True)
class RolloutMetrics:
    """Aggregate teacher-forced and closed-loop metrics over complete trajectories."""

    one_step_normalized_mse: Tensor
    rollout_normalized_mse: dict[int, Tensor]
    divergence_time: Tensor
    divergence_rate: Tensor
    median_first_divergence_time: Tensor | None
    rollout_non_finite_rate: dict[int, Tensor]

    @property
    def one_step_normalized_rmse(self) -> Tensor:
        """Return the legacy root of the aggregate normalized MSE."""
        return self.one_step_normalized_mse.sqrt()

    @property
    def rollout_normalized_rmse(self) -> dict[int, Tensor]:
        """Return legacy roots of aggregate endpoint normalized MSE values."""
        return {horizon: value.sqrt() for horizon, value in self.rollout_normalized_mse.items()}


class RolloutMetricAccumulator:
    """Stream rollout metrics one trajectory batch at a time.

    Non-finite endpoint predictions receive the minimum divergent normalized-MSE
    penalty (``divergence_threshold ** 2``) and are reported separately by rate.
    """

    def __init__(
        self,
        horizons: Sequence[int],
        divergence_threshold: float,
        count_non_finite_as_divergence: bool = True,
    ) -> None:
        configured_horizons = tuple(horizons)
        if not configured_horizons or any(
            not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
            for horizon in configured_horizons
        ):
            raise ValueError("horizons must contain positive values")
        if len(set(configured_horizons)) != len(configured_horizons):
            raise ValueError("horizons must not contain duplicates")
        if not isfinite(divergence_threshold) or divergence_threshold <= 0:
            raise ValueError("divergence_threshold must be finite and positive")

        self.horizons = configured_horizons
        self.divergence_threshold = divergence_threshold
        self.count_non_finite_as_divergence = count_non_finite_as_divergence
        self._one_step_sum = 0.0
        self._one_step_count = 0
        self._horizon_sums = dict.fromkeys(configured_horizons, 0.0)
        self._horizon_counts = dict.fromkeys(configured_horizons, 0)
        self._horizon_non_finite_counts = dict.fromkeys(configured_horizons, 0)
        self._divergence_times: list[Tensor] = []

    def update(
        self,
        teacher_forced_predictions: Tensor,
        rollout_predictions: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> None:
        """Accumulate one batch, counting every non-empty mask row as one trajectory."""
        _validate_evaluation_tensors(
            teacher_forced_predictions,
            targets,
            mask,
            prediction_name="teacher_forced_predictions",
        )
        _validate_evaluation_tensors(
            rollout_predictions,
            targets,
            mask,
            prediction_name="rollout_predictions",
            require_finite_predictions=False,
        )
        if not mask.any(dim=1).all():
            raise ValueError("each trajectory must contain at least one valid transition")

        one_step_values = _per_trajectory_normalized_mse(
            teacher_forced_predictions,
            targets,
            mask,
            prediction_name="teacher_forced_predictions",
        )
        horizon_values: dict[int, Tensor] = {}
        for horizon in self.horizons:
            if horizon > mask.shape[1]:
                continue
            endpoint_mask = mask[:, horizon - 1 : horizon]
            if not endpoint_mask.any():
                continue
            endpoint_predictions = rollout_predictions[:, horizon - 1 : horizon]
            finite_endpoint = torch.isfinite(endpoint_predictions).all(dim=-1)
            finite_mask = endpoint_mask & finite_endpoint
            non_finite_count = int((endpoint_mask & ~finite_endpoint).sum().item())
            self._horizon_non_finite_counts[horizon] += non_finite_count
            if finite_mask.any():
                horizon_values[horizon] = _per_trajectory_normalized_mse(
                    endpoint_predictions,
                    targets[:, horizon - 1 : horizon],
                    finite_mask,
                    prediction_name=f"rollout_predictions at horizon {horizon}",
                )
            self._horizon_sums[horizon] += non_finite_count * self.divergence_threshold**2
            self._horizon_counts[horizon] += int(endpoint_mask.sum().item())

        times = divergence_times(
            rollout_predictions,
            targets,
            mask,
            self.divergence_threshold,
            count_non_finite_as_divergence=self.count_non_finite_as_divergence,
        )
        self._one_step_sum += float(one_step_values.sum().item())
        self._one_step_count += one_step_values.numel()
        for horizon, values in horizon_values.items():
            self._horizon_sums[horizon] += float(values.sum().item())
        self._divergence_times.append(times.detach().cpu())

    def compute(self) -> RolloutMetrics:
        """Return aggregates; median time covers divergent trajectories only."""
        if self._one_step_count == 0 or not self._divergence_times:
            raise ValueError("cannot compute rollout metrics before an update")

        divergence_time = torch.cat(self._divergence_times)
        divergent_times = divergence_time[divergence_time >= 0]
        divergence_rate = torch.tensor(
            divergent_times.numel() / divergence_time.numel(), dtype=torch.float64
        )
        median_time = (
            torch.tensor(median(divergent_times.tolist()), dtype=torch.float64)
            if divergent_times.numel() > 0
            else None
        )
        horizon_metrics = {
            horizon: torch.tensor(
                self._horizon_sums[horizon] / self._horizon_counts[horizon],
                dtype=torch.float64,
            )
            for horizon in self.horizons
            if self._horizon_counts[horizon] > 0
        }
        return RolloutMetrics(
            one_step_normalized_mse=torch.tensor(
                self._one_step_sum / self._one_step_count, dtype=torch.float64
            ),
            rollout_normalized_mse=horizon_metrics,
            divergence_time=divergence_time,
            divergence_rate=divergence_rate,
            median_first_divergence_time=median_time,
            rollout_non_finite_rate={
                horizon: torch.tensor(
                    self._horizon_non_finite_counts[horizon] / self._horizon_counts[horizon],
                    dtype=torch.float64,
                )
                for horizon in self.horizons
                if self._horizon_counts[horizon] > 0
            },
        )


@torch.no_grad()
def evaluate_rollouts(
    predictor: DynamicsPredictor,
    batch: TrajectoryBatch,
    horizons: tuple[int, ...],
    divergence_threshold: float,
    count_non_finite_as_divergence: bool = True,
) -> RolloutMetrics:
    """Evaluate one padded trajectory batch with the streaming metric primitives."""
    targets = batch.latents[:, 1:]
    teacher_forced = teacher_forced_rollout(predictor, batch.latents, batch.actions)
    state = predictor.init_state(batch.latents.shape[0], str(batch.latents.device))
    rollout, _ = predictor.rollout(batch.latents[:, 0], batch.actions, state)
    accumulator = RolloutMetricAccumulator(
        horizons,
        divergence_threshold,
        count_non_finite_as_divergence=count_non_finite_as_divergence,
    )
    accumulator.update(teacher_forced, rollout, targets, batch.transition_mask)
    return accumulator.compute()


def normalized_mse(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Return the arithmetic mean of per-trajectory normalized latent MSE."""
    _validate_evaluation_tensors(predictions, targets, mask)
    values = _per_trajectory_normalized_mse(predictions, targets, mask)
    return values.mean()


def normalized_rmse(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Return the legacy root of aggregate per-trajectory normalized latent MSE."""
    return normalized_mse(predictions, targets, mask).sqrt()


def divergence_times(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    threshold: float,
    count_non_finite_as_divergence: bool = True,
) -> Tensor:
    """Return each trajectory's first one-indexed divergent horizon, or ``-1``."""
    _validate_evaluation_tensors(
        predictions,
        targets,
        mask,
        prediction_name="predictions",
        require_finite_predictions=False,
    )
    if not isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")

    finite_predictions = torch.isfinite(predictions).all(dim=-1)
    valid_finite = mask & finite_predictions
    expanded_mask = mask.unsqueeze(-1)
    safe_targets = torch.where(expanded_mask, targets, torch.zeros_like(targets)).double()
    safe_predictions = torch.where(
        valid_finite.unsqueeze(-1), predictions, safe_targets.to(predictions.dtype)
    ).double()
    squared_error = (safe_predictions - safe_targets).square().sum(dim=-1)
    target_energy = safe_targets.square().sum(dim=-1)
    normalized_error = (
        squared_error / target_energy.clamp_min(torch.finfo(torch.float64).eps)
    ).sqrt()
    is_divergent = valid_finite & (normalized_error > threshold)
    if count_non_finite_as_divergence:
        is_divergent |= mask & ~finite_predictions

    time_indices = torch.arange(1, mask.shape[1] + 1, device=mask.device)
    no_divergence = torch.full_like(time_indices, mask.shape[1] + 1)
    first_times = torch.where(is_divergent, time_indices, no_divergence).min(dim=1).values
    return torch.where(first_times <= mask.shape[1], first_times, -1)


def _per_trajectory_normalized_mse(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    prediction_name: str = "predictions",
) -> Tensor:
    valid_trajectories = mask.any(dim=1)
    if not valid_trajectories.any():
        raise ValueError("mask must contain at least one valid transition")
    _require_finite_at_mask(predictions, mask, prediction_name)
    _require_finite_at_mask(targets, mask, "targets")

    expanded_mask = mask.unsqueeze(-1)
    safe_predictions = torch.where(
        expanded_mask, predictions, torch.zeros_like(predictions)
    ).double()
    safe_targets = torch.where(expanded_mask, targets, torch.zeros_like(targets)).double()
    error_energy = (safe_predictions - safe_targets).square().sum(dim=(1, 2))
    target_energy = safe_targets.square().sum(dim=(1, 2))
    selected_target_energy = target_energy[valid_trajectories]
    if (selected_target_energy <= 0).any():
        indices = (valid_trajectories & (target_energy <= 0)).nonzero(as_tuple=False).flatten()
        raise ValueError(
            "normalized latent MSE requires positive target squared sum for trajectories "
            f"{indices.tolist()}"
        )
    values = error_energy[valid_trajectories] / selected_target_energy
    if not torch.isfinite(values).all():
        raise ValueError("normalized latent MSE produced non-finite valid metric values")
    return values


def _validate_evaluation_tensors(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    prediction_name: str = "predictions",
    require_finite_predictions: bool = True,
) -> None:
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError(
            f"{prediction_name} and targets must share shape (batch, time, latent_dim)"
        )
    if predictions.shape[0] == 0 or predictions.shape[1] == 0 or predictions.shape[2] == 0:
        raise ValueError(
            "evaluation tensors must have non-empty batch, time, and latent dimensions"
        )
    if mask.shape != predictions.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape (batch, time)")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise ValueError("predictions and targets must be floating-point tensors")
    if predictions.device != targets.device or mask.device != targets.device:
        raise ValueError("predictions, targets, and mask must be on the same device")
    _require_finite_at_mask(targets, mask, "targets")
    if require_finite_predictions:
        _require_finite_at_mask(predictions, mask, prediction_name)


def _require_finite_at_mask(values: Tensor, mask: Tensor, name: str) -> None:
    if not torch.isfinite(values[mask]).all():
        raise ValueError(f"{name} contains non-finite values at valid metric positions")
