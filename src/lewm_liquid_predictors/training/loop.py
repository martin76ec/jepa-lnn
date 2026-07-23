"""Shared masked next-latent predictor training."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from tqdm import tqdm

from lewm_liquid_predictors.data import ObservationTrajectoryBatch, TrajectoryBatch
from lewm_liquid_predictors.models.protocol import DynamicsPredictor
from lewm_liquid_predictors.models.rollout import teacher_forced_rollout
from lewm_liquid_predictors.models.system import PredictorSystem


@dataclass(frozen=True)
class TrainEpochMetrics:
    """Aggregate metrics from one predictor training epoch."""

    mean_squared_error: float
    transitions: int


class PredictorTrainer:
    """Train an ``nn.Module`` that implements the shared predictor protocol."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        gradient_clip_val: float | None = None,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.gradient_clip_val = gradient_clip_val
        self.use_amp = use_amp and torch.cuda.is_available()
        self.predictor = (
            cast(DynamicsPredictor, model.predictor)
            if isinstance(model, PredictorSystem)
            else cast(DynamicsPredictor, model)
        )

    def train_epoch(
        self, batches: Iterable[TrajectoryBatch], total_batches: int | None = None
    ) -> TrainEpochMetrics:
        """Optimize masked teacher-forced next-latent mean squared error."""
        return self._train_latent_batches(batches, total_batches)

    def train_observation_epoch(
        self, batches: Iterable[ObservationTrajectoryBatch], total_batches: int | None = None
    ) -> TrainEpochMetrics:
        """Encode raw sequences and optimize their masked next-latent loss."""
        if not isinstance(self.model, PredictorSystem):
            raise TypeError("train_observation_epoch requires a PredictorSystem")
        return self._train_latent_batches(
            (self.model.encode_batch(batch) for batch in batches), total_batches
        )

    def _train_latent_batches(
        self, batches: Iterable[TrajectoryBatch], total_batches: int | None
    ) -> TrainEpochMetrics:
        self.model.train()
        total_squared_error = 0.0
        total_values = 0
        transitions = 0
        pbar = tqdm(batches, total=total_batches, desc="batches", file=sys.stderr, leave=False)
        for batch in pbar:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                predictions = teacher_forced_rollout(self.predictor, batch.latents, batch.actions)
                targets = batch.latents[:, 1:]
                loss = masked_transition_mse(predictions, targets, batch.transition_mask)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()

            values = int(batch.transition_mask.sum().item()) * targets.shape[-1]
            total_squared_error += loss.detach().item() * values
            total_values += values
            transitions += int(batch.transition_mask.sum().item())
            pbar.set_postfix(mse=f"{loss.item():.5f}")
        if total_values == 0:
            raise ValueError("batches contain no valid transitions")
        return TrainEpochMetrics(
            mean_squared_error=total_squared_error / total_values,
            transitions=transitions,
        )


def masked_transition_mse(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Compute MSE over valid transitions only."""
    _validate_transition_tensors(predictions, targets, mask)
    expanded_mask = mask.unsqueeze(-1)
    valid_values = expanded_mask.sum() * predictions.shape[-1]
    if valid_values.item() == 0:
        raise ValueError("mask must contain at least one valid transition")
    if not torch.isfinite(predictions[mask]).all() or not torch.isfinite(targets[mask]).all():
        raise ValueError("valid predictions and targets must be finite")
    squared_error = torch.where(
        expanded_mask,
        (predictions - targets).square(),
        torch.zeros_like(predictions),
    )
    return squared_error.sum() / valid_values


def _validate_transition_tensors(predictions: Tensor, targets: Tensor, mask: Tensor) -> None:
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("predictions and targets must share shape (batch, time, latent_dim)")
    if mask.shape != predictions.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape (batch, time)")
