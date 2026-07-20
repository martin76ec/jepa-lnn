"""Tests for teacher-forced and closed-loop rollout metrics."""

import torch
from torch import Tensor, nn, stack

from lewm_liquid_predictors.data import Trajectory, collate_trajectories
from lewm_liquid_predictors.evaluation import evaluate_rollouts
from lewm_liquid_predictors.models import PredictorState


class _AddActionPredictor(nn.Module):
    def init_state(self, batch_size: int, device: str) -> PredictorState:
        return None

    def step(
        self, latent: Tensor, action: Tensor, state: PredictorState, dt: Tensor | None
    ) -> tuple[Tensor, PredictorState]:
        return latent + action, None

    def rollout(
        self,
        initial_latent: Tensor,
        actions: Tensor,
        state: PredictorState = None,
        dt: Tensor | None = None,
    ) -> tuple[Tensor, PredictorState]:
        predictions: list[Tensor] = []
        latent = initial_latent
        for action in actions.unbind(dim=1):
            latent, state = self.step(latent, action, state, dt)
            predictions.append(latent)
        return stack(predictions, dim=1), state


class _ZeroPredictor(_AddActionPredictor):
    def step(
        self, latent: Tensor, action: Tensor, state: PredictorState, dt: Tensor | None
    ) -> tuple[Tensor, PredictorState]:
        return torch.zeros_like(latent), None


def _batch() -> object:
    first = Trajectory(
        "first",
        latents=torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        actions=torch.ones(3, 1),
    )
    second = Trajectory(
        "second",
        latents=torch.tensor([[0.0], [1.0], [2.0]]),
        actions=torch.ones(2, 1),
    )
    return collate_trajectories([first, second])


def test_perfect_predictor_has_zero_teacher_forced_and_rollout_error() -> None:
    batch = _batch()
    assert hasattr(batch, "latents")

    metrics = evaluate_rollouts(_AddActionPredictor(), batch, (1, 2, 3), 0.1)

    assert metrics.one_step_normalized_rmse.item() == 0
    assert all(error.item() == 0 for error in metrics.rollout_normalized_rmse.values())
    assert torch.equal(metrics.divergence_time, torch.tensor([-1, -1]))
    assert metrics.divergence_rate.item() == 0


def test_divergence_uses_first_valid_closed_loop_horizon() -> None:
    batch = _batch()
    assert hasattr(batch, "latents")

    metrics = evaluate_rollouts(_ZeroPredictor(), batch, (1, 2), 0.5)

    assert torch.equal(metrics.divergence_time, torch.tensor([1, 1]))
    assert metrics.divergence_rate.item() == 1
