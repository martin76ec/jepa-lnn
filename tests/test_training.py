"""Tests for shared masked predictor training."""

import pytest
import torch

from lewm_liquid_predictors.data import Trajectory, collate_trajectories
from lewm_liquid_predictors.models import PredictorMLP, teacher_forced_rollout
from lewm_liquid_predictors.training import PredictorTrainer, masked_transition_mse


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


def test_masked_transition_mse_ignores_padded_values() -> None:
    predictions = torch.tensor([[[1.0], [float("nan")]]])
    targets = torch.tensor([[[0.0], [float("nan")]]])
    mask = torch.tensor([[True, False]])

    assert masked_transition_mse(predictions, targets, mask).item() == pytest.approx(1.0)


def test_trainer_updates_model_with_teacher_forced_loss() -> None:
    batch = _batch()
    assert hasattr(batch, "latents")
    predictor = PredictorMLP(latent_dim=1, action_dim=1, hidden_dim=4)
    optimizer = torch.optim.SGD(predictor.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in predictor.parameters()]

    metrics = PredictorTrainer(predictor, optimizer).train_epoch([batch])

    after = list(predictor.parameters())
    assert metrics.transitions == 5
    assert metrics.mean_squared_error >= 0
    assert any(
        not torch.equal(previous, current) for previous, current in zip(before, after, strict=True)
    )


def test_teacher_forced_rollout_has_one_prediction_per_transition() -> None:
    batch = _batch()
    assert hasattr(batch, "latents") and hasattr(batch, "actions")
    predictor = PredictorMLP(latent_dim=1, action_dim=1, hidden_dim=4)

    predictions = teacher_forced_rollout(predictor, batch.latents, batch.actions)

    assert predictions.shape == (2, 3, 1)
