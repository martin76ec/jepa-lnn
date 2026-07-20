"""Tests for the stateless MLP predictor."""

import pytest
import torch

from lewm_liquid_predictors.models import PredictorMLP


def test_init_state_is_resettable_and_stateless() -> None:
    predictor = PredictorMLP(latent_dim=4, action_dim=2, hidden_dim=8)

    assert predictor.init_state(3, "cpu") is None
    assert predictor.init_state(3, "cpu") is None


def test_step_and_rollout_have_expected_shapes_and_gradients() -> None:
    predictor = PredictorMLP(latent_dim=4, action_dim=2, hidden_dim=8)
    latent = torch.randn(3, 4, requires_grad=True)
    actions = torch.randn(3, 5, 2, requires_grad=True)

    predictions, state = predictor.rollout(latent, actions)
    predictions.sum().backward()

    assert predictions.shape == (3, 5, 4)
    assert state is None
    assert latent.grad is not None
    assert actions.grad is not None
    assert all(parameter.grad is not None for parameter in predictor.parameters())


def test_repeated_step_matches_rollout() -> None:
    predictor = PredictorMLP(latent_dim=4, action_dim=2, hidden_dim=8)
    latent = torch.randn(2, 4)
    actions = torch.randn(2, 3, 2)
    rollout, rollout_state = predictor.rollout(latent, actions)

    state = predictor.init_state(latent.shape[0], "cpu")
    predictions = []
    current = latent
    for action in actions.unbind(dim=1):
        current, state = predictor.step(current, action, state, None)
        predictions.append(current)

    assert torch.allclose(rollout, torch.stack(predictions, dim=1))
    assert rollout_state is state


@pytest.mark.parametrize(
    ("latent_shape", "action_shape"),
    [((2, 3, 4), (2, 2)), ((2, 4), (2, 3, 2))],
)
def test_step_rejects_invalid_shapes(
    latent_shape: tuple[int, ...], action_shape: tuple[int, ...]
) -> None:
    predictor = PredictorMLP(latent_dim=4, action_dim=2, hidden_dim=8)

    with pytest.raises(ValueError):
        predictor.step(torch.randn(*latent_shape), torch.randn(*action_shape), None, None)
