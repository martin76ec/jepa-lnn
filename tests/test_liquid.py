"""Tests for CfC and LTC predictor adapters."""

from typing import Any

import pytest
import torch

from lewm_liquid_predictors.models import PredictorCfC, PredictorLTC


@pytest.mark.parametrize("predictor_class", [PredictorCfC, PredictorLTC])
def test_liquid_predictors_reset_state_and_support_gradients(predictor_class: Any) -> None:
    predictor = predictor_class(latent_dim=4, action_dim=2, hidden_dim=6)
    state = predictor.init_state(3, "cpu")
    assert isinstance(state, torch.Tensor)
    assert state.shape == (3, 6)
    assert torch.equal(state, torch.zeros_like(state))

    latent = torch.randn(3, 4, requires_grad=True)
    action = torch.randn(3, 2, requires_grad=True)
    prediction, next_state = predictor.step(latent, action, state, torch.ones(3))
    prediction.sum().backward()

    assert prediction.shape == (3, 4)
    assert isinstance(next_state, torch.Tensor)
    assert latent.grad is not None
    assert action.grad is not None
    assert any(parameter.grad is not None for parameter in predictor.parameters())


@pytest.mark.parametrize("predictor_class", [PredictorCfC, PredictorLTC])
def test_liquid_repeated_step_matches_rollout(predictor_class: Any) -> None:
    predictor = predictor_class(latent_dim=4, action_dim=2, hidden_dim=6)
    predictor.eval()
    initial_latent = torch.randn(2, 4)
    actions = torch.randn(2, 3, 2)
    dt = torch.ones(2, 3)
    rollout, rollout_state = predictor.rollout(initial_latent, actions, dt=dt)

    state = predictor.init_state(2, "cpu")
    current = initial_latent
    predictions = []
    for index, action in enumerate(actions.unbind(dim=1)):
        current, state = predictor.step(current, action, state, dt[:, index])
        predictions.append(current)

    assert torch.allclose(rollout, torch.stack(predictions, dim=1))
    assert isinstance(rollout_state, torch.Tensor)
    assert isinstance(state, torch.Tensor)
    assert torch.allclose(rollout_state, state)
