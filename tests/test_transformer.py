"""Tests for the causal Transformer predictor."""

import torch

from lewm_liquid_predictors.models import PredictorTransformer, TransformerState


def _predictor() -> PredictorTransformer:
    return PredictorTransformer(
        latent_dim=4,
        action_dim=2,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        context_length=3,
    )


def test_transformer_state_resets_and_is_bounded() -> None:
    predictor = _predictor()
    state = predictor.init_state(2, "cpu")
    assert isinstance(state, TransformerState)
    latent = torch.randn(2, 4)
    for _ in range(5):
        _, state = predictor.step(latent, torch.randn(2, 2), state, None)

    assert isinstance(state, TransformerState)
    assert state.features.shape == (2, 3, 6)


def test_repeated_step_matches_autoregressive_rollout() -> None:
    predictor = _predictor()
    predictor.eval()
    initial_latent = torch.randn(2, 4)
    actions = torch.randn(2, 4, 2)
    rollout, rollout_state = predictor.rollout(initial_latent, actions)

    state = predictor.init_state(2, "cpu")
    current = initial_latent
    predictions = []
    for action in actions.unbind(dim=1):
        current, state = predictor.step(current, action, state, None)
        predictions.append(current)

    assert torch.allclose(rollout, torch.stack(predictions, dim=1))
    assert isinstance(rollout_state, TransformerState)
    assert isinstance(state, TransformerState)
    assert torch.allclose(rollout_state.features, state.features)


def test_transformer_cannot_access_future_actions() -> None:
    predictor = _predictor()
    predictor.eval()
    initial_latent = torch.randn(1, 4)
    actions = torch.randn(1, 4, 2)
    modified_actions = actions.clone()
    modified_actions[:, 3] = 1_000

    predictions, _ = predictor.rollout(initial_latent, actions)
    modified_predictions, _ = predictor.rollout(initial_latent, modified_actions)

    assert torch.allclose(predictions[:, :3], modified_predictions[:, :3])
    assert not torch.allclose(predictions[:, 3], modified_predictions[:, 3])
