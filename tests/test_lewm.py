"""Tests for the LeWM baseline reproduction."""

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

from lewm_liquid_predictors.models import (
    ARPredictor,
    LeWMARPredictor,
    LeWMPredictorView,
    SIGReg,
    build_lewm_baseline,
    teacher_forced_rollout,
)

upstream_required = pytest.mark.skipif(
    find_spec("transformers") is None, reason="requires transformers (upstream extra)"
)


def test_sigreg_returns_finite_scalar_for_gaussian_input() -> None:
    sigreg = SIGReg(knots=17, num_proj=1024)
    proj = torch.randn(20, 4, 192)  # (T, B, D)
    loss = sigreg(proj)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_sigreg_penalizes_non_gaussian_input_more_than_gaussian() -> None:
    sigreg = SIGReg(knots=17, num_proj=1024)
    torch.manual_seed(42)
    gaussian = torch.randn(50, 4, 192)
    collapsed = torch.ones(50, 4, 192) * 0.5

    gaussian_loss = sigreg(gaussian)
    collapsed_loss = sigreg(collapsed)

    assert collapsed_loss > gaussian_loss


def test_arpredictor_outputs_match_input_sequence_length() -> None:
    predictor = ARPredictor(
        num_frames=3,
        depth=2,
        heads=4,
        mlp_dim=256,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        dim_head=64,
        dropout=0.0,
    )
    emb = torch.randn(2, 3, 192)
    act = torch.randn(2, 3, 192)

    output = predictor(emb, act)

    assert output.shape == (2, 3, 192)


def test_arpredictor_cannot_attend_future_positions() -> None:
    predictor = ARPredictor(
        num_frames=3,
        depth=2,
        heads=4,
        mlp_dim=256,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        dim_head=64,
        dropout=0.0,
    )
    predictor.eval()
    emb = torch.randn(1, 3, 192)
    act = torch.randn(1, 3, 192)
    modified_act = act.clone()
    modified_act[:, 2] = 1_000

    output = predictor(emb, act)
    modified_output = predictor(emb, modified_act)

    # Causal: positions 0 and 1 must not change when future action changes.
    assert torch.allclose(output[:, :2], modified_output[:, :2], atol=1e-5)


def test_lewm_ar_adapter_rollout_matches_repeated_steps() -> None:
    predictor = LeWMARPredictor(latent_dim=192, action_dim=192, history_size=3).eval()
    initial = torch.randn(2, 192)
    actions = torch.randn(2, 4, 192)

    with torch.no_grad():
        rollout, _ = predictor.rollout(initial, actions)
        state = predictor.init_state(initial.shape[0], str(initial.device))
        latent = initial
        repeated = []
        for action in actions.unbind(dim=1):
            latent, state = predictor.step(latent, action, state, None)
            repeated.append(latent)

    assert torch.allclose(rollout, torch.stack(repeated, dim=1), atol=1e-6)


class _AdditiveLeWM:
    history_size = 2
    predictor = SimpleNamespace(pos_embedding=torch.zeros(1, 2, 1))

    def predict(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return latents + actions


def test_non_owning_lewm_view_distinguishes_teacher_forcing_from_rollout() -> None:
    view = LeWMPredictorView(_AdditiveLeWM())  # type: ignore[arg-type]
    observed_latents = torch.tensor([[[1.0], [100.0], [1000.0]]])
    actions = torch.tensor([[[2.0], [3.0]]])

    teacher_forced = teacher_forced_rollout(view, observed_latents, actions)
    rollout, _ = view.rollout(observed_latents[:, 0], actions)

    assert torch.equal(teacher_forced, torch.tensor([[[3.0], [103.0]]]))
    assert torch.equal(rollout, torch.tensor([[[3.0], [6.0]]]))


@upstream_required
def test_build_lewm_baseline_produces_finite_two_term_loss() -> None:
    model = build_lewm_baseline(latent_dim=192, action_dim=10, history_size=3)
    batch = {
        "pixels": torch.randn(2, 4, 3, 224, 224),
        "action": torch.randn(2, 4, 10),
    }

    output = model(batch)

    assert "pred_loss" in output
    assert "sigreg_loss" in output
    assert "loss" in output
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["pred_loss"])
    assert torch.isfinite(output["sigreg_loss"])
    assert output["loss"] == output["pred_loss"] + model.sigreg_weight * output["sigreg_loss"]


@upstream_required
def test_lewm_baseline_loss_supports_gradients() -> None:
    model = build_lewm_baseline(latent_dim=192, action_dim=10, history_size=3)
    batch = {
        "pixels": torch.randn(1, 4, 3, 224, 224),
        "action": torch.randn(1, 4, 10),
    }

    output = model(batch)
    output["loss"].backward()

    has_grad = any(param.grad is not None for param in model.parameters())
    assert has_grad
