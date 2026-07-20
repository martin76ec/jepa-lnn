"""Tests for the shared encoder/action/predictor composition."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lewm_liquid_predictors.data import ObservationTrajectory, collate_observation_trajectories
from lewm_liquid_predictors.models import LeWMEncoder, PredictorMLP, PredictorSystem
from lewm_liquid_predictors.training import PredictorTrainer


@dataclass(frozen=True)
class _EncoderOutput:
    last_hidden_state: Tensor


class _VisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_batch_size = 0

    def forward(self, pixels: Tensor, *, interpolate_pos_encoding: bool) -> _EncoderOutput:
        self.last_batch_size = pixels.shape[0]
        cls_token = pixels.mean(dim=(1, 2, 3)).unsqueeze(-1)
        return _EncoderOutput(cls_token.unsqueeze(1))


def _observation_batch() -> object:
    first = ObservationTrajectory(
        "first",
        observations=torch.ones(3, 3, 2, 2),
        actions=torch.ones(2, 2),
    )
    second = ObservationTrajectory(
        "second",
        observations=torch.ones(2, 3, 2, 2),
        actions=torch.ones(1, 2),
    )
    return collate_observation_trajectories([first, second])


def _system(vision_encoder: _VisionEncoder) -> PredictorSystem:
    return PredictorSystem(
        encoder=LeWMEncoder(vision_encoder, nn.Linear(1, 1)),
        action_encoder=nn.Linear(2, 1),
        predictor=PredictorMLP(latent_dim=1, action_dim=1, hidden_dim=4),
    )


def test_system_encodes_only_valid_observations_and_actions() -> None:
    vision_encoder = _VisionEncoder()
    system = _system(vision_encoder)
    batch = _observation_batch()
    assert hasattr(batch, "transition_mask")

    encoded = system.encode_batch(batch)

    assert vision_encoder.last_batch_size == 5
    assert encoded.latents.shape == (2, 3, 1)
    assert encoded.actions.shape == (2, 2, 1)
    assert torch.equal(encoded.transition_mask, batch.transition_mask)
    assert encoded.latents[1, 2].item() == 0
    assert encoded.actions[1, 1].item() == 0


def test_system_trainer_updates_shared_and_predictor_parameters() -> None:
    vision_encoder = _VisionEncoder()
    system = _system(vision_encoder)
    batch = _observation_batch()
    assert hasattr(batch, "observations")
    before = [parameter.detach().clone() for parameter in system.parameters()]

    metrics = PredictorTrainer(
        system, torch.optim.SGD(system.parameters(), lr=0.01)
    ).train_observation_epoch([batch])

    assert metrics.transitions == 3
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, system.parameters(), strict=True)
    )
