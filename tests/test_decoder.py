"""Focused tests for the projected-latent decoder and trainer."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from lewm_liquid_predictors.data import ObservationTrajectory
from lewm_liquid_predictors.decoder import (
    CrossAttentionDecoder,
    DecoderArchitecture,
    DecoderTrainer,
    FrameDataset,
)
from lewm_liquid_predictors.models import LeWMEncoder


def _architecture(**overrides: object) -> DecoderArchitecture:
    values: dict[str, object] = {
        "latent_dim": 8,
        "hidden_dim": 16,
        "image_size": 16,
        "patch_size": 8,
        "channels": 3,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return DecoderArchitecture(**values)  # type: ignore[arg-type]


def test_decoder_shape_determinism_and_gradients() -> None:
    decoder = CrossAttentionDecoder(_architecture())
    latent = torch.randn(3, 8, requires_grad=True)

    first = decoder(latent)
    second = decoder(latent)
    first.square().mean().backward()

    assert first.shape == (3, 3, 16, 16)
    assert torch.equal(first, second)
    assert latent.grad is not None
    assert latent.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in decoder.parameters())


@pytest.mark.parametrize(
    "overrides",
    [
        {"image_size": 17},
        {"hidden_dim": 15},
        {"dropout": -0.1},
        {"channels": 1},
        {"num_layers": 0},
    ],
)
def test_decoder_rejects_invalid_architecture(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _architecture(**overrides)


def test_decoder_rejects_unprojected_latent_shape() -> None:
    decoder = CrossAttentionDecoder(_architecture())

    with pytest.raises(ValueError, match="latent must have shape"):
        decoder(torch.randn(2, 9))


def _trajectory(episode_id: str, frames: int) -> ObservationTrajectory:
    return ObservationTrajectory(
        episode_id,
        torch.randint(0, 256, (frames, 18, 20, 3), dtype=torch.uint8),
        torch.zeros(frames - 1, 2),
    )


def test_frame_dataset_flattens_indices_without_copying_frames() -> None:
    trajectories = (_trajectory("first", 2), _trajectory("second", 3))
    dataset = FrameDataset(trajectories)

    assert len(dataset) == 5
    assert torch.equal(dataset[2], trajectories[1].observations[0])
    assert (
        dataset[4].untyped_storage().data_ptr()
        == trajectories[1].observations.untyped_storage().data_ptr()
    )


@dataclass
class _VisionOutput:
    last_hidden_state: torch.Tensor


class _TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        pixels: torch.Tensor,
        *,
        interpolate_pos_encoding: bool,
    ) -> _VisionOutput:
        assert interpolate_pos_encoding
        token = pixels.mean(dim=(-2, -1)).unsqueeze(1) * self.scale
        return _VisionOutput(token)


def test_decoder_trainer_keeps_encoder_frozen_and_in_eval_mode() -> None:
    encoder = LeWMEncoder(_TinyVision(), nn.Linear(3, 8))
    decoder = CrossAttentionDecoder(_architecture())
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    trainer = DecoderTrainer(encoder, decoder, optimizer, use_amp=False)
    before = [parameter.detach().clone() for parameter in decoder.parameters()]

    encoder.train()
    metrics = trainer.train_epoch(
        [
            torch.randint(0, 256, (2, 18, 20, 3), dtype=torch.uint8),
            torch.rand(3, 20, 18),
        ]
    )

    assert metrics.frames == 3
    assert metrics.mean_squared_error >= 0
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, decoder.parameters(), strict=True)
    )
