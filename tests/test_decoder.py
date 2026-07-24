"""Focused tests for the projected-latent decoder and trainer."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from lewm_liquid_predictors.data import ObservationTrajectory
from lewm_liquid_predictors.decoder import (
    CrossAttentionDecoder,
    DecoderArchitecture,
    DecoderTrainer,
    FrameDataset,
    imagenet_to_lpips,
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
    assert metrics.total_loss == pytest.approx(metrics.mean_squared_error)
    assert metrics.mean_squared_error >= 0
    assert metrics.mean_absolute_error >= 0
    assert metrics.lpips_loss == 0
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, decoder.parameters(), strict=True)
    )


class _FakePerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.input_range: tuple[float, float] | None = None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.input_range = (
            min(float(prediction.min()), float(target.min())),
            max(float(prediction.max()), float(target.max())),
        )
        return (prediction - target).abs().mean(dim=(1, 2, 3), keepdim=True) * self.scale


def test_decoder_trainer_combines_l1_and_frozen_lpips_loss() -> None:
    encoder = LeWMEncoder(_TinyVision(), nn.Linear(3, 8))
    decoder = CrossAttentionDecoder(_architecture())
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    perceptual = _FakePerceptualLoss()
    trainer = DecoderTrainer(
        encoder,
        decoder,
        optimizer,
        use_amp=False,
        loss="l1_lpips",
        l1_weight=0.5,
        lpips_weight=0.25,
        perceptual_model=perceptual,
    )

    metrics = trainer.train_epoch([torch.randint(0, 256, (2, 18, 20, 3), dtype=torch.uint8)])

    assert metrics.lpips_loss > 0
    assert metrics.total_loss == pytest.approx(
        0.5 * metrics.mean_absolute_error + 0.25 * metrics.lpips_loss
    )
    assert perceptual.input_range is not None
    assert perceptual.input_range[0] >= -1
    assert perceptual.input_range[1] <= 1
    assert not perceptual.training
    assert all(not parameter.requires_grad for parameter in perceptual.parameters())
    assert all(parameter.grad is None for parameter in perceptual.parameters())


def test_imagenet_to_lpips_maps_black_and_white_to_required_range() -> None:
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    images = torch.cat(
        ((torch.zeros_like(mean) - mean) / std, (torch.ones_like(mean) - mean) / std)
    )

    converted = imagenet_to_lpips(images)

    assert torch.allclose(converted[0], torch.full_like(converted[0], -1))
    assert torch.allclose(converted[1], torch.ones_like(converted[1]))


def test_imagenet_to_lpips_preserves_prediction_gradients() -> None:
    images = torch.zeros(1, 3, 4, 4, requires_grad=True)

    imagenet_to_lpips(images).square().mean().backward()

    assert images.grad is not None
    assert images.grad.abs().sum() > 0


def test_decoder_trainer_steps_scheduler_per_optimizer_batch() -> None:
    encoder = LeWMEncoder(_TinyVision(), nn.Linear(3, 8))
    decoder = CrossAttentionDecoder(_architecture())
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda step: 1 / (step + 1))
    trainer = DecoderTrainer(
        encoder,
        decoder,
        optimizer,
        use_amp=False,
        scheduler=scheduler,
    )
    batch = torch.randint(0, 256, (1, 18, 20, 3), dtype=torch.uint8)

    trainer.train_epoch([batch, batch])

    assert scheduler.last_epoch == 2


def test_decoder_trainer_rejects_invalid_direct_loss() -> None:
    encoder = LeWMEncoder(_TinyVision(), nn.Linear(3, 8))
    decoder = CrossAttentionDecoder(_architecture())
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="loss must be one of"):
        DecoderTrainer(encoder, decoder, optimizer, loss="invalid")  # type: ignore[arg-type]
