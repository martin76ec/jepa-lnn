"""Tests for the shared LeWM encoder wrapper."""

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from lewm_liquid_predictors.models import LeWMEncoder


@dataclass(frozen=True)
class _EncoderOutput:
    last_hidden_state: Tensor


class _VisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_interpolate_pos_encoding: bool | None = None

    def forward(self, pixels: Tensor, *, interpolate_pos_encoding: bool) -> _EncoderOutput:
        self.last_interpolate_pos_encoding = interpolate_pos_encoding
        cls_token = pixels.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        return _EncoderOutput(last_hidden_state=cls_token.unsqueeze(1))


def test_encoder_matches_lewm_batch_time_flattening_and_preserves_gradients() -> None:
    vision_encoder = _VisionEncoder()
    projector = nn.Linear(1, 3)
    encoder = LeWMEncoder(vision_encoder, projector)
    observations = torch.randn(2, 4, 3, 8, 8, requires_grad=True)

    latents = encoder(observations)
    latents.sum().backward()

    assert latents.shape == (2, 4, 3)
    assert vision_encoder.last_interpolate_pos_encoding is True
    assert observations.grad is not None
    assert all(parameter.grad is not None for parameter in projector.parameters())


def test_encoder_rejects_non_sequence_observations() -> None:
    encoder = LeWMEncoder(_VisionEncoder(), nn.Identity())

    with pytest.raises(ValueError, match="batch, time"):
        encoder(torch.zeros(2, 3, 8, 8))
