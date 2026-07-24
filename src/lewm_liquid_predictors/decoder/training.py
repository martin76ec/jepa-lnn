"""Focused training utilities for a decoder over a frozen LeWM encoder."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset

from lewm_liquid_predictors.data import (
    ObservationTrajectory,
    channels_first,
    preprocess_observations,
)
from lewm_liquid_predictors.models.encoder import LeWMEncoder

from .model import CrossAttentionDecoder


class FrameDataset(Dataset[Tensor]):
    """Index every frame in complete trajectories without copying frame tensors."""

    def __init__(self, trajectories: tuple[ObservationTrajectory, ...]) -> None:
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        self._trajectories = trajectories
        self._frame_indices = tuple(
            (trajectory_index, frame_index)
            for trajectory_index, trajectory in enumerate(trajectories)
            for frame_index in range(trajectory.observations.shape[0])
        )

    def __len__(self) -> int:
        """Return the total number of observation frames."""
        return len(self._frame_indices)

    def __getitem__(self, index: int) -> Tensor:
        """Return one frame as a view into its trajectory observation tensor."""
        trajectory_index, frame_index = self._frame_indices[index]
        return self._trajectories[trajectory_index].observations[frame_index]


@dataclass(frozen=True)
class DecoderEpochMetrics:
    """Aggregate reconstruction metrics from one decoder epoch."""

    mean_squared_error: float
    frames: int


class DecoderTrainer:
    """Train only a decoder against normalized pixels from a frozen encoder."""

    def __init__(
        self,
        encoder: LeWMEncoder,
        decoder: CrossAttentionDecoder,
        optimizer: Optimizer,
        *,
        use_amp: bool = True,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.encoder.requires_grad_(False).eval()

    def train_epoch(self, batches: Iterable[Tensor]) -> DecoderEpochMetrics:
        """Train for one epoch over individual or batched raw RGB frame tensors."""
        self.decoder.train()
        self.encoder.eval()
        decoder_device = _module_device(self.decoder)
        encoder_device = _module_device(self.encoder, fallback=decoder_device)
        amp_enabled = self.use_amp and decoder_device.type == "cuda"
        squared_error_sum = 0.0
        frame_count = 0

        for raw_frames in batches:
            frames = _frame_batch(raw_frames).to(encoder_device)
            normalized = preprocess_observations(
                frames,
                img_size=self.decoder.architecture.image_size,
            )
            with torch.no_grad():
                latent = self.encoder(normalized)
            latent = latent[:, 0].to(decoder_device)
            targets = normalized[:, 0].to(decoder_device)

            self.optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                predictions = self.decoder(latent)
                loss = torch.nn.functional.mse_loss(predictions, targets)
            torch.autograd.backward(loss)
            self.optimizer.step()

            batch_size = targets.shape[0]
            squared_error_sum += float(loss.detach().item()) * batch_size
            frame_count += batch_size

        if frame_count == 0:
            raise ValueError("no frame batches provided")
        self.encoder.eval()
        return DecoderEpochMetrics(
            mean_squared_error=squared_error_sum / frame_count,
            frames=frame_count,
        )


def _frame_batch(frames: Tensor) -> Tensor:
    if not isinstance(frames, Tensor):
        raise TypeError("frame batches must be tensors")
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError("frames must have CHW/HWC or BCHW/BHWC shape")
    return channels_first(frames).unsqueeze(1)


def _module_device(module: torch.nn.Module, fallback: torch.device | None = None) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return fallback or torch.device("cpu")


def _autocast(enabled: bool) -> AbstractContextManager[object]:
    if not enabled:
        return nullcontext()
    return cast(
        AbstractContextManager[object],
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    )
