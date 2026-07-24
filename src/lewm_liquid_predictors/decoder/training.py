"""Focused training utilities for a decoder over a frozen LeWM encoder."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from typing import cast

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from lewm_liquid_predictors.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ObservationTrajectory,
    channels_first,
    preprocess_observations,
)
from lewm_liquid_predictors.models.encoder import LeWMEncoder

from .config import DecoderLoss, LPIPSNetwork
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

    total_loss: float
    mean_squared_error: float
    mean_absolute_error: float
    lpips_loss: float
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
        loss: DecoderLoss = "mse",
        l1_weight: float = 1.0,
        lpips_weight: float = 1.0,
        lpips_network: LPIPSNetwork = "vgg",
        perceptual_model: torch.nn.Module | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        if loss not in {"mse", "l1", "l1_lpips"}:
            raise ValueError("loss must be one of: mse, l1, l1_lpips")
        if not isfinite(l1_weight) or not isfinite(lpips_weight):
            raise ValueError("decoder loss weights must be finite")
        if l1_weight < 0 or lpips_weight < 0:
            raise ValueError("decoder loss weights must be nonnegative")
        if loss in {"l1", "l1_lpips"} and l1_weight <= 0:
            raise ValueError("l1_weight must be positive when L1 loss is enabled")
        if loss == "l1_lpips" and lpips_weight <= 0:
            raise ValueError("lpips_weight must be positive when LPIPS loss is enabled")
        if perceptual_model is not None and loss != "l1_lpips":
            raise ValueError("perceptual_model is only valid for l1_lpips loss")
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.loss = loss
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.scheduler = scheduler
        self.encoder.requires_grad_(False).eval()
        self.perceptual_model = perceptual_model
        if loss == "l1_lpips" and self.perceptual_model is None:
            self.perceptual_model = _build_lpips(lpips_network)
        if self.perceptual_model is not None:
            self.perceptual_model.requires_grad_(False).eval()
            self.perceptual_model.to(_module_device(decoder))

    def train_epoch(self, batches: Iterable[Tensor]) -> DecoderEpochMetrics:
        """Train for one epoch over individual or batched raw RGB frame tensors."""
        self.decoder.train()
        self.encoder.eval()
        if self.perceptual_model is not None:
            self.perceptual_model.eval()
        decoder_device = _module_device(self.decoder)
        encoder_device = _module_device(self.encoder, fallback=decoder_device)
        amp_enabled = self.use_amp and decoder_device.type == "cuda"
        total_loss_sum = 0.0
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        lpips_sum = 0.0
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
                mean_squared_error = torch.nn.functional.mse_loss(predictions, targets)
                mean_absolute_error = torch.nn.functional.l1_loss(predictions, targets)
                perceptual_loss = self._perceptual_loss(predictions, targets)
                total_loss = self._total_loss(
                    mean_squared_error,
                    mean_absolute_error,
                    perceptual_loss,
                )
            torch.autograd.backward(total_loss)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_size = targets.shape[0]
            total_loss_sum += float(total_loss.detach().item()) * batch_size
            squared_error_sum += float(mean_squared_error.detach().item()) * batch_size
            absolute_error_sum += float(mean_absolute_error.detach().item()) * batch_size
            lpips_sum += float(perceptual_loss.detach().item()) * batch_size
            frame_count += batch_size

        if frame_count == 0:
            raise ValueError("no frame batches provided")
        self.encoder.eval()
        return DecoderEpochMetrics(
            total_loss=total_loss_sum / frame_count,
            mean_squared_error=squared_error_sum / frame_count,
            mean_absolute_error=absolute_error_sum / frame_count,
            lpips_loss=lpips_sum / frame_count,
            frames=frame_count,
        )

    def _perceptual_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if self.perceptual_model is None:
            return predictions.new_zeros(())
        result = self.perceptual_model(
            imagenet_to_lpips(predictions),
            imagenet_to_lpips(targets),
        )
        if not isinstance(result, Tensor):
            raise TypeError("LPIPS model must return a tensor")
        return result.mean()

    def _total_loss(
        self,
        mean_squared_error: Tensor,
        mean_absolute_error: Tensor,
        perceptual_loss: Tensor,
    ) -> Tensor:
        if self.loss == "mse":
            return mean_squared_error
        if self.loss == "l1":
            return self.l1_weight * mean_absolute_error
        return self.l1_weight * mean_absolute_error + self.lpips_weight * perceptual_loss


def imagenet_to_lpips(images: Tensor) -> Tensor:
    """Convert ImageNet-normalized RGB images to LPIPS's required [-1, 1] range."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("LPIPS images must have shape (batch, 3, height, width)")
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixels = (images * std + mean).clamp(0, 1)
    return pixels.mul(2).sub(1)


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


def _build_lpips(network: LPIPSNetwork) -> torch.nn.Module:
    try:
        lpips = import_module("lpips")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "LPIPS loss requires the decoder dependencies; run `make sync`"
        ) from error
    model_class = getattr(lpips, "LPIPS", None)
    if model_class is None:
        raise RuntimeError("installed lpips package does not expose LPIPS")
    model = model_class(net=network, verbose=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("lpips.LPIPS must return a torch module")
    return model
