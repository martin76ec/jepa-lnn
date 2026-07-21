"""Image and action preprocessing matching the upstream LeWM pipeline.

Upstream applies:
- ToImage: uint8 → float32, scale to [0,1], ImageNet normalize, RGB
- Resize: to 224x224
- Z-score normalization for action/proprio/state columns
"""

from __future__ import annotations

import torch
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_pixels(observations: Tensor) -> Tensor:
    """Convert uint8 pixels to normalized float32 with ImageNet statistics.

    Args:
        observations: Pixel tensor of shape (..., C, H, W), uint8 or float.
    """
    if observations.dtype == torch.uint8:
        observations = observations.float() / 255.0
    mean = observations.new_tensor(IMAGENET_MEAN).view(1, 1, -1, 1, 1)
    std = observations.new_tensor(IMAGENET_STD).view(1, 1, -1, 1, 1)
    return (observations - mean) / std


def resize_observations(observations: Tensor, size: int = 224) -> Tensor:
    """Bilinear resize observations to (size, size).

    Args:
        observations: Pixel tensor of shape (..., C, H, W).
    """
    import torch.nn.functional as F

    if observations.ndim < 3:
        raise ValueError("observations must have at least 3 dimensions (C, H, W)")
    leading_shape = observations.shape[:-3]
    flattened = observations.reshape(-1, *observations.shape[-3:])
    resized = F.interpolate(flattened, size=(size, size), mode="bilinear", align_corners=False)
    return resized.reshape(*leading_shape, *resized.shape[-3:])


def preprocess_observations(observations: Tensor, img_size: int = 224) -> Tensor:
    """Apply the full upstream pixel preprocessing pipeline."""
    return resize_observations(normalize_pixels(observations), img_size)


class ZScoreNormalizer:
    """Picklable z-score normalizer for action/proprio columns."""

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: Tensor) -> Tensor:
        return ((x - self.mean) / self.std).float()

    def to(self, device: torch.device | str) -> ZScoreNormalizer:
        return ZScoreNormalizer(self.mean.to(device), self.std.to(device))


def fit_zscore_normalizer(data: Tensor) -> ZScoreNormalizer:
    """Fit a z-score normalizer on non-NaN rows of a column tensor.

    Args:
        data: Tensor of shape (N, D) with possible NaN rows at sequence boundaries.
    """
    if data.ndim != 2:
        raise ValueError("data must have shape (N, D)")
    valid = ~torch.isnan(data).any(dim=1)
    clean = data[valid]
    if clean.shape[0] == 0:
        raise ValueError("no valid (non-NaN) rows to fit normalizer")
    mean = clean.mean(0, keepdim=True).clone()
    std = clean.std(0, keepdim=True).clone()
    return ZScoreNormalizer(mean, std)
