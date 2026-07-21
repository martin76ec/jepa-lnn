"""Tests for image and action preprocessing."""

import torch

from lewm_liquid_predictors.data import (
    fit_zscore_normalizer,
    normalize_pixels,
    preprocess_observations,
    resize_observations,
)


def test_normalize_pixels_converts_uint8_to_normalized_float() -> None:
    pixels = torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8)
    normalized = normalize_pixels(pixels)

    assert normalized.dtype == torch.float32
    assert normalized.shape == pixels.shape
    assert normalized.min() < 0
    assert normalized.max() > 0


def test_resize_observations_changes_spatial_dimensions() -> None:
    pixels = torch.randn(2, 4, 3, 64, 64)
    resized = resize_observations(pixels, size=224)

    assert resized.shape == (2, 4, 3, 224, 224)


def test_preprocess_observations_applies_both_normalization_and_resize() -> None:
    pixels = torch.randint(0, 256, (1, 2, 3, 64, 64), dtype=torch.uint8)
    processed = preprocess_observations(pixels, img_size=224)

    assert processed.shape == (1, 2, 3, 224, 224)
    assert processed.dtype == torch.float32


def test_zscore_normalizer_ignores_nan_rows_when_fitting() -> None:
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [float("nan"), 5.0], [5.0, 6.0]])
    normalizer = fit_zscore_normalizer(data)

    result = normalizer(torch.tensor([[3.0, 4.0]]))
    assert result.shape == (1, 2)
    assert torch.isfinite(result).all()
