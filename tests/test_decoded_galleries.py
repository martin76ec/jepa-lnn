"""Tests for post-hoc decoded rollout galleries."""

import json
from pathlib import Path

import imageio.v3 as imageio
import torch

from lewm_liquid_predictors.data import ObservationTrajectory
from lewm_liquid_predictors.decoder import (
    CrossAttentionDecoder,
    DecoderArchitecture,
    write_decoded_galleries,
)


def _decoder() -> CrossAttentionDecoder:
    return CrossAttentionDecoder(
        DecoderArchitecture(
            latent_dim=8,
            hidden_dim=16,
            image_size=16,
            patch_size=8,
            num_layers=2,
            num_heads=4,
        )
    )


def test_write_decoded_galleries_creates_labeled_png_and_metadata(tmp_path: Path) -> None:
    trajectory = ObservationTrajectory(
        "episode-000001",
        torch.randint(0, 256, (3, 12, 20, 3), dtype=torch.uint8),
        torch.zeros(2, 4),
    )
    decoder = _decoder()
    decoder.train()

    write_decoded_galleries(
        tmp_path,
        [(trajectory, torch.randn(2, 8))],
        decoder,
        horizons=(1, 2, 3),
        decoder_checkpoint="checkpoints/decoder.pt",
    )

    gallery = imageio.imread(tmp_path / "decoded_episode-000001.png")
    metadata = json.loads((tmp_path / "decoded.json").read_text(encoding="utf-8"))
    assert gallery.shape == (30 + 2 * (24 + 16), 3 * 16, 3)
    assert decoder.training
    assert metadata["generated_by"] == "post-hoc decoder"
    assert metadata["gallery_columns"] == [
        "Initial input",
        "Decoded prediction",
        "Actual future",
    ]
    assert metadata["decoder_checkpoint"] == "checkpoints/decoder.pt"
    assert metadata["records"] == [
        {
            "query_index": 0,
            "query_episode_id": "episode-000001",
            "input_timestep": 0,
            "horizon": 1,
            "actual_timestep": 1,
        },
        {
            "query_index": 0,
            "query_episode_id": "episode-000001",
            "input_timestep": 0,
            "horizon": 2,
            "actual_timestep": 2,
        },
    ]


def test_write_decoded_galleries_requires_existing_output_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    try:
        write_decoded_galleries(missing, [], _decoder(), horizons=(1,))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing output directory was accepted")
