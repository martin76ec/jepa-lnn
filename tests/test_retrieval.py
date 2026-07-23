"""Tests for latent nearest-frame retrieval diagnostics."""

import json
from pathlib import Path

import torch

from lewm_liquid_predictors.data import ObservationTrajectory
from lewm_liquid_predictors.evaluation import (
    build_retrieval_gallery,
    image_array,
    write_retrieval_galleries,
)


class _Source:
    def __init__(self, episodes: list[dict[str, torch.Tensor]]) -> None:
        self._episodes = episodes
        self.lengths = tuple(len(episode["pixels"]) for episode in episodes)

    def load_episode(self, episode_idx: int) -> dict[str, torch.Tensor]:
        return self._episodes[episode_idx]


def test_retrieval_image_conversion_accepts_hwc_and_chw_pixels() -> None:
    chw = torch.arange(3 * 2 * 4, dtype=torch.uint8).reshape(3, 2, 4)
    hwc = chw.permute(1, 2, 0)

    chw_image = image_array(chw)
    hwc_image = image_array(hwc)

    assert chw_image.shape == (2, 4, 3)
    assert (chw_image == hwc_image).all()


def test_retrieval_gallery_has_labeled_three_column_rows() -> None:
    input_frame = torch.zeros(3, 20, 30, dtype=torch.uint8)
    predicted = torch.ones(3, 20, 30, dtype=torch.uint8)
    actual = torch.full((3, 20, 30), 2, dtype=torch.uint8)

    gallery = build_retrieval_gallery(
        image_array(input_frame),
        [
            (1, image_array(predicted), image_array(actual)),
            (5, image_array(predicted), image_array(actual)),
        ],
    )

    assert gallery.shape == (30 + 2 * (24 + 20), 3 * 30, 3)


def test_retrieval_excludes_the_complete_query_episode(tmp_path: Path) -> None:
    pixels = torch.zeros(2, 3, 20, 20, dtype=torch.uint8)
    actions = torch.zeros(10, 2)
    source = _Source(
        [
            {"pixels": pixels, "action": actions},
            {"pixels": pixels + 1, "action": actions},
        ]
    )
    query = ObservationTrajectory("episode-000000", pixels, torch.zeros(1, 10))

    write_retrieval_galleries(
        tmp_path,
        [(query, torch.zeros(1, 1))],
        torch.tensor([[0.0], [0.0], [1.0], [1.0]]),
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        source,
        frameskip=5,
        horizons=(1,),
    )

    metadata = json.loads((tmp_path / "retrieval.json").read_text(encoding="utf-8"))
    assert metadata["records"][0]["retrieved_episode_id"] == "episode-000001"
    assert metadata["reference_policy"]["exclude_query_episode"] is True
    assert metadata["reference_policy"]["episode_count"] == 2
