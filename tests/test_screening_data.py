"""Tests for reusable controlled screening data preparation."""

from collections.abc import Mapping, Sized
from pathlib import Path

import pytest
import torch
from torch import Tensor

from lewm_liquid_predictors.data import (
    ObservationTrajectory,
    ObservationWindowDataset,
    action_normalizer_from_statistics,
    dataset_identity,
    load_or_create_split_manifest,
    prepare_screening_data_from_sources,
    split_manifest_digest,
    stream_action_statistics,
)


class _Source:
    def __init__(self, episodes: tuple[Mapping[str, Tensor], ...]) -> None:
        self._episodes = episodes
        self.loaded_indices: list[int] = []

    @property
    def lengths(self) -> Sized:
        return self._episodes

    def load_episode(self, episode_idx: int) -> Mapping[str, Tensor]:
        self.loaded_indices.append(episode_idx)
        return self._episodes[episode_idx]


def test_dataset_fingerprint_changes_when_lance_metadata_changes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.lance"
    dataset.mkdir()
    metadata = dataset / "manifest"
    metadata.write_bytes(b"first")

    source, first = dataset_identity(dataset)
    metadata.write_bytes(b"different-size")
    _, second = dataset_identity(dataset)

    assert source == str(dataset.resolve())
    assert first != second


def test_streaming_action_statistics_match_torch_sample_std_and_skip_nan_rows() -> None:
    source = _Source(
        (
            {"action": torch.tensor([[1.0, 10.0], [float("nan"), 12.0]])},
            {"action": torch.tensor([[3.0, 14.0], [5.0, 18.0]])},
        )
    )
    expected = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])

    statistics = stream_action_statistics(source)

    assert source.loaded_indices == [0, 1]
    assert statistics.sample_count == 3
    torch.testing.assert_close(statistics.mean, expected.mean(0, keepdim=True))
    torch.testing.assert_close(statistics.std, expected.std(0, keepdim=True))


def test_action_statistics_repeat_in_frameskip_flattening_order() -> None:
    source = _Source(({"action": torch.tensor([[1.0, 10.0], [3.0, 14.0]])},))

    normalizer = action_normalizer_from_statistics(stream_action_statistics(source), frameskip=3)

    assert torch.equal(normalizer.mean, torch.tensor([[2.0, 12.0, 2.0, 12.0, 2.0, 12.0]]))
    torch.testing.assert_close(
        normalizer.std,
        torch.tensor([[2.0**0.5, 8.0**0.5] * 3]),
    )


def test_load_or_create_manifest_validates_all_identity_inputs(tmp_path: Path) -> None:
    path = tmp_path / "manifests" / "split.json"
    episode_ids = tuple(f"episode-{index:06d}" for index in range(5))
    manifest = load_or_create_split_manifest(
        path, dataset="pusht", episode_ids=episode_ids, seed=17
    )

    assert path.exists()
    assert (
        load_or_create_split_manifest(path, dataset="pusht", episode_ids=episode_ids, seed=17)
        == manifest
    )
    assert split_manifest_digest(manifest) == split_manifest_digest(manifest)
    with pytest.raises(ValueError, match="dataset"):
        load_or_create_split_manifest(path, dataset="other", episode_ids=episode_ids, seed=17)
    with pytest.raises(ValueError, match="seed"):
        load_or_create_split_manifest(path, dataset="pusht", episode_ids=episode_ids, seed=18)
    with pytest.raises(ValueError, match="episodes"):
        load_or_create_split_manifest(
            path, dataset="pusht", episode_ids=(*episode_ids[:-1], "replacement"), seed=17
        )


def test_preparation_does_not_materialize_pixel_episodes(tmp_path: Path) -> None:
    pixel_source = _Source(
        tuple(
            {
                "pixels": torch.zeros(3, 3, 2, 2, dtype=torch.uint8),
                "action": torch.zeros(6, 2),
            }
            for _ in range(5)
        )
    )
    action_source = _Source(
        tuple({"action": torch.arange(12, dtype=torch.float32).reshape(6, 2)} for _ in range(5))
    )

    prepared = prepare_screening_data_from_sources(
        pixel_source,
        action_source,
        dataset="pusht",
        manifest_path=tmp_path / "split.json",
        split_seed=7,
        frameskip=2,
    )

    assert pixel_source.loaded_indices == []
    assert action_source.loaded_indices == [0, 1, 2, 3, 4]
    assert prepared.manifest_path == (tmp_path / "split.json").resolve()
    assert prepared.manifest_digest == split_manifest_digest(prepared.manifest)
    assert prepared.action_input_dim == 4


def test_fixed_windows_preserve_episode_boundaries() -> None:
    first = ObservationTrajectory(
        "first", torch.arange(5).reshape(5, 1), torch.arange(4).reshape(4, 1)
    )
    second = ObservationTrajectory(
        "second", torch.arange(3).reshape(3, 1), torch.arange(2).reshape(2, 1)
    )

    dataset = ObservationWindowDataset((first, second), window_size=3)

    assert len(dataset) == 4
    assert [dataset[index].episode_id for index in range(len(dataset))] == [
        "first",
        "first",
        "first",
        "second",
    ]
    assert all(dataset[index].observations.shape[0] == 3 for index in range(len(dataset)))
