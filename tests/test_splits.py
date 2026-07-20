"""Tests for deterministic trajectory-level split manifests."""

from pathlib import Path

import pytest

from lewm_liquid_predictors.data import (
    create_split_manifest,
    load_split_manifest,
    sample_episode_ids,
    write_split_manifest,
)


def test_manifest_is_reproducible_and_assigns_each_episode_once() -> None:
    episode_ids = [f"episode-{index:02d}" for index in range(10)]

    first = create_split_manifest("pusht_expert_train", episode_ids, seed=7)
    second = create_split_manifest("pusht_expert_train", list(reversed(episode_ids)), seed=7)

    assert first == second
    assert len(first.train) == 8
    assert len(first.validation) == 1
    assert len(first.test) == 1
    first.validate_episode_ids(episode_ids)


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = create_split_manifest("pusht_expert_train", [f"episode-{i}" for i in range(7)], 19)
    path = tmp_path / "splits.json"

    write_split_manifest(manifest, path)

    assert load_split_manifest(path) == manifest


def test_local_fractional_sampling_is_reproducible_and_keeps_test_untouched() -> None:
    manifest = create_split_manifest("pusht_expert_train", [f"episode-{i}" for i in range(20)], 43)

    sampled = sample_episode_ids(manifest.train, fraction=0.25, seed=7)

    assert sampled == sample_episode_ids(manifest.train, fraction=0.25, seed=7)
    assert len(sampled) == 4
    assert set(sampled).issubset(manifest.train)
    assert set(sampled).isdisjoint(manifest.test)


def test_manifest_requires_at_least_one_episode_per_split() -> None:
    with pytest.raises(ValueError, match="at least three"):
        create_split_manifest("pusht_expert_train", ["one", "two"], 7)


def test_manifest_rejects_changed_source_episode_ids() -> None:
    manifest = create_split_manifest("pusht_expert_train", ["one", "two", "three"], 7)

    with pytest.raises(ValueError, match="does not match"):
        manifest.validate_episode_ids(["one", "two", "four"])
