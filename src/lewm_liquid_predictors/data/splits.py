"""Deterministic trajectory-level splits and manifests."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

SplitName: TypeAlias = Literal["train", "validation", "test"]
_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitManifest:
    """Immutable, serializable assignment of complete episodes to data splits."""

    dataset: str
    seed: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        episode_ids = self.train + self.validation + self.test
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("split manifest assigns an episode more than once")
        if not all(episode_ids):
            raise ValueError("split manifest episode IDs must be non-empty")

    def episode_ids(self, split: SplitName) -> tuple[str, ...]:
        """Return the episode IDs assigned to one split."""
        if split == "train":
            return self.train
        if split == "validation":
            return self.validation
        return self.test

    def to_dict(self) -> dict[str, object]:
        """Convert the manifest to a JSON-compatible mapping."""
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "splits": {split: list(self.episode_ids(split)) for split in _SPLITS},
        }

    def validate_episode_ids(self, episode_ids: Iterable[str]) -> None:
        """Require the manifest to cover exactly the supplied source episodes."""
        source_ids = tuple(episode_ids)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source episode IDs must be unique")
        manifest_ids = self.train + self.validation + self.test
        if set(manifest_ids) != set(source_ids):
            raise ValueError("split manifest does not match the supplied source episodes")

    def validate(self, dataset: str, seed: int, episode_ids: Iterable[str]) -> None:
        """Require the manifest identity and episode universe to match a request."""
        if self.dataset != dataset:
            raise ValueError(
                f"split manifest dataset {self.dataset!r} does not match requested {dataset!r}"
            )
        if self.seed != seed:
            raise ValueError(
                f"split manifest seed {self.seed} does not match requested seed {seed}"
            )
        self.validate_episode_ids(episode_ids)


def create_split_manifest(
    dataset: str,
    episode_ids: Sequence[str],
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> SplitManifest:
    """Create a reproducible 80/10/10-style split over complete trajectories."""
    _validate_split_inputs(dataset, episode_ids, train_fraction, validation_fraction)
    shuffled_ids = sorted(episode_ids)
    random.Random(seed).shuffle(shuffled_ids)
    train_count, validation_count = _split_counts(
        len(shuffled_ids), train_fraction, validation_fraction
    )
    train_end = train_count
    validation_end = train_end + validation_count
    return SplitManifest(
        dataset=dataset,
        seed=seed,
        train=tuple(shuffled_ids[:train_end]),
        validation=tuple(shuffled_ids[train_end:validation_end]),
        test=tuple(shuffled_ids[validation_end:]),
    )


def write_split_manifest(manifest: SplitManifest, path: str | Path) -> None:
    """Write a manifest in a stable JSON representation."""
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load a split manifest written by :func:`write_split_manifest`."""
    input_path = Path(path)
    with input_path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, Mapping):
        raise ValueError("split manifest root must be a mapping")
    splits = raw.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("split manifest splits must be a mapping")
    dataset = raw.get("dataset")
    seed = raw.get("seed")
    if not isinstance(dataset, str) or not isinstance(seed, int):
        raise ValueError("split manifest dataset and seed are invalid")
    return SplitManifest(
        dataset=dataset,
        seed=seed,
        train=_episode_ids(splits, "train"),
        validation=_episode_ids(splits, "validation"),
        test=_episode_ids(splits, "test"),
    )


def sample_episode_ids(episode_ids: Sequence[str], fraction: float, seed: int) -> tuple[str, ...]:
    """Select a deterministic non-empty fraction without modifying other splits."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")
    if not episode_ids:
        raise ValueError("episode_ids must not be empty")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_ids must be unique")
    count = max(1, round(len(episode_ids) * fraction))
    indices = sorted(random.Random(seed).sample(range(len(episode_ids)), count))
    return tuple(episode_ids[index] for index in indices)


def _validate_split_inputs(
    dataset: str,
    episode_ids: Sequence[str],
    train_fraction: float,
    validation_fraction: float,
) -> None:
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if len(episode_ids) < len(_SPLITS):
        raise ValueError("at least three episodes are required for train/validation/test splits")
    if len(set(episode_ids)) != len(episode_ids) or not all(episode_ids):
        raise ValueError("episode_ids must be unique and non-empty")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test fraction")


def _split_counts(total: int, train_fraction: float, validation_fraction: float) -> tuple[int, int]:
    train_count = round(total * train_fraction)
    validation_count = round(total * validation_fraction)
    test_count = total - train_count - validation_count
    if validation_count == 0:
        validation_count = 1
        train_count -= 1
    if test_count == 0:
        test_count = 1
        train_count -= 1
    return train_count, validation_count


def _episode_ids(splits: Mapping[str, object], split: SplitName) -> tuple[str, ...]:
    raw_ids = splits.get(split)
    if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
        raise ValueError(f"split manifest {split} IDs must be a list of strings")
    return tuple(raw_ids)
