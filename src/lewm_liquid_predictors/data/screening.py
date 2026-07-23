"""Reusable data preparation for controlled predictor screening."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .preprocessing import ZScoreNormalizer
from .pusht import (
    EpisodeSource,
    ObservationTrajectory,
    adapt_pusht_episode,
    open_pusht_lance_source,
)
from .splits import (
    SplitManifest,
    create_split_manifest,
    load_split_manifest,
    sample_episode_ids,
    write_split_manifest,
)


@dataclass(frozen=True)
class ActionStatistics:
    """Sample statistics over valid raw action rows."""

    mean: Tensor
    std: Tensor
    sample_count: int

    def __post_init__(self) -> None:
        if self.mean.ndim != 2 or self.mean.shape[0] != 1:
            raise ValueError("mean must have shape (1, raw_action_dim)")
        if self.std.shape != self.mean.shape:
            raise ValueError("std must have the same shape as mean")
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least two")
        if self.mean.device != self.std.device or self.mean.dtype != self.std.dtype:
            raise ValueError("mean and std must have the same dtype and device")

    @property
    def raw_action_dim(self) -> int:
        """Return the unstacked action dimension."""
        return self.mean.shape[1]


@dataclass(frozen=True)
class ScreeningData:
    """Sources, split identity, and statistics prepared without training windows."""

    source: EpisodeSource
    manifest: SplitManifest
    manifest_path: Path
    manifest_digest: str
    train_episode_ids: tuple[str, ...]
    action_statistics: ActionStatistics
    frameskip: int
    dataset_source: str
    dataset_fingerprint: str

    @property
    def validation_episode_ids(self) -> tuple[str, ...]:
        """Return all held-out validation episode IDs."""
        return self.manifest.validation

    @property
    def test_episode_ids(self) -> tuple[str, ...]:
        """Return all held-out test episode IDs."""
        return self.manifest.test

    @property
    def action_input_dim(self) -> int:
        """Return the flattened frameskip action dimension."""
        return self.action_statistics.raw_action_dim * self.frameskip

    @property
    def action_normalizer(self) -> ZScoreNormalizer:
        """Build a normalizer for frameskip-flattened action blocks."""
        return action_normalizer_from_statistics(self.action_statistics, self.frameskip)


class ObservationWindowDataset(Dataset[ObservationTrajectory]):
    """Fixed-length observation windows that preserve episode boundaries."""

    def __init__(
        self,
        trajectories: tuple[ObservationTrajectory, ...],
        window_size: int,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must include at least two observations")
        self.window_size = window_size
        self._trajectories = trajectories
        self._windows = tuple(
            (trajectory_index, start)
            for trajectory_index, trajectory in enumerate(trajectories)
            for start in range(trajectory.observations.shape[0] - window_size + 1)
        )
        if not self._windows:
            raise ValueError("training trajectories are shorter than the context window")

    def __len__(self) -> int:
        """Return the number of fixed-length windows."""
        return len(self._windows)

    def __getitem__(self, index: int) -> ObservationTrajectory:
        """Return one fixed-length window without crossing an episode boundary."""
        trajectory_index, start = self._windows[index]
        trajectory = self._trajectories[trajectory_index]
        return ObservationTrajectory(
            episode_id=trajectory.episode_id,
            observations=trajectory.observations[start : start + self.window_size],
            actions=trajectory.actions[start : start + self.window_size - 1],
        )


def stream_action_statistics(source: EpisodeSource) -> ActionStatistics:
    """Compute upstream-equivalent statistics over all valid raw action rows.

    Only the small action column is retained; pixels remain lazy. A single final
    ``mean``/``std`` reduction matches the upstream implementation rather than
    introducing position-dependent streaming roundoff.
    """
    valid_batches: list[Tensor] = []
    expected_shape: tuple[int, ...] | None = None
    expected_dtype: torch.dtype | None = None
    expected_device: torch.device | None = None
    for episode_index in range(len(source.lengths)):
        actions = _action_rows(source.load_episode(episode_index))
        valid_actions = actions[~torch.isnan(actions).any(dim=1)]
        if valid_actions.shape[0] == 0:
            continue
        if expected_shape is None:
            expected_shape = valid_actions.shape[1:]
            expected_dtype = valid_actions.dtype
            expected_device = valid_actions.device
        if valid_actions.shape[1:] != expected_shape:
            raise ValueError("all action rows must have the same flattened dimension")
        if valid_actions.dtype != expected_dtype or valid_actions.device != expected_device:
            raise ValueError("all action rows must have the same dtype and device")
        valid_batches.append(valid_actions)
    if not valid_batches:
        raise ValueError("at least two valid action rows are required for sample statistics")
    data = torch.cat(valid_batches)
    if data.shape[0] < 2:
        raise ValueError("at least two valid action rows are required for sample statistics")
    return ActionStatistics(
        data.mean(dim=0, keepdim=True).clone(),
        data.std(dim=0, keepdim=True).clone(),
        data.shape[0],
    )


def action_normalizer_from_statistics(
    statistics: ActionStatistics, frameskip: int
) -> ZScoreNormalizer:
    """Repeat raw action statistics in frameskip flattening order."""
    if frameskip <= 0:
        raise ValueError("frameskip must be positive")
    return ZScoreNormalizer(
        statistics.mean.repeat(1, frameskip),
        statistics.std.repeat(1, frameskip),
    )


def load_or_create_split_manifest(
    path: str | Path,
    *,
    dataset: str,
    episode_ids: tuple[str, ...],
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> SplitManifest:
    """Load a matching manifest or create and persist a new one."""
    manifest_path = Path(path)
    if manifest_path.exists():
        manifest = load_split_manifest(manifest_path)
        manifest.validate(dataset, seed, episode_ids)
        return manifest
    manifest = create_split_manifest(
        dataset,
        episode_ids,
        seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_split_manifest(manifest, manifest_path)
    return manifest


def split_manifest_digest(manifest: SplitManifest) -> str:
    """Return a stable SHA-256 digest of canonical manifest content."""
    canonical = json.dumps(
        manifest.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def prepare_screening_data(
    data_path: str | Path,
    *,
    dataset: str,
    manifest_path: str | Path,
    split_seed: int,
    frameskip: int = 5,
    training_fraction: float = 1.0,
) -> ScreeningData:
    """Open PushT sources and prepare split/statistics without decoding pixels."""
    dataset_source, dataset_fingerprint = dataset_identity(data_path)
    _validate_or_write_dataset_identity(
        Path(manifest_path), dataset, dataset_source, dataset_fingerprint
    )
    source = open_pusht_lance_source(data_path, frameskip, keys_to_load=("pixels", "action"))
    action_source = open_pusht_lance_source(data_path, frameskip, keys_to_load=("action",))
    return prepare_screening_data_from_sources(
        source,
        action_source,
        dataset=dataset,
        manifest_path=manifest_path,
        split_seed=split_seed,
        frameskip=frameskip,
        training_fraction=training_fraction,
        dataset_source=dataset_source,
        dataset_fingerprint=dataset_fingerprint,
    )


def prepare_screening_data_from_sources(
    source: EpisodeSource,
    action_source: EpisodeSource,
    *,
    dataset: str,
    manifest_path: str | Path,
    split_seed: int,
    frameskip: int,
    training_fraction: float = 1.0,
    dataset_source: str = "injected-source",
    dataset_fingerprint: str = "injected-source",
) -> ScreeningData:
    """Prepare screening metadata from lazy pixel and action-only sources."""
    if frameskip <= 0:
        raise ValueError("frameskip must be positive")
    episode_ids = _source_episode_ids(source)
    if len(action_source.lengths) != len(episode_ids):
        raise ValueError("pixel and action sources must contain the same episodes")
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = load_or_create_split_manifest(
        resolved_manifest_path,
        dataset=dataset,
        episode_ids=episode_ids,
        seed=split_seed,
    )
    train_episode_ids = sample_episode_ids(manifest.train, training_fraction, manifest.seed)
    statistics = stream_action_statistics(action_source)
    return ScreeningData(
        source=source,
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        manifest_digest=split_manifest_digest(manifest),
        train_episode_ids=train_episode_ids,
        action_statistics=statistics,
        frameskip=frameskip,
        dataset_source=dataset_source,
        dataset_fingerprint=dataset_fingerprint,
    )


def materialize_training_windows(data: ScreeningData, window_size: int) -> ObservationWindowDataset:
    """Load selected training episodes and materialize their fixed windows."""
    trajectories = tuple(
        adapt_pusht_episode(
            episode_id,
            data.source.load_episode(episode_index(episode_id)),
            data.frameskip,
        )
        for episode_id in data.train_episode_ids
    )
    return ObservationWindowDataset(trajectories, window_size)


def load_source_trajectory(
    source: EpisodeSource,
    episode_index_value: int,
    frameskip: int,
) -> ObservationTrajectory:
    """Materialize one stable-ID trajectory from a lazy source."""
    if episode_index_value < 0 or episode_index_value >= len(source.lengths):
        raise IndexError("episode index is outside the source")
    episode_id = f"episode-{episode_index_value:06d}"
    return adapt_pusht_episode(
        episode_id,
        source.load_episode(episode_index_value),
        frameskip,
    )


def _action_rows(episode: Mapping[str, Tensor]) -> Tensor:
    actions = episode.get("action")
    if not isinstance(actions, Tensor):
        raise ValueError("episode must contain a tensor 'action' column")
    if actions.ndim < 2:
        raise ValueError("action must have shape (raw_timesteps, action_dim)")
    if not actions.is_floating_point():
        raise ValueError("action rows must use a floating-point dtype")
    return actions.reshape(actions.shape[0], -1)


def _source_episode_ids(source: EpisodeSource) -> tuple[str, ...]:
    return tuple(f"episode-{index:06d}" for index in range(len(source.lengths)))


def episode_index(episode_id: str) -> int:
    """Parse a stable screen episode identifier."""
    prefix = "episode-"
    if not episode_id.startswith(prefix):
        raise ValueError(f"invalid episode ID: {episode_id}")
    suffix = episode_id.removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError(f"invalid episode ID: {episode_id}")
    return int(suffix)


def dataset_identity(data_path: str | Path) -> tuple[str, str]:
    """Return a source description and metadata fingerprint without hashing dataset payloads."""
    path = Path(data_path)
    if not path.exists():
        source = str(data_path)
        return source, hashlib.sha256(source.encode("utf-8")).hexdigest()
    resolved = path.resolve()
    entries = (
        [(resolved.name, resolved.stat().st_size)]
        if resolved.is_file()
        else [
            (str(item.relative_to(resolved)), item.stat().st_size)
            for item in sorted(resolved.rglob("*"))
            if item.is_file()
        ]
    )
    canonical = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return str(resolved), hashlib.sha256(canonical).hexdigest()


def _validate_or_write_dataset_identity(
    manifest_path: Path,
    dataset: str,
    source: str,
    fingerprint: str,
) -> None:
    identity_path = manifest_path.with_name(f"{manifest_path.stem}_dataset.json")
    identity = {"dataset": dataset, "source": source, "fingerprint": fingerprint}
    if identity_path.exists():
        with identity_path.open(encoding="utf-8") as file:
            existing = json.load(file)
        if not isinstance(existing, dict) or existing.get("dataset") != dataset:
            raise ValueError("dataset identity does not match the existing split manifest")
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("dataset identity does not match the existing split manifest")
        return
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    with identity_path.open("w", encoding="utf-8") as file:
        json.dump(identity, file, indent=2, sort_keys=True)
        file.write("\n")
