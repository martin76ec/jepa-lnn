"""PushT episode adapter for the upstream stable-worldmodel Lance dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor, cat


class EpisodeSource(Protocol):
    """The episode-oriented subset of the stable-worldmodel dataset API."""

    @property
    def lengths(self) -> Sized:
        """Return one entry for each source episode."""

    def load_episode(self, episode_idx: int) -> Mapping[str, Tensor]:
        """Return all requested columns from one complete source episode."""


@dataclass(frozen=True)
class ObservationTrajectory:
    """Pixel observations and action blocks ready for a shared encoder.

    Every action block drives the transition from one observation to the
    next. The final raw action block has no subsequent observation and is
    intentionally excluded.
    """

    episode_id: str
    observations: Tensor
    actions: Tensor

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.observations.ndim < 2:
            raise ValueError("observations must have shape (timesteps, ...)")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape (timesteps - 1, action_dim)")
        if self.observations.shape[0] != self.actions.shape[0] + 1:
            raise ValueError("observations must have exactly one more timestep than actions")
        if self.actions.shape[0] == 0:
            raise ValueError("trajectories must contain at least one transition")
        if self.observations.device != self.actions.device:
            raise ValueError("observations and actions must be on the same device")


@dataclass(frozen=True)
class ObservationTrajectoryBatch:
    """Padded pixel trajectories with an explicit valid-transition mask."""

    episode_ids: tuple[str, ...]
    observations: Tensor
    actions: Tensor
    transition_mask: Tensor

    def __post_init__(self) -> None:
        batch_size, max_transitions = self.transition_mask.shape
        if self.observations.shape[:2] != (batch_size, max_transitions + 1):
            raise ValueError("observations are incompatible with transition_mask")
        if self.actions.shape[:2] != (batch_size, max_transitions):
            raise ValueError("actions are incompatible with transition_mask")
        if len(self.episode_ids) != batch_size:
            raise ValueError("episode_ids are incompatible with transition_mask")
        if self.transition_mask.dtype != torch.bool:
            raise ValueError("transition_mask must be boolean")


def load_pusht_lance_episodes(
    path: str | Path,
    frameskip: int = 5,
    max_episodes: int | None = None,
) -> tuple[ObservationTrajectory, ...]:
    """Load complete upstream PushT episodes from a Lance dataset.

    Install the optional ``upstream`` dependency group before calling this
    function. The reader requests raw action rows because LeWM concatenates
    ``frameskip`` consecutive actions for each downsampled visual step.
    """
    if frameskip <= 0:
        raise ValueError("frameskip must be positive")
    try:
        stable_worldmodel: Any = import_module("stable_worldmodel")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "install the upstream dependency group to load PushT Lance data"
        ) from error
    dataset_path = str(path)
    if "://" not in dataset_path:
        dataset_path = str(Path(dataset_path).resolve())
    source = cast(
        EpisodeSource,
        stable_worldmodel.data.load_dataset(
            dataset_path,
            frameskip=frameskip,
            num_steps=1,
            keys_to_load=["pixels", "action"],
        ),
    )
    return adapt_pusht_episodes(source, frameskip, max_episodes=max_episodes)


def adapt_pusht_episodes(
    source: EpisodeSource,
    frameskip: int,
    max_episodes: int | None = None,
) -> tuple[ObservationTrajectory, ...]:
    """Convert an upstream episode source without crossing episode boundaries."""
    if frameskip <= 0:
        raise ValueError("frameskip must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive when provided")
    episode_count = len(source.lengths)
    if max_episodes is not None:
        episode_count = min(episode_count, max_episodes)
    return tuple(
        adapt_pusht_episode(f"episode-{index:06d}", source.load_episode(index), frameskip)
        for index in range(episode_count)
    )


def adapt_pusht_episode(
    episode_id: str,
    episode: Mapping[str, Tensor],
    frameskip: int,
) -> ObservationTrajectory:
    """Align downsampled pixels with flattened action blocks from one episode."""
    if frameskip <= 0:
        raise ValueError("frameskip must be positive")
    observations = _tensor_column(episode, "pixels")
    raw_actions = _tensor_column(episode, "action")
    if raw_actions.ndim < 2:
        raise ValueError("action must have shape (raw_timesteps, action_dim)")
    action_blocks = _action_blocks(raw_actions, frameskip)
    usable_observations = min(observations.shape[0], action_blocks.shape[0])
    if usable_observations < 2:
        raise ValueError("episode has fewer than two aligned observations")
    return ObservationTrajectory(
        episode_id=episode_id,
        observations=observations[:usable_observations],
        actions=action_blocks[: usable_observations - 1],
    )


def collate_observation_trajectories(
    trajectories: Sequence[ObservationTrajectory],
) -> ObservationTrajectoryBatch:
    """Pad complete pixel episodes without introducing cross-episode transitions."""
    if not trajectories:
        raise ValueError("trajectories must not be empty")
    _validate_observation_batch_shapes(trajectories)
    first = trajectories[0]
    max_transitions = max(trajectory.actions.shape[0] for trajectory in trajectories)
    batch_size = len(trajectories)
    observations = torch.zeros(
        (batch_size, max_transitions + 1, *first.observations.shape[1:]),
        dtype=first.observations.dtype,
        device=first.observations.device,
    )
    actions = torch.zeros(
        (batch_size, max_transitions, first.actions.shape[-1]),
        dtype=first.actions.dtype,
        device=first.actions.device,
    )
    transition_mask = torch.zeros(
        (batch_size, max_transitions), dtype=torch.bool, device=first.observations.device
    )
    for index, trajectory in enumerate(trajectories):
        transitions = trajectory.actions.shape[0]
        observations[index, : transitions + 1] = trajectory.observations
        actions[index, :transitions] = trajectory.actions
        transition_mask[index, :transitions] = True
    return ObservationTrajectoryBatch(
        episode_ids=tuple(trajectory.episode_id for trajectory in trajectories),
        observations=observations,
        actions=actions,
        transition_mask=transition_mask,
    )


def _tensor_column(episode: Mapping[str, Tensor], key: str) -> Tensor:
    value = episode.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"episode must contain a tensor {key!r} column")
    return value


def _action_blocks(raw_actions: Tensor, frameskip: int) -> Tensor:
    complete_steps = raw_actions.shape[0] // frameskip
    if complete_steps == 0:
        raise ValueError("action has no complete frameskip block")
    trimmed_actions = raw_actions[: complete_steps * frameskip]
    blocks = trimmed_actions.reshape(complete_steps, frameskip, -1)
    return cat(tuple(blocks.unbind(dim=1)), dim=-1)


def _validate_observation_batch_shapes(trajectories: Sequence[ObservationTrajectory]) -> None:
    first = trajectories[0]
    for trajectory in trajectories[1:]:
        if trajectory.observations.shape[1:] != first.observations.shape[1:]:
            raise ValueError("all trajectories must have the same observation shape")
        if trajectory.actions.shape[-1] != first.actions.shape[-1]:
            raise ValueError("all trajectories must have the same action dimension")
        if trajectory.observations.device != first.observations.device:
            raise ValueError("all trajectories must be on the same device")
