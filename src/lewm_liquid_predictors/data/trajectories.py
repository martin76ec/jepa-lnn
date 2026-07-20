"""Episode-preserving trajectory containers and batching utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Trajectory:
    """One complete episode of latent states and actions.

    ``latents`` contains the initial state and one target state per action, so its
    length is exactly one greater than ``actions`` along the time dimension.
    """

    episode_id: str
    latents: Tensor
    actions: Tensor

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.latents.ndim != 2:
            raise ValueError("latents must have shape (timesteps + 1, latent_dim)")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape (timesteps, action_dim)")
        if self.latents.shape[0] != self.actions.shape[0] + 1:
            raise ValueError("latents must have exactly one more timestep than actions")
        if self.actions.shape[0] == 0:
            raise ValueError("trajectories must contain at least one transition")
        if self.latents.device != self.actions.device:
            raise ValueError("latents and actions must be on the same device")

    @property
    def num_transitions(self) -> int:
        """Return the number of action-conditioned transitions in the episode."""
        return self.actions.shape[0]


@dataclass(frozen=True)
class TrajectoryBatch:
    """Padded variable-length batch retaining an explicit valid-transition mask."""

    episode_ids: tuple[str, ...]
    latents: Tensor
    actions: Tensor
    transition_mask: Tensor

    def __post_init__(self) -> None:
        batch_size, max_transitions = self.transition_mask.shape
        if self.latents.shape[:2] != (batch_size, max_transitions + 1):
            raise ValueError("latents are incompatible with transition_mask")
        if self.actions.shape[:2] != (batch_size, max_transitions):
            raise ValueError("actions are incompatible with transition_mask")
        if len(self.episode_ids) != batch_size:
            raise ValueError("episode_ids are incompatible with transition_mask")
        if self.transition_mask.dtype != torch.bool:
            raise ValueError("transition_mask must be boolean")


class TrajectoryDataset(Dataset[Trajectory]):
    """Dataset of full episodes that never creates samples across episode boundaries."""

    def __init__(self, trajectories: Sequence[Trajectory]) -> None:
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        episode_ids = tuple(trajectory.episode_id for trajectory in trajectories)
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("episode_ids must be unique")
        self._trajectories = tuple(trajectories)

    def __len__(self) -> int:
        """Return the number of complete episodes."""
        return len(self._trajectories)

    def __getitem__(self, index: int) -> Trajectory:
        """Return one complete episode."""
        return self._trajectories[index]


def collate_trajectories(trajectories: Sequence[Trajectory]) -> TrajectoryBatch:
    """Pad complete episodes while masking invalid trailing transitions."""
    if not trajectories:
        raise ValueError("trajectories must not be empty")
    _validate_batch_shapes(trajectories)
    device = trajectories[0].latents.device
    max_transitions = max(trajectory.num_transitions for trajectory in trajectories)
    batch_size = len(trajectories)
    latent_dim = trajectories[0].latents.shape[-1]
    action_dim = trajectories[0].actions.shape[-1]
    latents = torch.zeros(
        (batch_size, max_transitions + 1, latent_dim),
        dtype=trajectories[0].latents.dtype,
        device=device,
    )
    actions = torch.zeros(
        (batch_size, max_transitions, action_dim),
        dtype=trajectories[0].actions.dtype,
        device=device,
    )
    transition_mask = torch.zeros((batch_size, max_transitions), dtype=torch.bool, device=device)
    for index, trajectory in enumerate(trajectories):
        transitions = trajectory.num_transitions
        latents[index, : transitions + 1] = trajectory.latents
        actions[index, :transitions] = trajectory.actions
        transition_mask[index, :transitions] = True
    return TrajectoryBatch(
        episode_ids=tuple(trajectory.episode_id for trajectory in trajectories),
        latents=latents,
        actions=actions,
        transition_mask=transition_mask,
    )


def _validate_batch_shapes(trajectories: Sequence[Trajectory]) -> None:
    first = trajectories[0]
    for trajectory in trajectories[1:]:
        if trajectory.latents.shape[-1] != first.latents.shape[-1]:
            raise ValueError("all trajectories must have the same latent dimension")
        if trajectory.actions.shape[-1] != first.actions.shape[-1]:
            raise ValueError("all trajectories must have the same action dimension")
        if trajectory.latents.device != first.latents.device:
            raise ValueError("all trajectories must be on the same device")
        if trajectory.actions.device != first.actions.device:
            raise ValueError("all trajectories must be on the same device")
