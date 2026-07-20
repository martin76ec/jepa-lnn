"""Tests for adapting upstream PushT episodes."""

from collections.abc import Mapping, Sized

import pytest
import torch
from torch import Tensor

from lewm_liquid_predictors.data import (
    ObservationTrajectory,
    adapt_pusht_episode,
    adapt_pusht_episodes,
    collate_observation_trajectories,
)


class _Source:
    def __init__(self, episodes: tuple[Mapping[str, Tensor], ...]) -> None:
        self._episodes = episodes

    @property
    def lengths(self) -> Sized:
        return self._episodes

    def load_episode(self, episode_idx: int) -> Mapping[str, Tensor]:
        return self._episodes[episode_idx]


def test_adapter_flattens_raw_action_blocks_and_drops_terminal_action_block() -> None:
    episode = {
        "pixels": torch.arange(5 * 2 * 2 * 3, dtype=torch.uint8).reshape(5, 2, 2, 3),
        "action": torch.arange(10, dtype=torch.float32).reshape(10, 1),
    }

    trajectory = adapt_pusht_episode("episode-0", episode, frameskip=2)

    assert trajectory.observations.shape == (5, 2, 2, 3)
    assert trajectory.actions.shape == (4, 2)
    assert torch.equal(
        trajectory.actions,
        torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
    )


def test_adapter_preserves_source_episode_boundaries() -> None:
    episodes = tuple(
        {
            "pixels": torch.zeros(3, 2, 2, 3, dtype=torch.uint8),
            "action": torch.zeros(6, 2),
        }
        for _ in range(2)
    )

    trajectories = adapt_pusht_episodes(_Source(episodes), frameskip=2)

    assert tuple(trajectory.episode_id for trajectory in trajectories) == (
        "episode-000000",
        "episode-000001",
    )
    assert all(trajectory.actions.shape == (2, 4) for trajectory in trajectories)


def test_adapter_can_limit_the_number_of_source_episodes() -> None:
    episodes = tuple(
        {
            "pixels": torch.zeros(3, 2, 2, 3, dtype=torch.uint8),
            "action": torch.zeros(6, 2),
        }
        for _ in range(2)
    )

    trajectories = adapt_pusht_episodes(_Source(episodes), frameskip=2, max_episodes=1)

    assert len(trajectories) == 1
    assert trajectories[0].episode_id == "episode-000000"


def test_observation_collator_masks_variable_length_pixel_episodes() -> None:
    first = ObservationTrajectory(
        "first",
        observations=torch.ones(3, 3, 2, 2),
        actions=torch.ones(2, 10),
    )
    second = ObservationTrajectory(
        "second",
        observations=torch.ones(2, 3, 2, 2),
        actions=torch.ones(1, 10),
    )

    batch = collate_observation_trajectories([first, second])

    assert batch.observations.shape == (2, 3, 3, 2, 2)
    assert batch.actions.shape == (2, 2, 10)
    assert torch.equal(batch.transition_mask, torch.tensor([[True, True], [True, False]]))
    assert torch.equal(batch.observations[1, 2], torch.zeros(3, 2, 2))


def test_adapter_rejects_episodes_without_two_aligned_observations() -> None:
    episode = {
        "pixels": torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
        "action": torch.zeros(2, 1),
    }

    with pytest.raises(ValueError, match="fewer than two"):
        adapt_pusht_episode("episode-0", episode, frameskip=2)
