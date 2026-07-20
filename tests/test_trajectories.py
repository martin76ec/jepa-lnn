"""Tests for episode-preserving data structures and padded batches."""

import pytest
import torch

from lewm_liquid_predictors.data import Trajectory, TrajectoryDataset, collate_trajectories


def _trajectory(episode_id: str, transitions: int) -> Trajectory:
    latents = torch.arange((transitions + 1) * 3, dtype=torch.float32).reshape(transitions + 1, 3)
    actions = torch.arange(transitions * 2, dtype=torch.float32).reshape(transitions, 2)
    return Trajectory(episode_id=episode_id, latents=latents, actions=actions)


def test_dataset_preserves_complete_episode_boundaries() -> None:
    first = _trajectory("first", 2)
    second = _trajectory("second", 4)
    dataset = TrajectoryDataset([first, second])

    assert len(dataset) == 2
    assert dataset[0] is first
    assert dataset[1] is second


def test_collate_pads_variable_lengths_and_masks_trailing_transitions() -> None:
    first = _trajectory("first", 2)
    second = _trajectory("second", 4)

    batch = collate_trajectories([first, second])

    assert batch.episode_ids == ("first", "second")
    assert batch.latents.shape == (2, 5, 3)
    assert batch.actions.shape == (2, 4, 2)
    assert torch.equal(
        batch.transition_mask,
        torch.tensor([[True, True, False, False], [True, True, True, True]]),
    )
    assert torch.equal(batch.latents[0, :3], first.latents)
    assert torch.equal(batch.actions[0, :2], first.actions)
    assert torch.equal(batch.latents[0, 3:], torch.zeros(2, 3))
    assert torch.equal(batch.actions[0, 2:], torch.zeros(2, 2))


def test_trajectory_rejects_non_transition_aligned_tensors() -> None:
    with pytest.raises(ValueError, match="one more timestep"):
        Trajectory("invalid", torch.zeros(3, 2), torch.zeros(3, 1))


def test_dataset_rejects_duplicate_episode_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        TrajectoryDataset([_trajectory("duplicate", 1), _trajectory("duplicate", 2)])
