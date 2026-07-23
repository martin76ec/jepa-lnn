"""Streaming evaluation over the controlled screen split."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from lewm_liquid_predictors.data import (
    EpisodeSource,
    ObservationTrajectory,
    ObservationTrajectoryBatch,
    ZScoreNormalizer,
    collate_observation_trajectories,
    load_source_trajectory,
    prepare_observation_batch,
)

from .rollout import RolloutMetricAccumulator


@dataclass(frozen=True)
class EpisodePredictions:
    """Observed latents and aligned teacher-forced/closed-loop predictions."""

    latents: Tensor
    teacher_forced: Tensor
    closed_loop: Tensor

    def __post_init__(self) -> None:
        if self.latents.ndim != 3 or self.latents.shape[1] < 2:
            raise ValueError("latents must have shape (batch, transitions + 1, latent_dim)")
        expected = (self.latents.shape[0], self.latents.shape[1] - 1, self.latents.shape[2])
        if self.teacher_forced.shape != expected or self.closed_loop.shape != expected:
            raise ValueError("predictions must align with every latent transition")
        if (
            self.teacher_forced.device != self.latents.device
            or self.closed_loop.device != self.latents.device
        ):
            raise ValueError("latents and predictions must be on the same device")


@dataclass(frozen=True)
class HeldOutEvaluation:
    """Scalar metrics plus optional inputs for retrieval rendering."""

    metrics: dict[str, float]
    retrieval_latents: Tensor
    retrieval_references: tuple[tuple[int, int], ...]
    gallery_queries: tuple[tuple[ObservationTrajectory, Tensor], ...]


@torch.no_grad()
def evaluate_screen_split(
    source: EpisodeSource,
    test_indices: tuple[int, ...],
    action_normalizer: ZScoreNormalizer,
    *,
    frameskip: int,
    horizons: tuple[int, ...],
    divergence_threshold: float,
    count_non_finite_as_divergence: bool,
    device: torch.device,
    predict_episode: Callable[[ObservationTrajectoryBatch], EpisodePredictions],
    gallery_episode_count: int = 3,
) -> HeldOutEvaluation:
    """Stream screen-test episodes through one predictor and aggregate metrics."""
    if not test_indices:
        raise ValueError("test_indices must not be empty")
    accumulator = RolloutMetricAccumulator(
        horizons,
        divergence_threshold,
        count_non_finite_as_divergence=count_non_finite_as_divergence,
    )
    retrieval_latents: list[Tensor] = []
    retrieval_references: list[tuple[int, int]] = []
    gallery_queries: list[tuple[ObservationTrajectory, Tensor]] = []
    for test_episode_index in test_indices:
        trajectory = load_source_trajectory(source, test_episode_index, frameskip)
        prepared = prepare_observation_batch(
            collate_observation_trajectories([trajectory]), action_normalizer, device
        )
        predictions = predict_episode(prepared)
        targets = predictions.latents[:, 1:]
        if prepared.transition_mask.shape != targets.shape[:2]:
            raise ValueError("episode predictions do not align with the transition mask")
        accumulator.update(
            predictions.teacher_forced,
            predictions.closed_loop,
            targets,
            prepared.transition_mask,
        )
        retrieval_latents.append(predictions.latents[0].detach().cpu())
        retrieval_references.extend(
            (test_episode_index, timestep) for timestep in range(predictions.latents.shape[1])
        )
        if len(gallery_queries) < gallery_episode_count:
            gallery_queries.append((trajectory, predictions.closed_loop[0].detach().cpu()))

    aggregate = accumulator.compute()
    metrics = {
        "test/one_step_normalized_mse": aggregate.one_step_normalized_mse.item(),
        "test/one_step_normalized_rmse": aggregate.one_step_normalized_rmse.item(),
        "test/divergence_rate": aggregate.divergence_rate.item(),
    }
    if aggregate.median_first_divergence_time is not None:
        metrics["test/median_first_divergence_time"] = aggregate.median_first_divergence_time.item()
    metrics.update(
        {
            f"test/rollout_normalized_mse/{horizon}": error.item()
            for horizon, error in aggregate.rollout_normalized_mse.items()
        }
    )
    metrics.update(
        {
            f"test/rollout_normalized_rmse/{horizon}": error.item()
            for horizon, error in aggregate.rollout_normalized_rmse.items()
        }
    )
    metrics.update(
        {
            f"test/rollout_non_finite_rate/{horizon}": rate.item()
            for horizon, rate in aggregate.rollout_non_finite_rate.items()
        }
    )
    return HeldOutEvaluation(
        metrics=metrics,
        retrieval_latents=torch.cat(retrieval_latents),
        retrieval_references=tuple(retrieval_references),
        gallery_queries=tuple(gallery_queries),
    )
