"""Human-readable nearest-frame diagnostics for predicted latent rollouts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from lewm_liquid_predictors.data import (
    EpisodeSource,
    ObservationTrajectory,
    episode_index,
    load_source_trajectory,
)


def write_retrieval_galleries(
    run_dir: Path,
    queries: Sequence[tuple[ObservationTrajectory, Tensor]],
    reference_latents: Tensor,
    references: Sequence[tuple[int, int]],
    source: EpisodeSource,
    frameskip: int,
    horizons: tuple[int, ...],
) -> None:
    """Write leave-one-episode-out nearest-frame galleries and metadata."""
    import imageio.v3 as imageio

    if reference_latents.shape[0] != len(references):
        raise ValueError("retrieval latents and references must align")
    records: list[dict[str, object]] = []
    episode_cache: dict[int, ObservationTrajectory] = {}
    for trajectory, rollout in queries:
        query_episode_index = episode_index(trajectory.episode_id)
        eligible = torch.tensor(
            [reference_index != query_episode_index for reference_index, _ in references],
            dtype=torch.bool,
        )
        if not eligible.any():
            raise ValueError("retrieval requires a reference episode distinct from the query")
        input_frame = image_array(trajectory.observations[0])
        rows: list[tuple[int, NDArray[np.uint8], NDArray[np.uint8]]] = []
        for horizon in horizons:
            if horizon > rollout.shape[0]:
                continue
            predicted = rollout[horizon - 1]
            distances = (reference_latents - predicted).square().mean(dim=1)
            distances = torch.where(eligible, distances, torch.full_like(distances, torch.inf))
            index = int(distances.argmin().item())
            reference_index, reference_timestep = references[index]
            reference = episode_cache.get(reference_index)
            if reference is None:
                reference = load_source_trajectory(source, reference_index, frameskip)
                episode_cache[reference_index] = reference
            actual = image_array(trajectory.observations[horizon])
            retrieved = image_array(reference.observations[reference_timestep])
            rows.append((horizon, retrieved, actual))
            records.append(
                {
                    "query_episode_id": trajectory.episode_id,
                    "input_timestep": 0,
                    "horizon": horizon,
                    "actual_timestep": horizon,
                    "retrieved_episode_id": reference.episode_id,
                    "retrieved_timestep": reference_timestep,
                    "mean_squared_latent_distance": float(distances[index].item()),
                }
            )
        if rows:
            imageio.imwrite(
                run_dir / f"retrieval_{trajectory.episode_id}.png",
                build_retrieval_gallery(input_frame, rows),
            )
    (run_dir / "retrieval.json").write_text(
        json.dumps(
            {
                "description": "Each row shows the initial input frame, the nearest screen-test "
                "frame latent from a different episode, and the actual future frame. "
                "Retrieved frames are a proxy, not generated images.",
                "gallery_columns": [
                    "Initial input (t=0)",
                    "Predicted retrieval (t+h)",
                    "Actual future (t+h)",
                ],
                "reference_policy": {
                    "corpus": "evaluated_screen_test_episodes",
                    "exclude_query_episode": True,
                    "episode_count": len({episode for episode, _ in references}),
                },
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_retrieval_gallery(
    input_frame: NDArray[np.uint8],
    rows: list[tuple[int, NDArray[np.uint8], NDArray[np.uint8]]],
) -> NDArray[np.uint8]:
    """Build a labeled input/prediction/actual retrieval gallery."""
    import cv2

    if not rows:
        raise ValueError("retrieval gallery requires at least one row")
    _, width, channels = input_frame.shape
    if channels != 3:
        raise ValueError("retrieval gallery images must be RGB")
    expected_shape = input_frame.shape
    if any(
        image.shape != expected_shape
        for _, predicted, actual in rows
        for image in (predicted, actual)
    ):
        raise ValueError("retrieval gallery images must share a shape")

    columns = ("Initial input (t=0)", "Predicted retrieval (t+h)", "Actual future (t+h)")
    header = np.concatenate(tuple(_text_banner(width, 30, title, cv2) for title in columns), axis=1)
    gallery_rows: list[NDArray[np.uint8]] = [header]
    for horizon, predicted, actual in rows:
        gallery_rows.append(_text_banner(width * 3, 24, f"Rollout horizon: {horizon}", cv2))
        gallery_rows.append(np.concatenate((input_frame, predicted, actual), axis=1))
    return np.concatenate(gallery_rows, axis=0)


def image_array(observation: Tensor) -> NDArray[np.uint8]:
    """Convert a source RGB observation to an HWC uint8 image."""
    image = _channels_first(observation.unsqueeze(0)).squeeze(0)
    if image.dtype != torch.uint8:
        image = (image.clamp(0, 1) * 255).to(torch.uint8)
    return np.asarray(image.permute(1, 2, 0).cpu(), dtype=np.uint8)


def _channels_first(observations: Tensor) -> Tensor:
    if observations.shape[-3] == 3:
        return observations
    if observations.shape[-1] == 3:
        return observations.movedim(-1, -3)
    raise ValueError("pixels must be RGB in CHW or HWC layout")


def _text_banner(width: int, height: int, text: str, cv2: object) -> NDArray[np.uint8]:
    banner = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX  # type: ignore[attr-defined]
    font_scale = 0.42
    thickness = 1
    while True:
        (text_width, text_height), _ = cv2.getTextSize(  # type: ignore[attr-defined]
            text, font, font_scale, thickness
        )
        if text_width <= width - 6 or font_scale <= 0.2:
            break
        font_scale -= 0.02
    origin = (max(3, (width - text_width) // 2), (height + text_height) // 2)
    cv2.putText(  # type: ignore[attr-defined]
        banner,
        text,
        origin,
        font,
        font_scale,
        (35, 35, 35),
        thickness,
        cv2.LINE_AA,  # type: ignore[attr-defined]
    )
    return banner
