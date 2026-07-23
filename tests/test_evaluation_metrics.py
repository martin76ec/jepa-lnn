"""Tests for streaming rollout metric aggregation."""

import pytest
import torch

from lewm_liquid_predictors.evaluation import (
    RolloutMetricAccumulator,
    divergence_times,
    normalized_mse,
)


def test_normalized_mse_averages_trajectory_ratios_not_global_energy() -> None:
    targets = torch.tensor([[[1.0]], [[10.0]]])
    predictions = torch.tensor([[[0.0]], [[10.0]]])
    mask = torch.ones(2, 1, dtype=torch.bool)

    value = normalized_mse(predictions, targets, mask)

    assert value.item() == pytest.approx(0.5)
    assert value.item() != pytest.approx(1.0 / 101.0)


def test_streaming_matches_batch_aggregation_for_unequal_lengths() -> None:
    targets = torch.tensor([[[1.0], [float("nan")], [float("nan")]], [[1.0], [1.0], [1.0]]])
    teacher_forced = torch.tensor([[[0.0], [float("nan")], [float("nan")]], [[1.0], [1.0], [1.0]]])
    rollout = teacher_forced.clone()
    mask = torch.tensor([[True, False, False], [True, True, True]])
    batch_accumulator = RolloutMetricAccumulator((1, 3), 2.0)
    streaming_accumulator = RolloutMetricAccumulator((1, 3), 2.0)

    batch_accumulator.update(teacher_forced, rollout, targets, mask)
    for index, length in ((0, 1), (1, 3)):
        streaming_accumulator.update(
            teacher_forced[index : index + 1, :length],
            rollout[index : index + 1, :length],
            targets[index : index + 1, :length],
            mask[index : index + 1, :length],
        )

    batch_metrics = batch_accumulator.compute()
    streaming_metrics = streaming_accumulator.compute()
    assert batch_metrics.one_step_normalized_mse.item() == pytest.approx(0.5)
    assert streaming_metrics.one_step_normalized_mse.item() == pytest.approx(0.5)
    assert streaming_metrics.rollout_normalized_mse.keys() == {1, 3}
    assert streaming_metrics.rollout_normalized_mse[1].item() == pytest.approx(0.5)
    assert streaming_metrics.rollout_normalized_mse[3].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("count_non_finite", "expected_time", "expected_rate"),
    [(True, 1, 1.0), (False, -1, 0.0)],
)
def test_non_finite_divergence_policy(
    count_non_finite: bool, expected_time: int, expected_rate: float
) -> None:
    targets = torch.ones(1, 1, 1)
    teacher_forced = targets.clone()
    rollout = torch.full_like(targets, float("nan"))
    mask = torch.ones(1, 1, dtype=torch.bool)
    accumulator = RolloutMetricAccumulator(
        (2,),
        1.0,
        count_non_finite_as_divergence=count_non_finite,
    )

    accumulator.update(teacher_forced, rollout, targets, mask)
    metrics = accumulator.compute()

    assert metrics.divergence_time.item() == expected_time
    assert metrics.divergence_rate.item() == expected_rate
    assert (metrics.median_first_divergence_time is not None) is count_non_finite
    assert metrics.rollout_normalized_mse == {}


def test_divergence_and_metrics_ignore_nan_in_padded_positions() -> None:
    predictions = torch.tensor([[[1.0], [float("nan")]]])
    targets = torch.tensor([[[1.0], [float("nan")]]])
    mask = torch.tensor([[True, False]])

    assert normalized_mse(predictions, targets, mask).item() == 0.0
    assert divergence_times(predictions, targets, mask, 0.5).item() == -1


@pytest.mark.parametrize("invalid_input", ["teacher", "target"])
def test_accumulator_rejects_non_finite_valid_metric_values(invalid_input: str) -> None:
    targets = torch.ones(1, 1, 1)
    teacher_forced = targets.clone()
    rollout = targets.clone()
    if invalid_input == "teacher":
        teacher_forced[0, 0, 0] = float("nan")
    else:
        targets[0, 0, 0] = float("nan")
    accumulator = RolloutMetricAccumulator((1,), 1.0)

    with pytest.raises(ValueError, match="non-finite values at valid metric positions"):
        accumulator.update(
            teacher_forced,
            rollout,
            targets,
            torch.ones(1, 1, dtype=torch.bool),
        )


def test_non_finite_endpoint_is_penalized_and_recorded_instead_of_aborting() -> None:
    targets = torch.ones(1, 1, 1)
    rollout = torch.full_like(targets, float("nan"))
    accumulator = RolloutMetricAccumulator((1,), 2.0)

    accumulator.update(targets, rollout, targets, torch.ones(1, 1, dtype=torch.bool))
    metrics = accumulator.compute()

    assert metrics.rollout_normalized_mse[1].item() == pytest.approx(4.0)
    assert metrics.rollout_non_finite_rate[1].item() == pytest.approx(1.0)
    assert metrics.divergence_rate.item() == pytest.approx(1.0)


def test_unavailable_horizons_are_omitted_until_observed() -> None:
    accumulator = RolloutMetricAccumulator((2, 4), 2.0)
    targets = torch.ones(1, 2, 1)
    predictions = targets.clone()

    accumulator.update(
        predictions,
        predictions,
        targets,
        torch.ones(1, 2, dtype=torch.bool),
    )
    metrics = accumulator.compute()

    assert metrics.rollout_normalized_mse.keys() == {2}


def test_divergence_aggregates_rate_and_median_first_time() -> None:
    accumulator = RolloutMetricAccumulator((3,), 0.5)
    targets = torch.ones(3, 3, 1)
    teacher_forced = targets.clone()
    rollout = torch.tensor(
        [
            [[0.0], [1.0], [1.0]],
            [[1.0], [1.0], [0.0]],
            [[1.0], [1.0], [1.0]],
        ]
    )

    accumulator.update(
        teacher_forced,
        rollout,
        targets,
        torch.ones(3, 3, dtype=torch.bool),
    )
    metrics = accumulator.compute()

    assert metrics.divergence_rate.item() == pytest.approx(2.0 / 3.0)
    assert metrics.median_first_divergence_time is not None
    assert metrics.median_first_divergence_time.item() == 2.0
