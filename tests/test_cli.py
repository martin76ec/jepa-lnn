"""Tests for executable smoke-test commands."""

from pathlib import Path

import torch

from lewm_liquid_predictors import cli
from lewm_liquid_predictors.data import ObservationTrajectory

ROOT = Path(__file__).parents[1]


def test_validate_config_command_prints_normalized_config(capsys: object) -> None:
    assert cli.main(["validate-config", str(ROOT / "configs" / "local.yaml")]) == 0

    output = capsys.readouterr().out
    assert '"frameskip": 5' in output
    assert '"seeds": [' in output


def test_inspect_pusht_command_prints_trajectory_shapes(
    monkeypatch: object, capsys: object
) -> None:
    trajectory = ObservationTrajectory(
        "episode-000000",
        observations=torch.zeros(2, 3, 4, 4),
        actions=torch.zeros(1, 10),
    )
    monkeypatch.setattr(cli, "load_pusht_lance_episodes", lambda path, max_episodes: (trajectory,))

    assert cli.main(["inspect-pusht", "hf://dataset", "--max-episodes", "1"]) == 0

    output = capsys.readouterr().out
    assert '"observations_shape": [' in output
    assert "episode-000000" in output


def test_retrieval_image_conversion_accepts_hwc_and_chw_pixels() -> None:
    chw = torch.arange(3 * 2 * 4, dtype=torch.uint8).reshape(3, 2, 4)
    hwc = chw.permute(1, 2, 0)

    chw_image = cli._image_array(chw)
    hwc_image = cli._image_array(hwc)

    assert chw_image.shape == (2, 4, 3)
    assert (chw_image == hwc_image).all()
