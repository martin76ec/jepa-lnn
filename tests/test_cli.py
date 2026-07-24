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


def test_train_decoder_command_delegates_without_predictor_training(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    run_dir = tmp_path / "decoder-run"
    calls: list[tuple[object, ...]] = []

    def fake_train(*args: object, **kwargs: object) -> Path:
        calls.append((*args, kwargs))
        return run_dir

    monkeypatch.setattr(cli, "train_decoder", fake_train)  # type: ignore[attr-defined]

    assert cli.main(["train-decoder", "decoder.yaml", "--max-frames", "4"]) == 0

    assert calls
    assert '"decoder_run"' in capsys.readouterr().out


def test_render_decoder_command_delegates_to_existing_predictor_root(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    output_dir = tmp_path / "galleries"
    calls: list[tuple[object, ...]] = []

    def fake_render(*args: object, **kwargs: object) -> Path:
        calls.append((*args, kwargs))
        return output_dir

    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "render_decoder_galleries", fake_render
    )

    assert (
        cli.main(
            [
                "render-decoder-galleries",
                "decoder.yaml",
                "--predictor-root",
                "runs/h200-screen",
                "--max-predictor-runs",
                "1",
            ]
        )
        == 0
    )

    assert calls
    assert '"decoded_galleries"' in capsys.readouterr().out
