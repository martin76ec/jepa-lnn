"""Tests for the isolated decoder experiment configuration."""

from pathlib import Path

import pytest

from lewm_liquid_predictors.decoder import load_decoder_config

ROOT = Path(__file__).parents[1]


def test_load_h200_decoder_config_uses_isolated_artifact_namespace() -> None:
    config = load_decoder_config(ROOT / "configs" / "h200-decoder.yaml")

    assert config.experiment.output_dir == Path("runs/h200-decoder")
    assert config.data.split_manifest == Path("runs/h200-screen/split_manifest.json")
    assert config.architecture.latent_dim == 192
    assert config.architecture.num_patches == 196
    assert config.data.fraction == 1.0
    assert config.training.epochs == 100
    assert config.training.loss == "l1_lpips"
    assert config.training.l1_weight == 1.0
    assert config.training.lpips_weight == 1.0
    assert config.training.lpips_network == "vgg"
    assert config.galleries.horizons == (1, 5, 10, 20)


def test_decoder_config_rejects_nondivisible_patch_grid(tmp_path: Path) -> None:
    content = (ROOT / "configs" / "h200-decoder.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(content.replace("patch_size: 16", "patch_size: 15"), encoding="utf-8")

    with pytest.raises(ValueError, match="divisible"):
        load_decoder_config(path)


def test_decoder_config_rejects_unknown_loss(tmp_path: Path) -> None:
    content = (ROOT / "configs" / "h200-decoder.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(content.replace("loss: l1_lpips", "loss: perceptual"), encoding="utf-8")

    with pytest.raises(ValueError, match="training.loss"):
        load_decoder_config(path)


def test_decoder_config_rejects_non_finite_loss_weight(tmp_path: Path) -> None:
    content = (ROOT / "configs" / "h200-decoder.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(content.replace("l1_weight: 1.0", "l1_weight: .nan"), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_decoder_config(path)
