"""Tests for experiment configuration loading."""

from pathlib import Path

import pytest

from lewm_liquid_predictors.utils import load_config

ROOT = Path(__file__).parents[1]


def test_load_local_config_normalizes_scalar_seed_and_split() -> None:
    config = load_config(ROOT / "configs" / "local.yaml")

    assert config.experiment.seeds == (7,)
    assert config.data.splits == ("train",)
    assert config.data.frameskip == 5
    assert config.training.batch_size == 8
    assert config.evaluation.rollout_horizons == (1, 5, 10, 20, 50)


def test_load_h200_config_preserves_unresolved_training_values() -> None:
    config = load_config(ROOT / "configs" / "h200.yaml")

    assert config.experiment.seeds == (7, 19, 43, 71, 97)
    assert config.training.batch_size is None
    assert config.training.max_epochs is None


def test_config_rejects_invalid_fraction(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
experiment:
  name: invalid
  seed: 1
  output_dir: runs/invalid
data:
  dataset: test
  fraction: 0
  num_workers: 0
  split: train
training:
  device: cpu
  max_epochs: 1
  batch_size: 1
  deterministic: true
model:
  variant: mlp
  latent_dim: 1
  action_dim: 1
  hidden_dim: 1
  transformer_heads: 1
  transformer_layers: 1
  transformer_context_length: 1
evaluation:
  rollout_horizons: [1]
  divergence:
    normalized_error_threshold: 1
    count_non_finite_as_divergence: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fraction"):
        load_config(config_path)
