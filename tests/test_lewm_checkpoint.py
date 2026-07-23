"""Tests for the pinned official LeWM checkpoint adapter."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from lewm_liquid_predictors.models.checkpoint_adapters import (
    OFFICIAL_LEWM_PUSHT,
    load_official_lewm,
)
from lewm_liquid_predictors.models.checkpoint_adapters.lewm import convert_official_encoder_key

ROOT = Path(__file__).parents[1]
CHECKPOINT = ROOT / "checkpoints" / "lewm-pusht" / "weights.pt"


@pytest.mark.parametrize(
    ("official", "local"),
    [
        ("encoder.embeddings.cls_token", "encoder.encoder.embeddings.cls_token"),
        (
            "encoder.encoder.layer.2.attention.attention.query.weight",
            "encoder.encoder.layers.2.attention.q_proj.weight",
        ),
        (
            "encoder.encoder.layer.4.intermediate.dense.bias",
            "encoder.encoder.layers.4.mlp.fc1.bias",
        ),
        (
            "encoder.encoder.layer.7.output.dense.weight",
            "encoder.encoder.layers.7.mlp.fc2.weight",
        ),
    ],
)
def test_official_encoder_checkpoint_keys_are_converted(official: str, local: str) -> None:
    assert convert_official_encoder_key(official) == local


def test_official_checkpoint_rejects_wrong_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / OFFICIAL_LEWM_PUSHT.filename
    checkpoint.write_bytes(b"not the official checkpoint")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_official_lewm(checkpoint, action_input_dim=10)


@pytest.mark.skipif(
    not CHECKPOINT.is_file() or find_spec("transformers") is None,
    reason="requires the downloaded official checkpoint and upstream dependencies",
)
def test_downloaded_official_checkpoint_loads_completely() -> None:
    model = load_official_lewm(CHECKPOINT, action_input_dim=10)

    assert model.history_size == 3
    assert model.predictor.pos_embedding.shape == (1, 3, 192)
