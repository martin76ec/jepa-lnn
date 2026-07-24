"""Strict adapter for the pinned official PushT LeWM checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from ...artifacts import file_sha256
from ..lewm import LeWMJEPA, build_lewm_baseline


@dataclass(frozen=True)
class CheckpointSpec:
    """Immutable identity for a downloadable external checkpoint."""

    repository: str
    revision: str
    filename: str
    sha256: str
    config_filename: str
    config_sha256: str
    training_seed: int


OFFICIAL_LEWM_PUSHT = CheckpointSpec(
    repository="quentinll/lewm-pusht",
    revision="22b330c28c27ead4bfd1888615af1340e3fe9052",
    filename="weights.pt",
    sha256="48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
    config_filename="config.json",
    config_sha256="2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09",
    training_seed=3072,
)


def load_official_lewm(checkpoint_path: str | Path, action_input_dim: int) -> LeWMJEPA:
    """Verify and load every tensor from the pinned official checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"official LeWM checkpoint not found: {path}")
    digest = file_sha256(path)
    if digest != OFFICIAL_LEWM_PUSHT.sha256:
        raise ValueError(
            "official LeWM checkpoint SHA-256 mismatch: "
            f"expected {OFFICIAL_LEWM_PUSHT.sha256}, got {digest}"
        )
    config_path = path.with_name(OFFICIAL_LEWM_PUSHT.config_filename)
    if not config_path.is_file():
        raise FileNotFoundError(f"official LeWM model config not found: {config_path}")
    config_digest = file_sha256(config_path)
    if config_digest != OFFICIAL_LEWM_PUSHT.config_sha256:
        raise ValueError(
            "official LeWM config SHA-256 mismatch: "
            f"expected {OFFICIAL_LEWM_PUSHT.config_sha256}, got {config_digest}"
        )
    model = build_lewm_baseline(
        latent_dim=192,
        action_dim=action_input_dim,
        history_size=3,
        num_preds=1,
        sigreg_weight=0.09,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise ValueError("official LeWM checkpoint must be a tensor state dictionary")
    load_official_lewm_weights(model, cast(dict[str, Tensor], state))
    return model.eval()


def load_official_lewm_weights(
    model: LeWMJEPA,
    state_dict: Mapping[str, Tensor],
) -> None:
    """Convert released names and require complete model compatibility."""
    converted: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            converted[convert_official_encoder_key(key)] = value
            continue
        if key.startswith("projector."):
            converted[key] = value
            converted[f"encoder.{key}"] = value
            continue
        converted[key] = value

    incompatible = model.load_state_dict(converted, strict=False)
    expected_missing = {"sigreg.t", "sigreg.phi", "sigreg.weights"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "official LeWM checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def convert_official_encoder_key(key: str) -> str:
    """Map released ViT names to the locked Transformers implementation."""
    converted = key.removeprefix("encoder.")
    converted = converted.replace("encoder.layer.", "layers.")
    converted = converted.replace(".attention.attention.query.", ".attention.q_proj.")
    converted = converted.replace(".attention.attention.key.", ".attention.k_proj.")
    converted = converted.replace(".attention.attention.value.", ".attention.v_proj.")
    converted = converted.replace(".attention.output.dense.", ".attention.o_proj.")
    converted = converted.replace(".intermediate.dense.", ".mlp.fc1.")
    converted = converted.replace(".output.dense.", ".mlp.fc2.")
    return f"encoder.encoder.{converted}"
