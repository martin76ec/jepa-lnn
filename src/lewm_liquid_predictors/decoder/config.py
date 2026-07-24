"""Typed configuration for post-hoc decoder training and rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import yaml

from .model import DecoderArchitecture

DecoderLoss: TypeAlias = Literal["mse", "l1", "l1_lpips"]
LPIPSNetwork: TypeAlias = Literal["alex", "vgg", "squeeze"]


@dataclass(frozen=True)
class DecoderExperimentSettings:
    """Identity and output location for one decoder run."""

    name: str
    output_dir: Path
    seed: int


@dataclass(frozen=True)
class DecoderDataSettings:
    """Dataset and existing predictor split used by the decoder."""

    dataset: str
    frameskip: int
    fraction: float
    num_workers: int
    split_manifest: Path


@dataclass(frozen=True)
class DecoderTrainingSettings:
    """Fixed decoder-only optimization budget."""

    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    deterministic: bool
    use_amp: bool
    loss: DecoderLoss
    l1_weight: float
    lpips_weight: float
    lpips_network: LPIPSNetwork

    def __post_init__(self) -> None:
        if not isfinite(self.l1_weight) or not isfinite(self.lpips_weight):
            raise ValueError("decoder loss weights must be finite")
        if self.loss in {"l1", "l1_lpips"} and self.l1_weight <= 0:
            raise ValueError("l1_weight must be positive when L1 loss is enabled")
        if self.loss == "l1_lpips" and self.lpips_weight <= 0:
            raise ValueError("lpips_weight must be positive when LPIPS loss is enabled")


@dataclass(frozen=True)
class DecoderGallerySettings:
    """Inference-only gallery rendering settings."""

    horizons: tuple[int, ...]
    episode_count: int


@dataclass(frozen=True)
class DecoderConfig:
    """Complete immutable decoder experiment configuration."""

    experiment: DecoderExperimentSettings
    data: DecoderDataSettings
    training: DecoderTrainingSettings
    architecture: DecoderArchitecture
    galleries: DecoderGallerySettings


def load_decoder_config(path: str | Path) -> DecoderConfig:
    """Load and validate a decoder YAML configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ValueError(f"decoder configuration root must be a mapping: {config_path}")

    experiment = _mapping(raw, "experiment")
    data = _mapping(raw, "data")
    training = _mapping(raw, "training")
    architecture = _mapping(raw, "architecture")
    galleries = _mapping(raw, "galleries")
    fraction = _float(data.get("fraction"), "data.fraction")
    if not 0 < fraction <= 1:
        raise ValueError("data.fraction must be greater than 0 and at most 1")
    horizons = _tuple_of_positive_ints(galleries.get("horizons"), "galleries.horizons")
    if len(set(horizons)) != len(horizons):
        raise ValueError("galleries.horizons must not contain duplicates")

    return DecoderConfig(
        experiment=DecoderExperimentSettings(
            name=_string(experiment.get("name"), "experiment.name"),
            output_dir=Path(_string(experiment.get("output_dir"), "experiment.output_dir")),
            seed=_positive_int(experiment.get("seed"), "experiment.seed", allow_zero=True),
        ),
        data=DecoderDataSettings(
            dataset=_string(data.get("dataset"), "data.dataset"),
            frameskip=_positive_int(data.get("frameskip"), "data.frameskip"),
            fraction=fraction,
            num_workers=_positive_int(data.get("num_workers"), "data.num_workers", allow_zero=True),
            split_manifest=Path(_string(data.get("split_manifest"), "data.split_manifest")),
        ),
        training=DecoderTrainingSettings(
            device=_string(training.get("device"), "training.device"),
            epochs=_positive_int(training.get("epochs"), "training.epochs"),
            batch_size=_positive_int(training.get("batch_size"), "training.batch_size"),
            learning_rate=_positive_float(training.get("learning_rate"), "training.learning_rate"),
            weight_decay=_nonnegative_float(training.get("weight_decay"), "training.weight_decay"),
            deterministic=_bool(training.get("deterministic"), "training.deterministic"),
            use_amp=_bool(training.get("use_amp"), "training.use_amp"),
            loss=_decoder_loss(training.get("loss")),
            l1_weight=_nonnegative_float(training.get("l1_weight"), "training.l1_weight"),
            lpips_weight=_nonnegative_float(training.get("lpips_weight"), "training.lpips_weight"),
            lpips_network=_lpips_network(training.get("lpips_network")),
        ),
        architecture=DecoderArchitecture(
            latent_dim=_positive_int(architecture.get("latent_dim"), "architecture.latent_dim"),
            hidden_dim=_positive_int(architecture.get("hidden_dim"), "architecture.hidden_dim"),
            image_size=_positive_int(architecture.get("image_size"), "architecture.image_size"),
            patch_size=_positive_int(architecture.get("patch_size"), "architecture.patch_size"),
            channels=_positive_int(architecture.get("channels"), "architecture.channels"),
            num_layers=_positive_int(architecture.get("num_layers"), "architecture.num_layers"),
            num_heads=_positive_int(architecture.get("num_heads"), "architecture.num_heads"),
            mlp_ratio=_positive_int(architecture.get("mlp_ratio"), "architecture.mlp_ratio"),
            dropout=_nonnegative_float(architecture.get("dropout"), "architecture.dropout"),
        ),
        galleries=DecoderGallerySettings(
            horizons=horizons,
            episode_count=_positive_int(galleries.get("episode_count"), "galleries.episode_count"),
        ),
    )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return nested


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = _float(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = _float(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive")
    return value


def _tuple_of_positive_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    return tuple(_positive_int(item, name) for item in value)


def _decoder_loss(value: Any) -> DecoderLoss:
    loss = _string(value, "training.loss")
    if loss not in {"mse", "l1", "l1_lpips"}:
        raise ValueError("training.loss must be one of: mse, l1, l1_lpips")
    return cast(DecoderLoss, loss)


def _lpips_network(value: Any) -> LPIPSNetwork:
    network = _string(value, "training.lpips_network")
    if network not in {"alex", "vgg", "squeeze"}:
        raise ValueError("training.lpips_network must be one of: alex, vgg, squeeze")
    return cast(LPIPSNetwork, network)
