"""Typed experiment configuration loading and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import yaml

PredictorVariant: TypeAlias = Literal["lewm_ar", "mlp", "transformer", "cfc", "ltc"]
EncoderMode: TypeAlias = Literal["smoke", "upstream"]


@dataclass(frozen=True)
class ExperimentSettings:
    """Experiment identity and random seed settings."""

    name: str
    output_dir: Path
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class DataSettings:
    """Dataset selection and split settings."""

    dataset: str
    frameskip: int
    fraction: float
    num_workers: int
    splits: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSettings:
    """Training runtime settings; unresolved values are allowed in draft configs."""

    device: str
    max_epochs: int | None
    batch_size: int | None
    deterministic: bool


@dataclass(frozen=True)
class PredictorSettings:
    """Architecture selection and dimensions shared by predictor variants."""

    variant: PredictorVariant
    encoder_mode: EncoderMode
    latent_dim: int
    action_dim: int
    hidden_dim: int
    transformer_heads: int
    transformer_layers: int
    transformer_context_length: int


@dataclass(frozen=True)
class DivergenceSettings:
    """Closed-loop divergence policy."""

    normalized_error_threshold: float
    count_non_finite_as_divergence: bool


@dataclass(frozen=True)
class EvaluationSettings:
    """Rollout evaluation settings."""

    rollout_horizons: tuple[int, ...]
    divergence: DivergenceSettings


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved, immutable experiment configuration."""

    experiment: ExperimentSettings
    data: DataSettings
    training: TrainingSettings
    model: PredictorSettings
    evaluation: EvaluationSettings


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return _parse_config(raw)


def _parse_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    experiment = _mapping(raw, "experiment")
    data = _mapping(raw, "data")
    training = _mapping(raw, "training")
    model = _mapping(raw, "model")
    evaluation = _mapping(raw, "evaluation")
    divergence = _mapping(evaluation, "divergence")

    seeds = _tuple_of_ints(experiment.get("seeds", experiment.get("seed")), "seeds")
    splits = _tuple_of_strings(data.get("splits", data.get("split")), "splits")
    horizons = _tuple_of_ints(evaluation.get("rollout_horizons"), "rollout_horizons")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("rollout_horizons must contain only positive values")

    fraction = _float(data.get("fraction"), "fraction")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")

    return ExperimentConfig(
        experiment=ExperimentSettings(
            name=_string(experiment.get("name"), "experiment.name"),
            output_dir=Path(_string(experiment.get("output_dir"), "experiment.output_dir")),
            seeds=seeds,
        ),
        data=DataSettings(
            dataset=_string(data.get("dataset"), "data.dataset"),
            frameskip=_positive_int(data.get("frameskip"), "data.frameskip"),
            fraction=fraction,
            num_workers=_positive_int(data.get("num_workers"), "data.num_workers", allow_zero=True),
            splits=splits,
        ),
        training=TrainingSettings(
            device=_string(training.get("device"), "training.device"),
            max_epochs=_optional_positive_int(training.get("max_epochs"), "training.max_epochs"),
            batch_size=_optional_positive_int(training.get("batch_size"), "training.batch_size"),
            deterministic=_bool(training.get("deterministic"), "training.deterministic"),
        ),
        model=PredictorSettings(
            variant=_predictor_variant(model.get("variant")),
            encoder_mode=_encoder_mode(model.get("encoder_mode")),
            latent_dim=_positive_int(model.get("latent_dim"), "model.latent_dim"),
            action_dim=_positive_int(model.get("action_dim"), "model.action_dim"),
            hidden_dim=_positive_int(model.get("hidden_dim"), "model.hidden_dim"),
            transformer_heads=_positive_int(
                model.get("transformer_heads"), "model.transformer_heads"
            ),
            transformer_layers=_positive_int(
                model.get("transformer_layers"), "model.transformer_layers"
            ),
            transformer_context_length=_positive_int(
                model.get("transformer_context_length"),
                "model.transformer_context_length",
            ),
        ),
        evaluation=EvaluationSettings(
            rollout_horizons=horizons,
            divergence=DivergenceSettings(
                normalized_error_threshold=_positive_float(
                    divergence.get("normalized_error_threshold"),
                    "evaluation.divergence.normalized_error_threshold",
                ),
                count_non_finite_as_divergence=_bool(
                    divergence.get("count_non_finite_as_divergence"),
                    "evaluation.divergence.count_non_finite_as_divergence",
                ),
            ),
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
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    parsed = _float(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _tuple_of_ints(value: Any, name: str) -> tuple[int, ...]:
    values = (value,) if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty integer or sequence of integers")
    return tuple(_positive_int(item, name) for item in values)


def _tuple_of_strings(value: Any, name: str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty string or sequence of strings")
    return tuple(_string(item, name) for item in values)


def _predictor_variant(value: Any) -> PredictorVariant:
    variant = _string(value, "model.variant")
    if variant not in {"lewm_ar", "mlp", "transformer", "cfc", "ltc"}:
        raise ValueError("model.variant must be one of: lewm_ar, mlp, transformer, cfc, ltc")
    return cast(PredictorVariant, variant)


def _encoder_mode(value: Any) -> EncoderMode:
    mode = _string(value, "model.encoder_mode")
    if mode not in {"smoke", "upstream"}:
        raise ValueError("model.encoder_mode must be one of: smoke, upstream")
    return cast(EncoderMode, mode)
