"""Safe, isolated persistence for decoder-only checkpoints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
import yaml
from torch import Tensor

from lewm_liquid_predictors.artifacts import file_sha256 as file_sha256
from lewm_liquid_predictors.training.provenance import RunProvenance

from .config import DecoderConfig
from .model import CrossAttentionDecoder, DecoderArchitecture

DECODER_CHECKPOINT_FILENAME = "decoder.pt"
SOURCE_CHECKPOINTS_FILENAME = "source_checkpoints.json"


def initialize_decoder_run(
    config: DecoderConfig,
    provenance: RunProvenance,
) -> Path:
    """Create a unique decoder-only run directory with resolved metadata."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = config.experiment.output_dir / (f"run_{timestamp}_seed{config.experiment.seed}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(_jsonable(asdict(config)), sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(_jsonable(asdict(provenance)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_dir


def save_decoder_checkpoint(decoder: CrossAttentionDecoder, path: str | Path) -> None:
    """Atomically save a decoder state dict without overwriting any existing file."""
    checkpoint_path = _decoder_path(path)
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite decoder checkpoint: {checkpoint_path}")
    temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"temporary decoder checkpoint already exists: {temporary_path}")
    try:
        torch.save(decoder.state_dict(), temporary_path)
        if checkpoint_path.exists():
            raise FileExistsError(f"refusing to overwrite decoder checkpoint: {checkpoint_path}")
        temporary_path.replace(checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_decoder_checkpoint(
    path: str | Path,
    architecture: DecoderArchitecture,
    *,
    map_location: str | torch.device = "cpu",
) -> CrossAttentionDecoder:
    """Load a strictly validated decoder state dict for an explicit architecture."""
    checkpoint_path = _decoder_path(path)
    state_dict = cast(
        dict[str, Tensor],
        torch.load(checkpoint_path, map_location=map_location, weights_only=True),
    )
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state_dict.items()
    ):
        raise ValueError("decoder checkpoint must contain only a tensor state_dict")
    decoder = CrossAttentionDecoder(architecture)
    decoder.load_state_dict(state_dict, strict=True)
    return decoder


def write_source_checkpoints(
    output_dir: str | Path,
    source_paths: Sequence[str | Path],
) -> Path:
    """Write source checkpoint paths and SHA-256 digests as strict JSON."""
    destination = Path(output_dir) / SOURCE_CHECKPOINTS_FILENAME
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint metadata: {destination}")
    metadata = {str(path): file_sha256(path) for path in source_paths}
    destination.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _decoder_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.name != DECODER_CHECKPOINT_FILENAME:
        raise ValueError(f"decoder checkpoint filename must be {DECODER_CHECKPOINT_FILENAME}")
    return checkpoint_path


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value, default=str, allow_nan=False))
