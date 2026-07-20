"""Run metadata capture and persistence."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import torch
import yaml

from lewm_liquid_predictors.utils import ExperimentConfig


@dataclass(frozen=True)
class RunProvenance:
    """Immutable metadata required to reproduce an experiment run."""

    created_at: str
    git_commit: str
    seed: int
    requested_device: str
    device: Mapping[str, str | int | bool | None]
    python: str
    platform: str
    packages: Mapping[str, str]


def capture_run_provenance(
    seed: int,
    requested_device: str,
    *,
    repo_root: str | Path | None = None,
    git_commit: str | None = None,
    packages: Mapping[str, str] | None = None,
) -> RunProvenance:
    """Capture versioned source, environment, and device metadata for one run."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    root = Path.cwd() if repo_root is None else Path(repo_root)
    return RunProvenance(
        created_at=datetime.now(UTC).isoformat(),
        git_commit=_git_commit(root) if git_commit is None else git_commit,
        seed=seed,
        requested_device=requested_device,
        device=_device_metadata(),
        python=sys.version,
        platform=platform.platform(),
        packages=_package_versions() if packages is None else dict(sorted(packages.items())),
    )


def initialize_run(
    run_dir: str | Path,
    config: ExperimentConfig,
    provenance: RunProvenance,
) -> Path:
    """Create a run directory and persist resolved config and provenance."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_yaml(output_dir / "resolved_config.yaml", _jsonable(asdict(config)))
    _write_json(output_dir / "provenance.json", _jsonable(asdict(provenance)))
    return output_dir


def write_metrics(run_dir: str | Path, metrics: Mapping[str, float]) -> None:
    """Persist scalar metrics for a completed train or evaluation phase."""
    if not metrics:
        raise ValueError("metrics must not be empty")
    _write_json(Path(run_dir) / "metrics.json", dict(sorted(metrics.items())))


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _device_metadata() -> dict[str, str | int | bool | None]:
    metadata: dict[str, str | int | bool | None] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
    }
    if torch.cuda.is_available():
        metadata["device_name"] = torch.cuda.get_device_name(0)
    return metadata


def _package_versions() -> dict[str, str]:
    return dict(
        sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in distributions()
        )
    )


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_yaml(path: Path, content: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(content), file, sort_keys=True)
