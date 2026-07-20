"""Tests for reproducibility metadata persistence."""

import json
from pathlib import Path

from lewm_liquid_predictors.training import (
    capture_run_provenance,
    initialize_run,
    write_metrics,
)
from lewm_liquid_predictors.utils import load_config

ROOT = Path(__file__).parents[1]


def test_initialize_run_persists_config_provenance_and_metrics(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "local.yaml")
    provenance = capture_run_provenance(
        seed=7,
        requested_device="cpu",
        git_commit="test-commit",
        packages={"torch": "test"},
    )

    run_dir = initialize_run(tmp_path / "run", config, provenance)
    write_metrics(run_dir, {"validation/loss": 0.25})

    resolved_config = (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    saved_provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "frameskip: 5" in resolved_config
    assert saved_provenance["git_commit"] == "test-commit"
    assert saved_provenance["packages"] == {"torch": "test"}
    assert saved_metrics == {"validation/loss": 0.25}
