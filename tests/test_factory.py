"""Tests for configuration-driven predictor selection."""

from dataclasses import replace
from pathlib import Path

import pytest

from lewm_liquid_predictors.models import (
    PredictorCfC,
    PredictorLTC,
    PredictorMLP,
    PredictorTransformer,
    build_predictor,
)
from lewm_liquid_predictors.utils import load_config

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("variant", "predictor_type"),
    [
        ("mlp", PredictorMLP),
        ("transformer", PredictorTransformer),
        ("cfc", PredictorCfC),
        ("ltc", PredictorLTC),
    ],
)
def test_factory_builds_selected_predictor(variant: str, predictor_type: type[object]) -> None:
    settings = replace(load_config(ROOT / "configs" / "local.yaml").model, variant=variant)

    assert isinstance(build_predictor(settings), predictor_type)
