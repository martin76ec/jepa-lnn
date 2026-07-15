"""Smoke tests for the importable experiment package."""

from lewm_liquid_predictors.models.protocol import DynamicsPredictor


def test_predictor_protocol_is_importable() -> None:
    assert DynamicsPredictor.__name__ == "DynamicsPredictor"
