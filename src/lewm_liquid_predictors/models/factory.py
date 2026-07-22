"""Configuration-driven construction of dynamics predictors."""

from __future__ import annotations

from torch import nn

from lewm_liquid_predictors.utils.config import PredictorSettings

from .lewm import LeWMARPredictor
from .liquid import PredictorCfC, PredictorLTC
from .mlp import PredictorMLP
from .transformer import PredictorTransformer


def build_predictor(settings: PredictorSettings) -> nn.Module:
    """Build one predictor variant from the common typed configuration."""
    if settings.variant == "lewm_ar":
        return LeWMARPredictor(
            settings.latent_dim,
            settings.action_dim,
            settings.transformer_context_length,
        )
    if settings.variant == "mlp":
        return PredictorMLP(settings.latent_dim, settings.action_dim, settings.hidden_dim)
    if settings.variant == "transformer":
        return PredictorTransformer(
            latent_dim=settings.latent_dim,
            action_dim=settings.action_dim,
            hidden_dim=settings.hidden_dim,
            num_heads=settings.transformer_heads,
            num_layers=settings.transformer_layers,
            context_length=settings.transformer_context_length,
        )
    if settings.variant == "cfc":
        return PredictorCfC(settings.latent_dim, settings.action_dim, settings.hidden_dim)
    return PredictorLTC(settings.latent_dim, settings.action_dim, settings.hidden_dim)
