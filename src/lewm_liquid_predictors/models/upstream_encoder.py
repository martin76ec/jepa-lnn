"""Upstream LeWM encoder components, faithfully reimplemented.

These match the architecture used by the pinned upstream LeWM model:

- ViT: ``transformers.ViTModel`` with ``size="tiny"``, ``patch_size=14``,
  ``image_size=224``, ``pretrained=False``, ``use_mask_token=False``
- Projector: ``MLP(input_dim=192, hidden_dim=2048, output_dim=192,
  norm_fn=BatchNorm1d, act_fn=GELU)``
- Action encoder: ``Embedder(input_dim, smoothed_dim=input_dim,
  emb_dim=192, mlp_scale=4)``

The upstream source lives at commit ``c8a44170`` in
``https://github.com/lucas-maes/le-wm.git`` (module.py and
``config/train/model/lewm.yaml``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from torch import Tensor, nn

from .encoder import LeWMEncoder

_VIT_SIZE_CONFIGS: dict[str, dict[str, int]] = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
}


class UpstreamMLP(nn.Module):
    """Projector MLP with BatchNorm1d and GELU, matching upstream ``module.MLP``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        norm: nn.Module = nn.BatchNorm1d(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class UpstreamEmbedder(nn.Module):
    """Action encoder matching upstream ``module.Embedder``.

    Conv1d (1x1) → SiLU MLP, operating on (B, T, D) sequences.
    """

    def __init__(
        self,
        input_dim: int,
        smoothed_dim: int | None = None,
        emb_dim: int = 192,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        smoothed = smoothed_dim or input_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = x.permute(0, 2, 1)
        x = cast(Tensor, self.patch_embed(x))
        x = x.permute(0, 2, 1)
        return cast(Tensor, self.embed(x))


def build_upstream_vit(
    size: str = "tiny",
    patch_size: int = 14,
    image_size: int = 224,
) -> nn.Module:
    """Build a ViT matching the pinned upstream ``vit_hf`` helper."""
    if size not in _VIT_SIZE_CONFIGS:
        raise ValueError(f"invalid ViT size '{size}'; choose from {list(_VIT_SIZE_CONFIGS)}")
    transformers: Any = import_module("transformers")
    config_params = dict(_VIT_SIZE_CONFIGS[size])
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config = transformers.ViTConfig(**config_params)
    model = cast(
        nn.Module, transformers.ViTModel(config, add_pooling_layer=False, use_mask_token=False)
    )
    cast(Any, model).config.interpolate_pos_encoding = True
    return model


def build_upstream_encoder(
    latent_dim: int = 192,
    vit_size: str = "tiny",
    patch_size: int = 14,
    image_size: int = 224,
    projector_hidden_dim: int = 2048,
) -> LeWMEncoder:
    """Build the upstream LeWM ViT encoder + projector wrapper."""
    vit = build_upstream_vit(size=vit_size, patch_size=patch_size, image_size=image_size)
    projector = UpstreamMLP(latent_dim, projector_hidden_dim, latent_dim)
    return LeWMEncoder(vit, projector)


def build_upstream_action_encoder(
    input_dim: int,
    emb_dim: int = 192,
) -> UpstreamEmbedder:
    """Build the upstream action encoder with matching architecture."""
    return UpstreamEmbedder(input_dim=input_dim, emb_dim=emb_dim)
