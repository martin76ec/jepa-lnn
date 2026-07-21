"""Exact LeWM baseline reproduction from the pinned upstream source.

This module faithfully reimplements the upstream LeWM JEPA architecture from
``https://github.com/lucas-maes/le-wm.git`` at commit ``c8a44170``:

- SIGReg: Sketch Isotropic Gaussian Regularizer
- ARPredictor: Causal Transformer with AdaLN-zero conditioning
- JEPA: Encoder + projector + action encoder + predictor + pred_proj
- LeWM loss: pred_loss + lambda * sigreg_loss

The upstream ``module.py`` and ``jepa.py`` are the source of truth.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from .encoder import LeWMEncoder
from .upstream_encoder import UpstreamEmbedder, UpstreamMLP, build_upstream_vit


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN-zero modulation: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU).

    Computes the Epps-Pulley statistic on random projections of the latent
    embeddings and compares against a Gaussian characteristic function.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        """Compute the SIGReg statistic.

        Args:
            proj: Latent embeddings of shape (T, B, D).
        """
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        t = cast(Tensor, self.t)
        phi = cast(Tensor, self.phi)
        weights = cast(Tensor, self.weights)
        x_t = (proj @ a).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    """FeedForward network used in Transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out: nn.Module = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return cast(Tensor, self.to_out(out))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(
        self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(cast(nn.Linear, self.adaLN_modulation[-1]).weight, 0)
        nn.init.constant_(cast(nn.Linear, self.adaLN_modulation[-1]).bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block."""

    def __init__(
        self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LeWMTransformer(nn.Module):
    """Transformer with support for AdaLN-zero conditional blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class: type[nn.Module] = Block,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList()
        self.input_proj: nn.Module = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj: nn.Module = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.output_proj: nn.Module = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )
        for _ in range(depth):
            self.layers.append(block_class(hidden_dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else cast(Tensor, block(x, c))
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction.

    This is the exact upstream LeWM predictor: positional embedding +
    causal Transformer with AdaLN-zero conditioning on action embeddings.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = LeWMTransformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Predict next embeddings.

        Args:
            x: Latent embeddings of shape (B, T, D).
            c: Action embeddings of shape (B, T, A_emb).
        """
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return cast(Tensor, self.transformer(x, c))


def rearrange(tensor: Tensor, pattern: str, **kwargs: int) -> Tensor:
    """Minimal einops.rearrange for the patterns used by this module."""
    if pattern == "b t (h d) -> b h t d":
        h = kwargs["h"]
        b, t, _ = tensor.shape
        d = tensor.shape[-1] // h
        return tensor.view(b, t, h, d).permute(0, 2, 1, 3)
    if pattern == "b h t d -> b t (h d)":
        b, h, t, d = tensor.shape
        return tensor.permute(0, 2, 1, 3).contiguous().view(b, t, h * d)
    raise ValueError(f"unsupported rearrange pattern: {pattern}")


class LeWMJEPA(nn.Module):
    """Full LeWM JEPA model: encoder + projector + action encoder + predictor + pred_proj.

    This faithfully reproduces the upstream ``jepa.JEPA`` architecture.
    The forward pass encodes pixels and actions, predicts next embeddings,
    and computes the two-term LeWM loss.
    """

    def __init__(
        self,
        encoder: LeWMEncoder,
        action_encoder: UpstreamEmbedder,
        predictor: ARPredictor,
        projector: UpstreamMLP | None = None,
        pred_proj: UpstreamMLP | None = None,
        sigreg: SIGReg | None = None,
        sigreg_weight: float = 0.09,
        history_size: int = 3,
        num_preds: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector: nn.Module = projector or nn.Identity()
        self.pred_proj: nn.Module = pred_proj or nn.Identity()
        self.sigreg = sigreg or SIGReg(knots=17, num_proj=1024)
        self.sigreg_weight = sigreg_weight
        self.history_size = history_size
        self.num_preds = num_preds

    def encode(self, info: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode observations and actions into embeddings.

        Args:
            info: Dict with 'pixels' (B, T, C, H, W) and 'action' (B, T, A).
        """
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels_flat = pixels.view(b * pixels.size(1), *pixels.shape[2:])
        output = self.encoder(pixels_flat.unsqueeze(0)).squeeze(0)
        emb = output.view(b, pixels.size(1), -1)
        info["emb"] = emb
        if "action" in info:
            info["act_emb"] = cast(Tensor, self.action_encoder(info["action"]))
        return info

    def predict(self, emb: Tensor, act_emb: Tensor) -> Tensor:
        """Predict next state embeddings.

        Args:
            emb: (B, T, D)
            act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds_flat = preds.view(-1, preds.size(-1))
        projected = cast(Tensor, self.pred_proj(preds_flat))
        return projected.view(preds.size(0), preds.size(1), -1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Full LeWM forward pass with two-term loss.

        Args:
            batch: Dict with 'pixels' (B, T, C, H, W) and 'action' (B, T, A).
        """
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        output = self.encode(batch)

        emb = output["emb"]
        act_emb = output["act_emb"]

        ctx_len = self.history_size
        n_preds = self.num_preds

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]

        tgt_emb = emb[:, n_preds:]
        pred_emb = self.predict(ctx_emb, ctx_act)

        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["pred_loss"] + self.sigreg_weight * output["sigreg_loss"]
        return output


def build_lewm_baseline(
    latent_dim: int = 192,
    action_dim: int = 10,
    history_size: int = 3,
    num_preds: int = 1,
    sigreg_weight: float = 0.09,
) -> LeWMJEPA:
    """Build the complete LeWM baseline model matching upstream config.

    Args:
        latent_dim: Embedding dimension (upstream: 192).
        action_dim: Raw action dimension after frameskip (upstream: 10).
        history_size: Context length (upstream: 3).
        num_preds: Number of predictions (upstream: 1).
        sigreg_weight: SIGReg loss weight (upstream: 0.09).
    """
    vit = build_upstream_vit(size="tiny", patch_size=14, image_size=224)
    projector = UpstreamMLP(latent_dim, 2048, latent_dim)
    encoder = LeWMEncoder(vit, projector)
    action_encoder = UpstreamEmbedder(input_dim=action_dim, emb_dim=latent_dim)
    predictor = ARPredictor(
        num_frames=history_size,
        depth=6,
        heads=16,
        mlp_dim=2048,
        input_dim=latent_dim,
        hidden_dim=latent_dim,
        output_dim=latent_dim,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    pred_proj = UpstreamMLP(latent_dim, 2048, latent_dim)
    sigreg = SIGReg(knots=17, num_proj=1024)
    return LeWMJEPA(
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        projector=projector,
        pred_proj=pred_proj,
        sigreg=sigreg,
        sigreg_weight=sigreg_weight,
        history_size=history_size,
        num_preds=num_preds,
    )
