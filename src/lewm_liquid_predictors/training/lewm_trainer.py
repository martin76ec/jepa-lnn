"""LeWM baseline training loop with SIGReg, scheduler, and gradient clipping.

This reproduces the upstream ``train.py`` training behavior:
- Two-term loss: pred_loss + lambda * sigreg_loss
- AdamW optimizer with LinearWarmupCosineAnnealingLR scheduler
- Gradient clipping at 1.0
- bf16 mixed precision (on CUDA)
- 90/10 train/validation split
- Per-epoch validation and checkpointing
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from lewm_liquid_predictors.models.lewm import LeWMJEPA


@dataclass(frozen=True)
class LeWMTrainMetrics:
    """Metrics from one LeWM training epoch."""

    total_loss: float
    pred_loss: float
    sigreg_loss: float
    learning_rate: float
    transitions: int


class LeWMTrainer:
    """Train the full LeWM JEPA baseline with the two-term loss."""

    def __init__(
        self,
        model: LeWMJEPA,
        optimizer: Optimizer,
        gradient_clip_val: float = 1.0,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.gradient_clip_val = gradient_clip_val
        self.use_amp = use_amp and torch.cuda.is_available()

    def train_epoch(
        self,
        batches: Iterable[dict[str, Tensor]],
        total_batches: int | None = None,
    ) -> LeWMTrainMetrics:
        """Run one training epoch over pixel/action batches."""
        self.model.train()
        total_loss = 0.0
        total_pred = 0.0
        total_sigreg = 0.0
        total_transitions = 0
        num_batches = 0

        pbar = tqdm(
            batches,
            total=total_batches,
            desc="batches",
            file=sys.stderr,
            leave=False,
        )
        for batch in pbar:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                output = self.model(batch)
                loss = output["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()

            batch_size = batch["pixels"].shape[0]
            total_loss += loss.detach().item()
            total_pred += output["pred_loss"].detach().item()
            total_sigreg += output["sigreg_loss"].detach().item()
            total_transitions += batch_size
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if num_batches == 0:
            raise ValueError("no batches provided")

        lr = self.optimizer.param_groups[0]["lr"]
        return LeWMTrainMetrics(
            total_loss=total_loss / num_batches,
            pred_loss=total_pred / num_batches,
            sigreg_loss=total_sigreg / num_batches,
            learning_rate=lr,
            transitions=total_transitions,
        )

    @torch.no_grad()
    def validate_epoch(
        self,
        batches: Iterable[dict[str, Tensor]],
    ) -> LeWMTrainMetrics:
        """Evaluate the two-term loss without gradient updates."""
        self.model.eval()
        total_loss = 0.0
        total_pred = 0.0
        total_sigreg = 0.0
        total_transitions = 0
        num_batches = 0

        for batch in batches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                output = self.model(batch)
            batch_size = batch["pixels"].shape[0]
            total_loss += output["loss"].detach().item()
            total_pred += output["pred_loss"].detach().item()
            total_sigreg += output["sigreg_loss"].detach().item()
            total_transitions += batch_size
            num_batches += 1

        if num_batches == 0:
            raise ValueError("no validation batches")

        return LeWMTrainMetrics(
            total_loss=total_loss / num_batches,
            pred_loss=total_pred / num_batches,
            sigreg_loss=total_sigreg / num_batches,
            learning_rate=0.0,
            transitions=total_transitions,
        )


def build_linear_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> LambdaLR:
    """Linear warmup + cosine annealing schedule matching upstream."""
    if warmup_steps < 0 or max_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and max_steps must be positive")
    if warmup_steps >= max_steps:
        raise ValueError("warmup_steps must be less than max_steps")

    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
