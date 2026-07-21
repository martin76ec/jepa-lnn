"""Tests for the LeWM baseline trainer."""

import torch

from lewm_liquid_predictors.models import build_lewm_baseline
from lewm_liquid_predictors.training import LeWMTrainer, build_linear_warmup_cosine_scheduler


def _small_batch(batch_size: int = 2, seq_len: int = 4) -> dict[str, torch.Tensor]:
    return {
        "pixels": torch.randn(batch_size, seq_len, 3, 224, 224),
        "action": torch.randn(batch_size, seq_len, 10),
    }


def test_lewm_trainer_runs_one_epoch_and_updates_parameters() -> None:
    model = build_lewm_baseline(latent_dim=192, action_dim=10, history_size=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    trainer = LeWMTrainer(model, optimizer, gradient_clip_val=1.0)

    before = [param.detach().clone() for param in model.parameters()]
    metrics = trainer.train_epoch([_small_batch()])

    after = list(model.parameters())
    assert metrics.total_loss > 0
    assert metrics.pred_loss >= 0
    assert metrics.sigreg_loss >= 0
    assert metrics.transitions == 2
    assert any(not torch.equal(prev, curr) for prev, curr in zip(before, after, strict=True))


def test_lewm_trainer_validation_does_not_update_parameters() -> None:
    model = build_lewm_baseline(latent_dim=192, action_dim=10, history_size=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    trainer = LeWMTrainer(model, optimizer)

    before = [param.detach().clone() for param in model.parameters()]
    metrics = trainer.validate_epoch([_small_batch()])
    after = list(model.parameters())

    assert metrics.total_loss > 0
    assert all(torch.equal(prev, curr) for prev, curr in zip(before, after, strict=True))


def test_linear_warmup_cosine_scheduler_starts_at_zero() -> None:
    model = build_lewm_baseline(latent_dim=192, action_dim=10, history_size=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = build_linear_warmup_cosine_scheduler(optimizer, warmup_steps=10, max_steps=100)

    lrs = []
    for _ in range(20):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert lrs[0] < lrs[1] < lrs[10]
    assert lrs[10] > lrs[19]
