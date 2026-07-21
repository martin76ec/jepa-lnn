"""Executable entry points for configuration, data, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from tqdm import tqdm

from lewm_liquid_predictors.data import (
    ObservationTrajectory,
    collate_observation_trajectories,
    fit_zscore_normalizer,
    load_pusht_lance_episodes,
    preprocess_observations,
)
from lewm_liquid_predictors.evaluation import evaluate_rollouts
from lewm_liquid_predictors.models import build_lewm_baseline, build_predictor
from lewm_liquid_predictors.models.encoder import LeWMEncoder
from lewm_liquid_predictors.models.protocol import DynamicsPredictor
from lewm_liquid_predictors.models.smoke_encoder import SmokeActionEncoder, build_smoke_encoder
from lewm_liquid_predictors.models.system import PredictorSystem
from lewm_liquid_predictors.models.upstream_encoder import (
    build_upstream_action_encoder,
    build_upstream_encoder,
)
from lewm_liquid_predictors.training import (
    LeWMTrainer,
    PredictorTrainer,
    build_linear_warmup_cosine_scheduler,
    capture_run_provenance,
    initialize_run,
    write_metrics,
)
from lewm_liquid_predictors.utils import ExperimentConfig, load_config


def main(arguments: Sequence[str] | None = None) -> int:
    """Run a project command and return its process exit status."""
    parser = argparse.ArgumentParser(prog="lewm-liquid-predictors")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config", type=Path)

    inspect = subparsers.add_parser("inspect-pusht")
    inspect.add_argument("path")
    inspect.add_argument("--max-episodes", type=int, default=1)

    train = subparsers.add_parser("train")
    train.add_argument("config", type=Path)
    train.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")
    train.add_argument("--max-episodes", type=int, default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")
    evaluate.add_argument("--max-episodes", type=int, default=None)

    train_lewm = subparsers.add_parser("train-lewm")
    train_lewm.add_argument("config", type=Path)
    train_lewm.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")
    train_lewm.add_argument("--max-episodes", type=int, default=None)

    parsed = parser.parse_args(arguments)
    if parsed.command == "validate-config":
        config = load_config(parsed.config)
        print(json.dumps(asdict(config), default=str, indent=2, sort_keys=True))
        return 0
    if parsed.command == "inspect-pusht":
        trajectories = load_pusht_lance_episodes(parsed.path, max_episodes=parsed.max_episodes)
        print(
            json.dumps(
                [
                    {
                        "episode_id": trajectory.episode_id,
                        "observations_shape": list(trajectory.observations.shape),
                        "actions_shape": list(trajectory.actions.shape),
                    }
                    for trajectory in trajectories
                ],
                indent=2,
            )
        )
        return 0
    if parsed.command == "train":
        return _run_train(parsed)
    if parsed.command == "train-lewm":
        return _run_train_lewm(parsed)
    if parsed.command == "evaluate":
        return _run_evaluate(parsed)
    return 1


def _load_trajectories(
    data_path: str, max_episodes: int | None
) -> tuple[ObservationTrajectory, ...]:
    return load_pusht_lance_episodes(data_path, max_episodes=max_episodes)


def _build_system(config: ExperimentConfig, action_input_dim: int) -> PredictorSystem:
    settings = config.model
    latent_dim = settings.latent_dim
    predictor = build_predictor(settings)
    if settings.encoder_mode == "upstream":
        encoder: LeWMEncoder = build_upstream_encoder(latent_dim=latent_dim)
        action_encoder: nn.Module = build_upstream_action_encoder(
            action_input_dim, emb_dim=settings.action_dim
        )
    else:
        encoder = build_smoke_encoder(latent_dim)
        action_encoder = SmokeActionEncoder(action_input_dim, settings.action_dim)
    return PredictorSystem(encoder=encoder, action_encoder=action_encoder, predictor=predictor)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _run_train(parsed: argparse.Namespace) -> int:
    config = load_config(parsed.config)
    seed = config.experiment.seeds[0]
    torch.manual_seed(seed)
    device = _resolve_device(config.training.device)

    trajectories = _load_trajectories(parsed.data_path, parsed.max_episodes)
    if not trajectories:
        raise RuntimeError("no trajectories loaded")
    action_input_dim = trajectories[0].actions.shape[-1]

    system = _build_system(config, action_input_dim).to(device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=5e-5, weight_decay=1e-3)
    trainer = PredictorTrainer(system, optimizer)

    provenance = capture_run_provenance(seed=seed, requested_device=config.training.device)
    run_dir = initialize_run(config.experiment.output_dir, config, provenance)

    batch_size = config.training.batch_size or 8
    batches = [
        collate_observation_trajectories(trajectories[start : start + batch_size]).to(device)
        for start in range(0, len(trajectories), batch_size)
    ]
    for epoch in range(config.training.max_epochs or 1):
        metrics = trainer.train_observation_epoch(batches)
        print(
            f"epoch {epoch + 1}: mse={metrics.mean_squared_error:.6f}"
            f" transitions={metrics.transitions}"
        )
        write_metrics(run_dir, {"epoch": epoch + 1, "train/mse": metrics.mean_squared_error})

    torch.save(system.state_dict(), run_dir / "system.pt")
    print(f"run saved to {run_dir}")
    return 0


def _run_evaluate(parsed: argparse.Namespace) -> int:
    config = load_config(parsed.config)
    seed = config.experiment.seeds[0]
    torch.manual_seed(seed)
    device = _resolve_device(config.training.device)

    trajectories = _load_trajectories(parsed.data_path, parsed.max_episodes)
    if not trajectories:
        raise RuntimeError("no trajectories loaded")
    action_input_dim = trajectories[0].actions.shape[-1]

    system = _build_system(config, action_input_dim).to(device)
    run_dir = config.experiment.output_dir
    checkpoint_path = Path(run_dir) / "system.pt"
    if checkpoint_path.exists():
        system.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"loaded checkpoint from {checkpoint_path}")
    system.eval()

    batch = collate_observation_trajectories(trajectories).to(device)
    latent_batch = system.encode_batch(batch)
    metrics = evaluate_rollouts(
        cast("DynamicsPredictor", system.predictor),
        latent_batch,
        config.evaluation.rollout_horizons,
        config.evaluation.divergence.normalized_error_threshold,
    )
    result = {
        "one_step_normalized_rmse": metrics.one_step_normalized_rmse.item(),
        "rollout_normalized_rmse": {
            str(horizon): error.item() for horizon, error in metrics.rollout_normalized_rmse.items()
        },
        "divergence_rate": metrics.divergence_rate.item(),
    }
    print(json.dumps(result, indent=2))
    return 0


def _run_train_lewm(parsed: argparse.Namespace) -> int:
    """Run the full LeWM baseline reproduction with two-term loss."""
    config = load_config(parsed.config)
    seed = config.experiment.seeds[0]
    torch.manual_seed(seed)
    device = _resolve_device(config.training.device)

    print(f"[lewm] config: {parsed.config}", file=sys.stderr)
    print(f"[lewm] seed: {seed}, device: {device}", file=sys.stderr)
    print(f"[lewm] loading PushT data from {parsed.data_path}...", file=sys.stderr)
    trajectories = _load_trajectories(parsed.data_path, parsed.max_episodes)
    if not trajectories:
        raise RuntimeError("no trajectories loaded")
    action_input_dim = trajectories[0].actions.shape[-1]
    print(
        f"[lewm] loaded {len(trajectories)} episodes, action_dim={action_input_dim}",
        file=sys.stderr,
    )

    print(
        "[lewm] building model (ViT-tiny + projector + ARPredictor + pred_proj + SIGReg)...",
        file=sys.stderr,
    )
    model = build_lewm_baseline(
        latent_dim=config.model.latent_dim,
        action_dim=action_input_dim,
        history_size=config.model.transformer_context_length,
        num_preds=1,
        sigreg_weight=0.09,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[lewm] model parameters: {total_params:,}", file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    max_epochs = config.training.max_epochs or 100
    total_steps = max_epochs * len(trajectories)
    warmup_steps = max(1, int(0.01 * total_steps))
    scheduler = build_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    trainer = LeWMTrainer(model, optimizer, gradient_clip_val=1.0)

    print("[lewm] capturing provenance...", file=sys.stderr)
    provenance = capture_run_provenance(seed=seed, requested_device=config.training.device)
    run_dir = initialize_run(config.experiment.output_dir, config, provenance)
    print(f"[lewm] run directory: {run_dir}", file=sys.stderr)

    batch_size = config.training.batch_size or 128
    img_size = 224
    history_size = config.model.transformer_context_length
    seq_len = history_size + 1  # history_size + num_preds

    print("[lewm] fitting action normalizer...", file=sys.stderr)
    all_actions = torch.cat([t.actions for t in trajectories], dim=0)
    action_normalizer = fit_zscore_normalizer(all_actions)
    action_normalizer = action_normalizer.to(device)

    print("[lewm] creating fixed-length sequence windows...", file=sys.stderr)

    def make_windows(trajs: tuple[ObservationTrajectory, ...]) -> list[dict[str, Tensor]]:
        windows: list[dict[str, Tensor]] = []
        for traj in tqdm(trajs, desc="windows", file=sys.stderr):
            obs = traj.observations
            act = traj.actions
            for start in range(0, obs.shape[0] - seq_len + 1):
                obs_window = obs[start : start + seq_len].unsqueeze(0)
                act_window = act[start : start + seq_len - 1]
                pad = torch.full(
                    (1, 1, act_window.shape[-1]),
                    float("nan"),
                    dtype=act_window.dtype,
                    device=act_window.device,
                )
                act_window = torch.cat([act_window.unsqueeze(0), pad], dim=1)
                pixels = preprocess_observations(obs_window, img_size=img_size).to(device)
                actions = action_normalizer(act_window)
                actions = torch.nan_to_num(actions, 0.0).to(device)
                windows.append({"pixels": pixels, "action": actions})
        return windows

    all_windows = make_windows(trajectories)
    print(f"[lewm] {len(all_windows)} training windows, batch_size={batch_size}", file=sys.stderr)
    num_batches = (len(all_windows) + batch_size - 1) // batch_size

    print(
        f"[lewm] starting training: {max_epochs} epochs, {num_batches} batches/epoch",
        file=sys.stderr,
    )
    epoch_pbar = tqdm(range(max_epochs), desc="epochs", file=sys.stderr)

    for epoch in epoch_pbar:
        train_batches: list[dict[str, Tensor]] = []
        for start in range(0, len(all_windows), batch_size):
            chunk = all_windows[start : start + batch_size]
            batch_pixels = torch.cat([w["pixels"] for w in chunk], dim=0)
            batch_actions = torch.cat([w["action"] for w in chunk], dim=0)
            train_batches.append({"pixels": batch_pixels, "action": batch_actions})

        metrics = trainer.train_epoch(train_batches)
        scheduler.step()
        epoch_pbar.set_postfix(
            loss=f"{metrics.total_loss:.4f}",
            pred=f"{metrics.pred_loss:.4f}",
            sigreg=f"{metrics.sigreg_loss:.4f}",
            lr=f"{metrics.learning_rate:.2e}",
        )
        write_metrics(
            run_dir,
            {
                "epoch": epoch + 1,
                "train/loss": metrics.total_loss,
                "train/pred_loss": metrics.pred_loss,
                "train/sigreg_loss": metrics.sigreg_loss,
                "train/lr": metrics.learning_rate,
            },
        )

    torch.save(model.state_dict(), run_dir / "lewm.pt")
    print(f"LeWM baseline run saved to {run_dir}")
    return 0
