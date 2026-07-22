"""Executable entry points for configuration, data, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Generator, Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from tqdm import tqdm

from lewm_liquid_predictors.data import (
    ObservationTrajectory,
    ObservationTrajectoryBatch,
    collate_observation_trajectories,
    load_pusht_lance_episodes,
)
from lewm_liquid_predictors.data.preprocessing import (
    ZScoreNormalizer,
    fit_zscore_normalizer,
    preprocess_observations,
)
from lewm_liquid_predictors.data.splits import (
    create_split_manifest,
    sample_episode_ids,
    write_split_manifest,
)
from lewm_liquid_predictors.evaluation import divergence_times, evaluate_rollouts
from lewm_liquid_predictors.models import (
    build_lewm_baseline,
    build_predictor,
    teacher_forced_rollout,
)
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
from lewm_liquid_predictors.utils import ExperimentConfig, PredictorVariant, load_config


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

    screen = subparsers.add_parser("screen")
    screen.add_argument("config", type=Path)
    screen.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")

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
    if parsed.command == "screen":
        return _run_screen(parsed)
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


def _run_screen(parsed: argparse.Namespace) -> int:
    """Run the fixed-budget, held-out predictor comparison on H200."""
    from torch.utils.data import DataLoader, Dataset

    config = load_config(parsed.config)
    if config.model.encoder_mode != "upstream":
        raise ValueError("screen requires model.encoder_mode: upstream")
    trajectories = _load_trajectories(parsed.data_path, max_episodes=None)
    if not trajectories:
        raise RuntimeError("no trajectories loaded")
    episode_ids = tuple(trajectory.episode_id for trajectory in trajectories)
    manifest = create_split_manifest(
        config.data.dataset, episode_ids, seed=config.experiment.seeds[0]
    )
    output_root = config.experiment.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    write_split_manifest(manifest, output_root / "split_manifest.json")
    by_id = {trajectory.episode_id: trajectory for trajectory in trajectories}
    train_ids = sample_episode_ids(manifest.train, config.data.fraction, manifest.seed)
    train_trajectories = tuple(by_id[episode_id] for episode_id in train_ids)
    test_trajectories = tuple(by_id[episode_id] for episode_id in manifest.test)
    action_normalizer = fit_zscore_normalizer(
        torch.cat([item.actions for item in train_trajectories])
    )
    action_input_dim = train_trajectories[0].actions.shape[-1]
    window_size = config.model.transformer_context_length + 1

    class WindowDataset(Dataset[ObservationTrajectory]):
        def __init__(self, items: tuple[ObservationTrajectory, ...]) -> None:
            self.windows = tuple(
                (item, start)
                for item in items
                for start in range(item.actions.shape[0] - window_size + 2)
            )
            if not self.windows:
                raise ValueError("training trajectories are shorter than the context window")

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int) -> ObservationTrajectory:
            trajectory, start = self.windows[index]
            return ObservationTrajectory(
                episode_id=trajectory.episode_id,
                observations=trajectory.observations[start : start + window_size],
                actions=trajectory.actions[start : start + window_size - 1],
            )

    train_dataset = WindowDataset(train_trajectories)
    variants: tuple[PredictorVariant, ...] = ("lewm_ar", "mlp", "transformer", "cfc", "ltc")
    for seed in config.experiment.seeds:
        for variant in variants:
            torch.manual_seed(seed)
            device = _resolve_device(config.training.device)
            run_config = replace(
                config,
                experiment=replace(
                    config.experiment,
                    name=f"{config.experiment.name}-{variant}",
                    output_dir=output_root / variant,
                    seeds=(seed,),
                ),
                model=replace(config.model, variant=variant),
            )
            system = _build_system(run_config, action_input_dim).to(device)
            optimizer = torch.optim.AdamW(system.parameters(), lr=5e-5, weight_decay=1e-3)
            epochs = run_config.training.max_epochs or 1
            scheduler = build_linear_warmup_cosine_scheduler(
                optimizer, max(1, int(0.01 * epochs)), epochs
            )
            trainer = PredictorTrainer(system, optimizer, gradient_clip_val=1.0)
            run_dir = initialize_run(
                run_config.experiment.output_dir,
                run_config,
                capture_run_provenance(seed=seed, requested_device=run_config.training.device),
            )
            history: list[dict[str, float]] = []
            for epoch in tqdm(range(epochs), desc=f"{variant}/seed{seed}", file=sys.stderr):
                generator = torch.Generator().manual_seed(seed + epoch)
                loader = DataLoader(
                    train_dataset,
                    batch_size=run_config.training.batch_size or 128,
                    shuffle=True,
                    generator=generator,
                    num_workers=run_config.data.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=collate_observation_trajectories,
                )
                metrics = trainer.train_observation_epoch(
                    _prepared_batches(loader, action_normalizer, device)
                )
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "train/mse": metrics.mean_squared_error,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )
                scheduler.step()
            test_metrics = _evaluate_screen(
                system,
                test_trajectories,
                action_normalizer,
                run_config,
                device,
            )
            parameter_count = sum(parameter.numel() for parameter in system.predictor.parameters())
            result = {**history[-1], **test_metrics, "predictor/parameters": float(parameter_count)}
            write_metrics(run_dir, result)
            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            torch.save(system.state_dict(), run_dir / "system.pt")
            print(json.dumps({"variant": variant, "seed": seed, **result}, sort_keys=True))
    return 0


def _prepared_batches(
    batches: Iterable[ObservationTrajectoryBatch],
    action_normalizer: ZScoreNormalizer,
    device: torch.device,
) -> Generator[ObservationTrajectoryBatch, None, None]:
    """Apply common CPU preprocessing before moving a batch to the accelerator."""
    for batch in batches:
        observations = _channels_first(batch.observations)
        prepared = ObservationTrajectoryBatch(
            episode_ids=batch.episode_ids,
            observations=preprocess_observations(observations),
            actions=torch.nan_to_num(action_normalizer(batch.actions), 0.0),
            transition_mask=batch.transition_mask,
        )
        yield prepared.to(device)


def _channels_first(observations: Tensor) -> Tensor:
    """Accept either CHW or HWC source pixels while enforcing the encoder layout."""
    if observations.shape[-3] == 3:
        return observations
    if observations.shape[-1] == 3:
        return observations.movedim(-1, -3)
    raise ValueError("pixels must be RGB in CHW or HWC layout")


@torch.no_grad()
def _evaluate_screen(
    system: PredictorSystem,
    trajectories: tuple[ObservationTrajectory, ...],
    action_normalizer: ZScoreNormalizer,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    """Aggregate held-out normalized errors over complete episodes."""
    system.eval()
    predictor = cast(DynamicsPredictor, system.predictor)
    horizons = config.evaluation.rollout_horizons
    error_sums = {"one_step": 0.0, **{str(horizon): 0.0 for horizon in horizons}}
    target_sums = {key: 0.0 for key in error_sums}
    divergences = 0
    episodes = 0
    for trajectory in trajectories:
        batch = collate_observation_trajectories([trajectory])
        prepared = next(_prepared_batches(iter([batch]), action_normalizer, device))
        latent_batch = system.encode_batch(prepared)
        targets = latent_batch.latents[:, 1:]
        teacher_forced = teacher_forced_rollout(
            predictor, latent_batch.latents, latent_batch.actions
        )
        rollout, _ = predictor.rollout(latent_batch.latents[:, 0], latent_batch.actions)
        _accumulate_error(
            error_sums,
            target_sums,
            "one_step",
            teacher_forced,
            targets,
            latent_batch.transition_mask,
        )
        for horizon in horizons:
            if horizon <= rollout.shape[1]:
                _accumulate_error(
                    error_sums,
                    target_sums,
                    str(horizon),
                    rollout[:, horizon - 1 : horizon],
                    targets[:, horizon - 1 : horizon],
                    latent_batch.transition_mask[:, horizon - 1 : horizon],
                )
        divergence = divergence_times(
            rollout,
            targets,
            latent_batch.transition_mask,
            config.evaluation.divergence.normalized_error_threshold,
        )
        divergences += int((divergence >= 0).sum().item())
        episodes += 1
    metrics = {
        "test/one_step_normalized_rmse": _normalized_error(
            error_sums["one_step"], target_sums["one_step"]
        ),
        "test/divergence_rate": divergences / episodes,
    }
    metrics.update(
        {
            f"test/rollout_normalized_rmse/{horizon}": _normalized_error(
                error_sums[str(horizon)], target_sums[str(horizon)]
            )
            for horizon in horizons
            if target_sums[str(horizon)] > 0
        }
    )
    return metrics


def _accumulate_error(
    error_sums: dict[str, float],
    target_sums: dict[str, float],
    key: str,
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> None:
    expanded_mask = mask.unsqueeze(-1)
    error_sums[key] += float(((predictions - targets).square() * expanded_mask).sum().item())
    target_sums[key] += float((targets.square() * expanded_mask).sum().item())


def _normalized_error(error_sum: float, target_sum: float) -> float:
    if target_sum <= 0:
        raise ValueError("held-out data has no valid target energy")
    return float((error_sum / target_sum) ** 0.5)


def _run_train_lewm(parsed: argparse.Namespace) -> int:
    """Run the full LeWM baseline reproduction with two-term loss."""
    from importlib import import_module
    from pathlib import Path as P

    from torch.utils.data import DataLoader, Subset

    from lewm_liquid_predictors.data.preprocessing import (
        ZScoreNormalizer,
        normalize_pixels,
        resize_observations,
    )

    config = load_config(parsed.config)
    seed = config.experiment.seeds[0]
    torch.manual_seed(seed)
    device = _resolve_device(config.training.device)

    print(f"[lewm] config: {parsed.config}", file=sys.stderr)
    print(f"[lewm] seed: {seed}, device: {device}", file=sys.stderr)

    batch_size = config.training.batch_size or 128
    img_size = 224
    history_size = config.model.transformer_context_length
    seq_len = history_size + 1
    num_workers = config.data.num_workers

    print(f"[lewm] loading PushT dataset from {parsed.data_path}...", file=sys.stderr)
    dataset_path = str(parsed.data_path)
    if "://" not in dataset_path:
        dataset_path = str(P(dataset_path).resolve())
    import os

    cache_dir = os.environ.get("STABLEWM_HOME", str(P(parsed.data_path).resolve().parent))
    swm: object = import_module("stable_worldmodel")
    dataset = swm.data.load_dataset(  # type: ignore[attr-defined]
        dataset_path,
        frameskip=config.data.frameskip,
        num_steps=seq_len,
        keys_to_load=["pixels", "action"],
        cache_dir=cache_dir,
    )
    print(f"[lewm] dataset: {len(dataset)} windows", file=sys.stderr)

    action_dim = config.data.frameskip * dataset.get_dim("action")
    print(f"[lewm] action_dim: {action_dim}", file=sys.stderr)

    if config.data.fraction < 1:
        window_count = max(1, round(len(dataset) * config.data.fraction))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:window_count].tolist()
        dataset = Subset(dataset, indices)
        print(
            f"[lewm] using deterministic {config.data.fraction:.1%} subset: {len(dataset)} windows",
            file=sys.stderr,
        )

    print("[lewm] fitting action normalizer...", file=sys.stderr)
    sample = dataset[0]
    sample_action = sample["action"]
    if sample_action.ndim == 1:
        sample_action = sample_action.unsqueeze(0)
    action_dim_in_batch = sample_action.shape[-1]
    all_actions = []
    for i in range(min(len(dataset), 1000)):
        act = dataset[i]["action"]
        if act.ndim == 1:
            act = act.unsqueeze(0)
        all_actions.append(act.reshape(-1, action_dim_in_batch))
    action_tensor = torch.cat(all_actions, dim=0)
    action_tensor = action_tensor[~torch.isnan(action_tensor).any(dim=1)]
    mean = action_tensor.mean(0, keepdim=True).clone()
    std = action_tensor.std(0, keepdim=True).clone()
    action_normalizer = ZScoreNormalizer(mean, std)
    print(
        f"[lewm] action normalizer fitted on {action_tensor.shape[0]}"
        f" samples, dim={action_dim_in_batch}",
        file=sys.stderr,
    )

    print(
        "[lewm] building model (ViT-tiny + projector + ARPredictor + pred_proj + SIGReg)...",
        file=sys.stderr,
    )
    model = build_lewm_baseline(
        latent_dim=config.model.latent_dim,
        action_dim=action_dim,
        history_size=history_size,
        num_preds=1,
        sigreg_weight=0.09,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[lewm] model parameters: {total_params:,}", file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    max_epochs = config.training.max_epochs or 100
    warmup_epochs = max(1, int(0.01 * max_epochs))
    scheduler = build_linear_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs)

    trainer = LeWMTrainer(model, optimizer, gradient_clip_val=1.0)

    print("[lewm] capturing provenance...", file=sys.stderr)
    provenance = capture_run_provenance(seed=seed, requested_device=config.training.device)
    run_dir = initialize_run(config.experiment.output_dir, config, provenance)
    print(f"[lewm] run directory: {run_dir}", file=sys.stderr)

    def transform_batch(raw: dict[str, Tensor]) -> dict[str, Tensor]:
        pixels = raw["pixels"]
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        pixels = normalize_pixels(resize_observations(pixels, img_size))
        actions = action_normalizer(raw["action"])
        actions = torch.nan_to_num(actions, 0.0)
        return {"pixels": pixels.to(device), "action": actions.to(device)}

    print(
        f"[lewm] dataloader: batch_size={batch_size}, num_workers={num_workers}",
        file=sys.stderr,
    )
    print(f"[lewm] starting training: {max_epochs} epochs", file=sys.stderr)

    epoch_pbar = tqdm(range(max_epochs), desc="epochs", file=sys.stderr)

    for epoch in epoch_pbar:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

        def batch_iter(dl: DataLoader[object] = dataloader) -> Generator[dict[str, Tensor]]:
            for raw in dl:
                yield transform_batch(raw)

        metrics = trainer.train_epoch(batch_iter(), total_batches=len(dataloader))
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
    print(f"[lewm] baseline run saved to {run_dir}", file=sys.stderr)
    return 0
