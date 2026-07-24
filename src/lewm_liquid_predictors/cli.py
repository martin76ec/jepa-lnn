"""Executable entry points for configuration, data, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Generator, Iterable, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lewm_liquid_predictors.data import (
    ObservationTrajectory,
    ObservationTrajectoryBatch,
    ScreeningData,
    collate_observation_trajectories,
    episode_index,
    load_pusht_lance_episodes,
    materialize_training_windows,
    prepare_observation_batch,
    prepare_screening_data,
)
from lewm_liquid_predictors.data.preprocessing import ZScoreNormalizer
from lewm_liquid_predictors.data.pusht import EpisodeSource
from lewm_liquid_predictors.decoder import (
    render_decoder_galleries,
    train_decoder,
)
from lewm_liquid_predictors.evaluation import (
    EpisodePredictions,
    HeldOutEvaluation,
    evaluate_rollouts,
    evaluate_screen_split,
    write_retrieval_galleries,
)
from lewm_liquid_predictors.models import (
    LeWMJEPA,
    LeWMPredictorView,
    build_lewm_baseline,
    build_predictor,
    teacher_forced_rollout,
)
from lewm_liquid_predictors.models.checkpoint_adapters import (
    OFFICIAL_LEWM_PUSHT,
    load_official_lewm,
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
    evaluate.add_argument("--checkpoint", type=Path, default=None)

    train_lewm = subparsers.add_parser("train-lewm")
    train_lewm.add_argument("config", type=Path)
    train_lewm.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")
    train_lewm.add_argument("--max-episodes", type=int, default=None)

    screen = subparsers.add_parser("screen")
    screen.add_argument("config", type=Path)
    screen.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")
    screen.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/lewm-pusht/weights.pt")
    )

    evaluate_lewm = subparsers.add_parser("evaluate-lewm-official")
    evaluate_lewm.add_argument("config", type=Path)
    evaluate_lewm.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")
    evaluate_lewm.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/lewm-pusht/weights.pt")
    )
    evaluate_lewm.add_argument("--max-test-episodes", type=int, default=None)

    train_decoder_parser = subparsers.add_parser("train-decoder")
    train_decoder_parser.add_argument("config", type=Path)
    train_decoder_parser.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")
    train_decoder_parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/lewm-pusht/weights.pt")
    )
    train_decoder_parser.add_argument("--max-train-episodes", type=int, default=None)
    train_decoder_parser.add_argument("--max-frames", type=int, default=None)

    render_decoder_parser = subparsers.add_parser("render-decoder-galleries")
    render_decoder_parser.add_argument("config", type=Path)
    render_decoder_parser.add_argument("--data-path", default="data/raw/pusht_expert_train.lance")
    render_decoder_parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/lewm-pusht/weights.pt")
    )
    render_decoder_parser.add_argument(
        "--predictor-root", type=Path, default=Path("runs/h200-screen")
    )
    render_decoder_parser.add_argument("--decoder-checkpoint", type=Path, default=None)
    render_decoder_parser.add_argument("--max-predictor-runs", type=int, default=None)
    render_decoder_parser.add_argument("--max-gallery-episodes", type=int, default=None)

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
    if parsed.command == "evaluate-lewm-official":
        return _run_evaluate_lewm_official(parsed)
    if parsed.command == "train-decoder":
        run_dir = train_decoder(
            parsed.config,
            parsed.data_path,
            parsed.checkpoint,
            max_train_episodes=parsed.max_train_episodes,
            max_frames=parsed.max_frames,
        )
        print(json.dumps({"decoder_run": str(run_dir)}, sort_keys=True))
        return 0
    if parsed.command == "render-decoder-galleries":
        output_dir = render_decoder_galleries(
            parsed.config,
            parsed.data_path,
            parsed.checkpoint,
            parsed.predictor_root,
            decoder_checkpoint=parsed.decoder_checkpoint,
            max_predictor_runs=parsed.max_predictor_runs,
            max_gallery_episodes=parsed.max_gallery_episodes,
        )
        print(json.dumps({"decoded_galleries": str(output_dir)}, sort_keys=True))
        return 0
    return 1


def _load_trajectories(
    data_path: str, max_episodes: int | None
) -> tuple[ObservationTrajectory, ...]:
    return load_pusht_lance_episodes(data_path, max_episodes=max_episodes)


def _build_system(
    config: ExperimentConfig,
    action_input_dim: int,
    shared_lewm: LeWMJEPA | None = None,
) -> PredictorSystem:
    settings = config.model
    latent_dim = settings.latent_dim
    encoder: LeWMEncoder
    action_encoder: nn.Module
    if shared_lewm is not None:
        encoder = deepcopy(shared_lewm.encoder)
        action_encoder = deepcopy(shared_lewm.action_encoder)
    elif settings.encoder_mode == "upstream":
        encoder = build_upstream_encoder(latent_dim=latent_dim)
        action_encoder = build_upstream_action_encoder(
            action_input_dim, emb_dim=settings.action_dim
        )
    else:
        encoder = build_smoke_encoder(latent_dim)
        action_encoder = SmokeActionEncoder(action_input_dim, settings.action_dim)
    system = PredictorSystem(
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=build_predictor(settings),
    )
    if shared_lewm is not None:
        system.freeze_shared_modules()
    return system


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _prepare_screen_data(config: ExperimentConfig, data_path: str) -> ScreeningData:
    """Prepare the validated split and exact upstream action statistics."""
    print("[screen] opening PushT dataset...", file=sys.stderr, flush=True)
    return prepare_screening_data(
        data_path,
        dataset=config.data.dataset,
        manifest_path=config.experiment.output_dir / "split_manifest.json",
        split_seed=config.experiment.seeds[0],
        frameskip=config.data.frameskip,
        training_fraction=config.data.fraction,
    )


def _load_official_lewm(checkpoint_path: Path, action_input_dim: int) -> tuple[LeWMJEPA, str]:
    """Load and verify the pinned official PushT LeWM checkpoint on CPU."""
    try:
        model = load_official_lewm(checkpoint_path, action_input_dim)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"official LeWM checkpoint not found: {checkpoint_path}; "
            "run `make download-lewm-checkpoint`"
        ) from error
    return model, OFFICIAL_LEWM_PUSHT.sha256


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
    checkpoint_path = parsed.checkpoint
    if checkpoint_path is None:
        candidates = sorted(config.experiment.output_dir.glob("run_*/system.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"no trained system checkpoint found under {config.experiment.output_dir}"
            )
        checkpoint_path = candidates[-1]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"system checkpoint not found: {checkpoint_path}")
    system.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"loaded checkpoint from {checkpoint_path}")
    system.eval()

    batch = collate_observation_trajectories(trajectories).to(device)
    latent_batch = system.encode_batch(batch)
    metrics = evaluate_rollouts(
        cast("DynamicsPredictor", system.predictor),
        latent_batch,
        config.evaluation.rollout_horizons,
        config.evaluation.divergence.normalized_error_threshold,
        config.evaluation.divergence.count_non_finite_as_divergence,
    )
    result = {
        "one_step_normalized_mse": metrics.one_step_normalized_mse.item(),
        "one_step_normalized_rmse": metrics.one_step_normalized_rmse.item(),
        "rollout_normalized_mse": {
            str(horizon): error.item() for horizon, error in metrics.rollout_normalized_mse.items()
        },
        "rollout_normalized_rmse": {
            str(horizon): error.item() for horizon, error in metrics.rollout_normalized_rmse.items()
        },
        "divergence_rate": metrics.divergence_rate.item(),
    }
    print(json.dumps(result, indent=2))
    return 0


def _run_screen(parsed: argparse.Namespace) -> int:
    """Run the fixed-budget, held-out predictor comparison on H200."""
    config = load_config(parsed.config)
    if config.model.encoder_mode != "upstream":
        raise ValueError("screen requires model.encoder_mode: upstream")
    if not config.experiment.variants:
        raise ValueError("screen requires experiment.variants")
    torch.use_deterministic_algorithms(config.training.deterministic)
    print(f"[screen] config: {parsed.config}", file=sys.stderr, flush=True)
    screen_data = _prepare_screen_data(config, parsed.data_path)
    train_dataset = materialize_training_windows(
        screen_data, config.model.transformer_context_length + 1
    )
    test_indices = tuple(episode_index(episode_id) for episode_id in screen_data.test_episode_ids)
    shared_lewm, checkpoint_digest = _load_official_lewm(
        parsed.checkpoint, screen_data.action_input_dim
    )
    output_root = config.experiment.output_dir
    print(
        f"[screen] episodes: train={len(screen_data.train_episode_ids)}, "
        f"test={len(test_indices)}; train_windows={len(train_dataset)}; "
        f"variants={','.join(config.experiment.variants)}; shared_encoder=frozen-official",
        file=sys.stderr,
        flush=True,
    )
    for seed in config.experiment.seeds:
        for variant in config.experiment.variants:
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
            system = _build_system(
                run_config, screen_data.action_input_dim, shared_lewm=shared_lewm
            ).to(device)
            optimizer = torch.optim.AdamW(
                (parameter for parameter in system.parameters() if parameter.requires_grad),
                lr=5e-5,
                weight_decay=1e-3,
            )
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
            _write_data_protocol(
                run_dir,
                screen_data,
                checkpoint_digest=checkpoint_digest,
                evaluation_scope="transductive_fixed_official_latent_space",
            )
            print(
                f"[screen] variant={variant}, seed={seed}, run={run_dir}",
                file=sys.stderr,
                flush=True,
            )
            history: list[dict[str, float]] = []
            epoch_pbar = tqdm(range(epochs), desc=f"{variant}/seed{seed}", file=sys.stderr)
            for epoch in epoch_pbar:
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
                    _prepared_batches(loader, screen_data.action_normalizer, device),
                    total_batches=len(loader),
                )
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "train/mse": metrics.mean_squared_error,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )
                scheduler.step()
                epoch_pbar.set_postfix(mse=f"{metrics.mean_squared_error:.5f}")
                print(
                    f"[screen] variant={variant}, seed={seed}, epoch={epoch + 1}/{epochs}, "
                    f"mse={metrics.mean_squared_error:.6f}",
                    file=sys.stderr,
                    flush=True,
                )
            print("[screen] evaluating held-out test episodes...", file=sys.stderr, flush=True)
            evaluation = _evaluate_screen(
                system,
                screen_data.source,
                test_indices,
                screen_data.action_normalizer,
                run_config,
                device,
            )
            parameter_count = sum(parameter.numel() for parameter in system.predictor.parameters())
            result = {
                **history[-1],
                **evaluation.metrics,
                "predictor/parameters": float(parameter_count),
            }
            write_metrics(run_dir, result)
            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            torch.save(system.state_dict(), run_dir / "system.pt")
            _render_retrieval_artifacts(run_dir, evaluation, screen_data.source, run_config)
            print(json.dumps({"variant": variant, "seed": seed, **result}, sort_keys=True))
    return 0


def _run_evaluate_lewm_official(parsed: argparse.Namespace) -> int:
    """Evaluate the released LeWM-JEPA checkpoint as an in-dataset reference."""
    config = load_config(parsed.config)
    checkpoint_path = parsed.checkpoint
    print(f"[lewm-official] config: {parsed.config}", file=sys.stderr, flush=True)
    screen_data = _prepare_screen_data(config, parsed.data_path)
    if parsed.max_test_episodes is not None and parsed.max_test_episodes <= 0:
        raise ValueError("--max-test-episodes must be positive")
    test_indices = tuple(episode_index(episode_id) for episode_id in screen_data.test_episode_ids)[
        : parsed.max_test_episodes
    ]
    device = _resolve_device(config.training.device)
    model, checkpoint_digest = _load_official_lewm(checkpoint_path, screen_data.action_input_dim)
    model = model.to(device)

    split_seed = screen_data.manifest.seed
    run_config = replace(
        config,
        experiment=replace(
            config.experiment,
            name="h200-screen-lewm-official",
            output_dir=config.experiment.output_dir / "lewm_full",
            seeds=(split_seed,),
        ),
    )
    run_dir = initialize_run(
        run_config.experiment.output_dir,
        run_config,
        capture_run_provenance(seed=split_seed, requested_device=run_config.training.device),
    )
    _write_data_protocol(
        run_dir,
        screen_data,
        checkpoint_digest=checkpoint_digest,
        evaluation_scope="in_dataset_pretrained_reference",
        max_test_episodes=parsed.max_test_episodes,
    )
    print(
        f"[lewm-official] checkpoint loaded; evaluating {len(test_indices)} "
        f"screen-split episodes as an in-dataset reference; run={run_dir}",
        file=sys.stderr,
        flush=True,
    )
    evaluation = _evaluate_lewm_model(
        model,
        screen_data.source,
        test_indices,
        screen_data.action_normalizer,
        run_config,
        device,
    )
    metrics = evaluation.metrics
    metrics["model/parameters"] = float(sum(parameter.numel() for parameter in model.parameters()))
    metrics["predictor/parameters"] = float(
        sum(parameter.numel() for parameter in model.predictor.parameters())
        + sum(parameter.numel() for parameter in model.pred_proj.parameters())
    )
    write_metrics(run_dir, metrics)
    checkpoint_metadata = {
        "repository": OFFICIAL_LEWM_PUSHT.repository,
        "revision": OFFICIAL_LEWM_PUSHT.revision,
        "sha256": checkpoint_digest,
        "upstream_training_seed": OFFICIAL_LEWM_PUSHT.training_seed,
        "screen_split_seed": split_seed,
        "objective": "pred_loss + 0.09 * sigreg_loss",
    }
    (run_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _render_retrieval_artifacts(run_dir, evaluation, screen_data.source, run_config)
    print(json.dumps({"variant": "lewm_full", **metrics}, sort_keys=True))
    return 0


@torch.no_grad()
def _evaluate_lewm_model(
    model: LeWMJEPA,
    source: EpisodeSource,
    test_indices: tuple[int, ...],
    action_normalizer: ZScoreNormalizer,
    config: ExperimentConfig,
    device: torch.device,
) -> HeldOutEvaluation:
    """Evaluate the exact LeWM prediction branch with shared held-out metrics."""
    model.eval()
    predictor = LeWMPredictorView(model)
    dynamics_predictor = cast(DynamicsPredictor, predictor)

    def predict_episode(prepared: ObservationTrajectoryBatch) -> EpisodePredictions:
        encoded = model.encode({"pixels": prepared.observations, "action": prepared.actions})
        latents = encoded["emb"]
        action_embeddings = encoded["act_emb"]
        teacher_forced = teacher_forced_rollout(dynamics_predictor, latents, action_embeddings)
        rollout, _ = predictor.rollout(
            latents[:, 0],
            action_embeddings,
        )
        return EpisodePredictions(latents, teacher_forced, rollout)

    return evaluate_screen_split(
        source,
        test_indices,
        action_normalizer,
        frameskip=config.data.frameskip,
        horizons=config.evaluation.rollout_horizons,
        divergence_threshold=config.evaluation.divergence.normalized_error_threshold,
        count_non_finite_as_divergence=(
            config.evaluation.divergence.count_non_finite_as_divergence
        ),
        device=device,
        predict_episode=predict_episode,
    )


def _write_data_protocol(
    run_dir: Path,
    data: ScreeningData,
    *,
    checkpoint_digest: str,
    evaluation_scope: str,
    max_test_episodes: int | None = None,
) -> None:
    """Persist split, normalization, checkpoint, and runtime evaluation identity."""
    (run_dir / "split_manifest.json").write_text(
        data.manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    metadata = {
        "dataset": data.manifest.dataset,
        "dataset_source": data.dataset_source,
        "dataset_fingerprint": data.dataset_fingerprint,
        "split_seed": data.manifest.seed,
        "split_manifest_sha256": data.manifest_digest,
        "selected_train_episode_count": len(data.train_episode_ids),
        "test_episode_count": len(data.test_episode_ids),
        "max_test_episodes": max_test_episodes,
        "action_normalization": {
            "policy": "full_raw_action_column_before_frameskip",
            "sample_count": data.action_statistics.sample_count,
            "raw_mean": data.action_statistics.mean.tolist(),
            "raw_std": data.action_statistics.std.tolist(),
        },
        "shared_encoder": {
            "repository": OFFICIAL_LEWM_PUSHT.repository,
            "revision": OFFICIAL_LEWM_PUSHT.revision,
            "checkpoint_sha256": checkpoint_digest,
            "config_sha256": OFFICIAL_LEWM_PUSHT.config_sha256,
            "frozen": True,
        },
        "evaluation_scope": evaluation_scope,
    }
    (run_dir / "data_protocol.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _render_retrieval_artifacts(
    run_dir: Path,
    evaluation: HeldOutEvaluation,
    source: EpisodeSource,
    config: ExperimentConfig,
) -> None:
    """Render optional retrieval artifacts after scientific metrics are durable."""
    try:
        write_retrieval_galleries(
            run_dir,
            evaluation.gallery_queries,
            evaluation.retrieval_latents,
            evaluation.retrieval_references,
            source,
            config.data.frameskip,
            config.evaluation.rollout_horizons,
        )
    except (ImportError, OSError, ValueError) as error:
        (run_dir / "retrieval_error.json").write_text(
            json.dumps({"error": str(error)}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[evaluation] retrieval rendering failed: {error}", file=sys.stderr, flush=True)


def _prepared_batches(
    batches: Iterable[ObservationTrajectoryBatch],
    action_normalizer: ZScoreNormalizer,
    device: torch.device,
) -> Generator[ObservationTrajectoryBatch, None, None]:
    """Apply common CPU preprocessing before moving a batch to the accelerator."""
    for batch in batches:
        yield prepare_observation_batch(batch, action_normalizer, device)


@torch.no_grad()
def _evaluate_screen(
    system: PredictorSystem,
    source: EpisodeSource,
    test_indices: tuple[int, ...],
    action_normalizer: ZScoreNormalizer,
    config: ExperimentConfig,
    device: torch.device,
) -> HeldOutEvaluation:
    """Aggregate held-out normalized errors over complete episodes."""
    system.eval()
    predictor = cast(DynamicsPredictor, system.predictor)

    def predict_episode(prepared: ObservationTrajectoryBatch) -> EpisodePredictions:
        latent_batch = system.encode_batch(prepared)
        teacher_forced = teacher_forced_rollout(
            predictor, latent_batch.latents, latent_batch.actions
        )
        rollout, _ = predictor.rollout(latent_batch.latents[:, 0], latent_batch.actions)
        return EpisodePredictions(latent_batch.latents, teacher_forced, rollout)

    return evaluate_screen_split(
        source,
        test_indices,
        action_normalizer,
        frameskip=config.data.frameskip,
        horizons=config.evaluation.rollout_horizons,
        divergence_threshold=config.evaluation.divergence.normalized_error_threshold,
        count_non_finite_as_divergence=(
            config.evaluation.divergence.count_non_finite_as_divergence
        ),
        device=device,
        predict_episode=predict_episode,
    )


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
