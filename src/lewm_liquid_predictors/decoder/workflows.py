"""Decoder-only training and inference workflows over existing predictor runs."""

from __future__ import annotations

import json
import math
import shutil
import sys
from collections.abc import Sequence
from copy import deepcopy
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from lewm_liquid_predictors.data import (
    EpisodeSource,
    ObservationTrajectory,
    ZScoreNormalizer,
    collate_observation_trajectories,
    episode_index,
    load_source_trajectory,
    load_split_manifest,
    open_pusht_lance_source,
    prepare_observation_batch,
    sample_episode_ids,
)
from lewm_liquid_predictors.models import PredictorSystem, build_predictor
from lewm_liquid_predictors.models.checkpoint_adapters import load_official_lewm
from lewm_liquid_predictors.models.lewm import LeWMJEPA
from lewm_liquid_predictors.models.protocol import DynamicsPredictor
from lewm_liquid_predictors.training import (
    build_linear_warmup_cosine_scheduler,
    capture_run_provenance,
    write_metrics,
)
from lewm_liquid_predictors.utils import ExperimentConfig, load_config

from .checkpoint import (
    DECODER_CHECKPOINT_FILENAME,
    file_sha256,
    initialize_decoder_run,
    load_decoder_checkpoint,
    save_decoder_checkpoint,
    write_source_checkpoints,
)
from .config import DecoderConfig, load_decoder_config
from .galleries import write_decoded_galleries
from .model import CrossAttentionDecoder
from .training import DecoderTrainer, FrameDataset


def train_decoder(
    config_path: str | Path,
    data_path: str | Path,
    official_checkpoint: str | Path,
    *,
    max_train_episodes: int | None = None,
    max_frames: int | None = None,
) -> Path:
    """Train one post-hoc decoder without optimizing any predictor parameter."""
    config = load_decoder_config(config_path)
    _validate_optional_limit(max_train_episodes, "max_train_episodes")
    _validate_optional_limit(max_frames, "max_frames")
    torch.manual_seed(config.experiment.seed)
    torch.use_deterministic_algorithms(config.training.deterministic)
    device = _resolve_device(config.training.device)

    source = open_pusht_lance_source(
        data_path,
        config.data.frameskip,
        keys_to_load=("pixels", "action"),
    )
    manifest = load_split_manifest(config.data.split_manifest)
    source_episode_ids = tuple(f"episode-{index:06d}" for index in range(len(source.lengths)))
    manifest.validate(config.data.dataset, manifest.seed, source_episode_ids)
    train_episode_ids = sample_episode_ids(
        manifest.train,
        config.data.fraction,
        manifest.seed,
    )
    if max_train_episodes is not None:
        train_episode_ids = train_episode_ids[:max_train_episodes]
    trajectories = tuple(
        load_source_trajectory(source, episode_index(episode_id), config.data.frameskip)
        for episode_id in tqdm(
            train_episode_ids,
            desc="decoder episodes",
            file=sys.stderr,
        )
    )
    frame_dataset = FrameDataset(trajectories)
    frames: Dataset[Tensor] = frame_dataset
    training_frame_count = len(frame_dataset)
    if max_frames is not None:
        training_frame_count = min(max_frames, training_frame_count)
        frames = Subset(frame_dataset, range(training_frame_count))

    action_input_dim = trajectories[0].actions.shape[-1]
    official_model = load_official_lewm(official_checkpoint, action_input_dim)
    encoder = deepcopy(official_model.encoder).to(device)
    decoder = CrossAttentionDecoder(config.architecture).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    steps_per_epoch = math.ceil(training_frame_count / config.training.batch_size)
    total_steps = config.training.epochs * steps_per_epoch
    warmup_steps = max(1, int(0.01 * total_steps)) if total_steps > 1 else 0
    scheduler = build_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    trainer = DecoderTrainer(
        encoder,
        decoder,
        optimizer,
        use_amp=config.training.use_amp,
        loss=config.training.loss,
        l1_weight=config.training.l1_weight,
        lpips_weight=config.training.lpips_weight,
        lpips_network=config.training.lpips_network,
        scheduler=scheduler,
    )
    provenance = capture_run_provenance(
        seed=config.experiment.seed,
        requested_device=config.training.device,
    )
    run_dir = initialize_decoder_run(config, provenance)
    shutil.copyfile(config.data.split_manifest, run_dir / "split_manifest.json")
    source_checkpoints = [Path(official_checkpoint)]
    if config.training.loss == "l1_lpips":
        source_checkpoints.append(_torchvision_backbone_checkpoint(config.training.lpips_network))
    write_source_checkpoints(run_dir, source_checkpoints)

    history: list[dict[str, float]] = []
    epochs = tqdm(range(config.training.epochs), desc="decoder epochs", file=sys.stderr)
    for epoch in epochs:
        generator = torch.Generator().manual_seed(config.experiment.seed + epoch)
        loader = DataLoader(
            frames,
            batch_size=config.training.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.data.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        metrics = trainer.train_epoch(loader)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": float(epoch + 1),
                "train/loss": metrics.total_loss,
                "train/mse": metrics.mean_squared_error,
                "train/l1": metrics.mean_absolute_error,
                "train/lpips": metrics.lpips_loss,
                "train/frames": float(metrics.frames),
                "train/lr": learning_rate,
            }
        )
        epochs.set_postfix(
            loss=f"{metrics.total_loss:.5f}",
            l1=f"{metrics.mean_absolute_error:.5f}",
            lpips=f"{metrics.lpips_loss:.5f}",
        )

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_metrics(
        run_dir,
        {
            **history[-1],
            "decoder/parameters": float(
                sum(parameter.numel() for parameter in decoder.parameters())
            ),
            "decoder/train_episode_count": float(len(train_episode_ids)),
        },
    )
    save_decoder_checkpoint(decoder, run_dir / DECODER_CHECKPOINT_FILENAME)
    return run_dir


def render_decoder_galleries(
    config_path: str | Path,
    data_path: str | Path,
    official_checkpoint: str | Path,
    predictor_root: str | Path,
    *,
    decoder_checkpoint: str | Path | None = None,
    max_predictor_runs: int | None = None,
    max_gallery_episodes: int | None = None,
) -> Path:
    """Render saved predictor rollouts through one decoder without training predictors."""
    config = load_decoder_config(config_path)
    _validate_optional_limit(max_predictor_runs, "max_predictor_runs")
    _validate_optional_limit(max_gallery_episodes, "max_gallery_episodes")
    checkpoint_path = _resolve_decoder_checkpoint(config, decoder_checkpoint)
    if checkpoint_path.parent.parent.resolve() != config.experiment.output_dir.resolve():
        raise ValueError(
            "decoder checkpoint must belong to the configured decoder output directory"
        )
    device = _resolve_device(config.training.device)
    decoder = load_decoder_checkpoint(
        checkpoint_path,
        config.architecture,
        map_location=device,
    ).to(device)

    predictor_checkpoints = tuple(sorted(Path(predictor_root).glob("*/run_*/system.pt")))
    if max_predictor_runs is not None:
        predictor_checkpoints = predictor_checkpoints[:max_predictor_runs]
    if not predictor_checkpoints:
        raise FileNotFoundError(f"no predictor system.pt files found under {predictor_root}")

    manifest = load_split_manifest(config.data.split_manifest)
    gallery_count = max_gallery_episodes or config.galleries.episode_count
    gallery_episode_ids = manifest.test[:gallery_count]
    source = open_pusht_lance_source(
        data_path,
        config.data.frameskip,
        keys_to_load=("pixels", "action"),
    )
    action_normalizer = _load_action_normalizer(
        predictor_checkpoints[0].parent / "data_protocol.json",
        config.data.frameskip,
    )
    action_input_dim = action_normalizer.mean.shape[-1]
    official_model = load_official_lewm(official_checkpoint, action_input_dim)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    render_root = checkpoint_path.parent / "galleries" / f"render_{timestamp}"
    render_root.mkdir(parents=True, exist_ok=False)
    index_records: list[dict[str, object]] = []
    for predictor_checkpoint in tqdm(
        predictor_checkpoints,
        desc="predictor galleries",
        file=sys.stderr,
    ):
        run_dir = predictor_checkpoint.parent
        run_config = load_config(run_dir / "resolved_config.yaml")
        if run_config.model.latent_dim != config.architecture.latent_dim:
            raise ValueError(f"decoder latent dimension does not match {predictor_checkpoint}")
        system = _build_saved_system(run_config, official_model)
        state_dict = torch.load(predictor_checkpoint, map_location="cpu", weights_only=True)
        system.load_state_dict(state_dict, strict=True)
        system = system.to(device).eval()
        queries = _predict_gallery_queries(
            system,
            source,
            gallery_episode_ids,
            action_normalizer,
            config.data.frameskip,
            device,
        )

        variant = run_config.model.variant
        output_dir = render_root / variant / run_dir.name
        output_dir.mkdir(parents=True, exist_ok=False)
        write_decoded_galleries(
            output_dir,
            queries,
            decoder,
            config.galleries.horizons,
            decoder_checkpoint=checkpoint_path,
        )
        write_source_checkpoints(output_dir, [predictor_checkpoint])
        index_records.append(
            {
                "variant": variant,
                "predictor_run": str(run_dir),
                "predictor_checkpoint": str(predictor_checkpoint),
                "predictor_checkpoint_sha256": file_sha256(predictor_checkpoint),
                "output_dir": str(output_dir),
            }
        )
        del system

    (render_root / "index.json").write_text(
        json.dumps(
            {
                "decoder_checkpoint": str(checkpoint_path),
                "decoder_checkpoint_sha256": file_sha256(checkpoint_path),
                "predictor_training_performed": False,
                "records": index_records,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return render_root


def _build_saved_system(config: ExperimentConfig, official_model: LeWMJEPA) -> PredictorSystem:
    system = PredictorSystem(
        encoder=deepcopy(official_model.encoder),
        action_encoder=deepcopy(official_model.action_encoder),
        predictor=build_predictor(config.model),
    )
    system.freeze_shared_modules()
    return system


@torch.no_grad()
def _predict_gallery_queries(
    system: PredictorSystem,
    source: EpisodeSource,
    episode_ids: Sequence[str],
    action_normalizer: ZScoreNormalizer,
    frameskip: int,
    device: torch.device,
) -> tuple[tuple[ObservationTrajectory, Tensor], ...]:
    predictor = cast(DynamicsPredictor, system.predictor)
    queries: list[tuple[ObservationTrajectory, Tensor]] = []
    for episode_id in episode_ids:
        trajectory = load_source_trajectory(
            source,
            episode_index(episode_id),
            frameskip,
        )
        prepared = prepare_observation_batch(
            collate_observation_trajectories([trajectory]),
            action_normalizer,
            device,
        )
        latent_batch = system.encode_batch(prepared)
        rollout, _ = predictor.rollout(latent_batch.latents[:, 0], latent_batch.actions)
        queries.append((trajectory, rollout[0].detach().cpu()))
    return tuple(queries)


def _load_action_normalizer(path: Path, frameskip: int) -> ZScoreNormalizer:
    with path.open(encoding="utf-8") as file:
        protocol = json.load(file)
    normalization = protocol.get("action_normalization")
    if not isinstance(normalization, dict):
        raise ValueError(f"missing action normalization metadata: {path}")
    mean = torch.tensor(normalization.get("raw_mean"), dtype=torch.float32)
    std = torch.tensor(normalization.get("raw_std"), dtype=torch.float32)
    if mean.ndim != 2 or mean.shape[0] != 1 or std.shape != mean.shape:
        raise ValueError(f"invalid action normalization metadata: {path}")
    return ZScoreNormalizer(mean.repeat(1, frameskip), std.repeat(1, frameskip))


def _resolve_decoder_checkpoint(
    config: DecoderConfig,
    checkpoint: str | Path | None,
) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"decoder checkpoint not found: {path}")
        return path
    candidates = sorted(config.experiment.output_dir.glob("run_*/decoder.pt"))
    if not candidates:
        raise FileNotFoundError(f"no decoder checkpoint found under {config.experiment.output_dir}")
    return candidates[-1]


def _torchvision_backbone_checkpoint(network: str) -> Path:
    from urllib.parse import urlparse

    models: Any = import_module("torchvision.models")

    weights_by_network = {
        "alex": models.AlexNet_Weights.IMAGENET1K_V1,
        "vgg": models.VGG16_Weights.IMAGENET1K_V1,
        "squeeze": models.SqueezeNet1_1_Weights.IMAGENET1K_V1,
    }
    weights = weights_by_network.get(network)
    if weights is None:
        raise ValueError(f"unsupported LPIPS network: {network}")
    filename = Path(urlparse(weights.url).path).name
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"LPIPS backbone checkpoint was not cached after initialization: {checkpoint}"
        )
    return checkpoint


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _validate_optional_limit(value: int | None, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")
