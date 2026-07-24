"""Post-hoc image decoder model, training, persistence, and diagnostics."""

from .checkpoint import (
    DECODER_CHECKPOINT_FILENAME,
    SOURCE_CHECKPOINTS_FILENAME,
    file_sha256,
    initialize_decoder_run,
    load_decoder_checkpoint,
    save_decoder_checkpoint,
    write_source_checkpoints,
)
from .config import (
    DecoderConfig,
    DecoderDataSettings,
    DecoderExperimentSettings,
    DecoderGallerySettings,
    DecoderLoss,
    DecoderTrainingSettings,
    LPIPSNetwork,
    load_decoder_config,
)
from .galleries import build_decoded_gallery, write_decoded_galleries
from .model import CrossAttentionDecoder, DecoderArchitecture
from .training import DecoderEpochMetrics, DecoderTrainer, FrameDataset, imagenet_to_lpips
from .workflows import render_decoder_galleries, train_decoder

__all__ = [
    "DECODER_CHECKPOINT_FILENAME",
    "SOURCE_CHECKPOINTS_FILENAME",
    "CrossAttentionDecoder",
    "DecoderArchitecture",
    "DecoderConfig",
    "DecoderDataSettings",
    "DecoderEpochMetrics",
    "DecoderExperimentSettings",
    "DecoderGallerySettings",
    "DecoderLoss",
    "DecoderTrainer",
    "DecoderTrainingSettings",
    "FrameDataset",
    "LPIPSNetwork",
    "build_decoded_gallery",
    "file_sha256",
    "initialize_decoder_run",
    "imagenet_to_lpips",
    "load_decoder_checkpoint",
    "load_decoder_config",
    "render_decoder_galleries",
    "save_decoder_checkpoint",
    "train_decoder",
    "write_decoded_galleries",
    "write_source_checkpoints",
]
