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
    DecoderTrainingSettings,
    load_decoder_config,
)
from .galleries import build_decoded_gallery, write_decoded_galleries
from .model import CrossAttentionDecoder, DecoderArchitecture
from .training import DecoderEpochMetrics, DecoderTrainer, FrameDataset
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
    "DecoderTrainer",
    "DecoderTrainingSettings",
    "FrameDataset",
    "build_decoded_gallery",
    "file_sha256",
    "initialize_decoder_run",
    "load_decoder_checkpoint",
    "load_decoder_config",
    "render_decoder_galleries",
    "save_decoder_checkpoint",
    "train_decoder",
    "write_decoded_galleries",
    "write_source_checkpoints",
]
