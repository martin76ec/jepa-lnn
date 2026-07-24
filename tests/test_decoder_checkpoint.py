"""Tests for isolated decoder checkpoint persistence and provenance."""

import json
from pathlib import Path

import pytest
import torch

from lewm_liquid_predictors.decoder import (
    CrossAttentionDecoder,
    DecoderArchitecture,
    file_sha256,
    load_decoder_checkpoint,
    save_decoder_checkpoint,
    write_source_checkpoints,
)


def _architecture() -> DecoderArchitecture:
    return DecoderArchitecture(
        latent_dim=8,
        hidden_dim=16,
        image_size=16,
        patch_size=8,
        num_layers=2,
        num_heads=4,
    )


def test_decoder_checkpoint_round_trip_is_strict_state_dict(tmp_path: Path) -> None:
    decoder = CrossAttentionDecoder(_architecture())
    checkpoint = tmp_path / "decoder.pt"

    save_decoder_checkpoint(decoder, checkpoint)
    loaded = load_decoder_checkpoint(checkpoint, _architecture())

    assert checkpoint.is_file()
    assert not (tmp_path / "decoder.pt.tmp").exists()
    for expected, actual in zip(decoder.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)

    with pytest.raises(RuntimeError):
        load_decoder_checkpoint(
            checkpoint,
            DecoderArchitecture(
                latent_dim=7,
                hidden_dim=16,
                image_size=16,
                patch_size=8,
                num_layers=2,
                num_heads=4,
            ),
        )


def test_decoder_checkpoint_refuses_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "decoder.pt"
    checkpoint.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        save_decoder_checkpoint(CrossAttentionDecoder(_architecture()), checkpoint)

    assert checkpoint.read_bytes() == b"existing"
    assert not (tmp_path / "decoder.pt.tmp").exists()


@pytest.mark.parametrize("filename", ["system.pt", "lewm.pt", "weights.pt", "other.pt"])
def test_decoder_checkpoint_rejects_every_other_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    path = tmp_path / filename

    with pytest.raises(ValueError, match="decoder.pt"):
        save_decoder_checkpoint(CrossAttentionDecoder(_architecture()), path)
    with pytest.raises(ValueError, match="decoder.pt"):
        load_decoder_checkpoint(path, _architecture())


def test_source_checkpoint_metadata_records_imported_digest_helper(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")

    destination = write_source_checkpoints(tmp_path, [source])
    metadata = json.loads(destination.read_text(encoding="utf-8"))

    assert destination.name == "source_checkpoints.json"
    assert metadata == {str(source): file_sha256(source)}
