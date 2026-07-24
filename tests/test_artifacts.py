from pathlib import Path
from unittest.mock import Mock

from lewm_liquid_predictors.artifacts import download_verified, file_sha256


def test_download_verified_skips_matching_destination(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"complete")
    run = Mock()
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "lewm_liquid_predictors.artifacts.subprocess.run", run
    )

    download_verified("https://example.test/artifact.bin", destination, file_sha256(destination))

    run.assert_not_called()


def test_download_verified_replaces_invalid_destination(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"invalid")
    expected = tmp_path / "expected.bin"
    expected.write_bytes(b"complete")

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check
        Path(command[command.index("--output") + 1]).write_bytes(b"complete")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "lewm_liquid_predictors.artifacts.subprocess.run", fake_run
    )

    download_verified("https://example.test/artifact.bin", destination, file_sha256(expected))

    assert destination.read_bytes() == b"complete"
