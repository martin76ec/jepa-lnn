"""Download and identify immutable external artifacts."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    """Download a file atomically unless the verified destination already exists."""
    if destination.is_file() and file_sha256(destination) == expected_sha256:
        print(f"Verified existing artifact: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--retry",
            "3",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )

    actual_sha256 = file_sha256(partial)
    if actual_sha256 != expected_sha256:
        partial.unlink()
        raise ValueError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_sha256}"
        )
    partial.replace(destination)
    print(f"Downloaded and verified artifact: {destination}")


def main() -> None:
    """Parse one artifact specification and download it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("sha256")
    args = parser.parse_args()
    download_verified(args.url, args.destination, args.sha256)


if __name__ == "__main__":
    main()
