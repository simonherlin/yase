"""Local model-artifact inspection and integrity helpers."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ArtifactInfo:
    """Reproducibility metadata for a local model artifact."""

    path: str
    size_bytes: int
    suffix: str
    modified_at: str
    sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "suffix": self.suffix,
            "modified_at": self.modified_at,
            "sha256": self.sha256,
        }


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Hash a local artifact without loading it into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifact(path: str, *, checksum: bool = False) -> ArtifactInfo:
    """Return size, suffix, timestamp, and optionally SHA-256 metadata."""
    artifact = Path(path)
    stat = artifact.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return ArtifactInfo(
        path=str(artifact),
        size_bytes=stat.st_size,
        suffix=artifact.suffix.lower(),
        modified_at=modified,
        sha256=sha256_file(str(artifact)) if checksum else None,
    )


def verify_artifact(path: str, expected_sha256: str) -> ArtifactInfo:
    """Verify a local artifact and return its complete metadata."""
    if len(expected_sha256) != 64:
        raise ValueError("expected_sha256 must be a 64-character SHA-256 digest")
    info = inspect_artifact(path, checksum=True)
    if info.sha256 != expected_sha256.lower():
        raise ValueError(
            f"artifact checksum mismatch for {path}: expected {expected_sha256}, "
            f"got {info.sha256}"
        )
    return info


__all__ = ["ArtifactInfo", "inspect_artifact", "sha256_file", "verify_artifact"]
