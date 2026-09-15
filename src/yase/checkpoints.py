"""Atomic checkpoints for resumable semantic video workers."""

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CHECKPOINT_VERSION = 1
_COMPONENTS = ("tracker", "memory", "identity_store")


def _state_for(name: str, component: Any | None) -> dict[str, Any] | None:
    if component is None:
        return None
    method = getattr(component, "state_dict", None)
    if not callable(method):
        raise TypeError(f"{name} must expose state_dict()")
    state = method()
    if not isinstance(state, Mapping):
        raise TypeError(f"{name}.state_dict() must return a mapping")
    return dict(state)


def make_stream_checkpoint(
    *,
    tracker: Any | None = None,
    memory: Any | None = None,
    identity_store: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    model_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-compatible checkpoint for stream state components."""
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if model_fingerprint is not None and not isinstance(model_fingerprint, Mapping):
        raise TypeError("model_fingerprint must be a mapping")
    components = {}
    for name, component in (
        ("tracker", tracker),
        ("memory", memory),
        ("identity_store", identity_store),
    ):
        state = _state_for(name, component)
        if state is not None:
            components[name] = state
    payload = {
        "version": CHECKPOINT_VERSION,
        "metadata": dict(metadata or {}),
        "model_fingerprint": dict(model_fingerprint or {}),
        "components": components,
    }
    # Validate numpy-free JSON compatibility before returning the payload.
    json.dumps(payload, ensure_ascii=False)
    return payload


def _validate_checkpoint(payload: Any) -> Mapping[str, Any]:
    payload = migrate_stream_checkpoint(payload)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported stream checkpoint version")
    metadata = payload.get("metadata", {})
    components = payload.get("components", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("stream checkpoint metadata must be a mapping")
    if not isinstance(components, Mapping):
        raise TypeError("stream checkpoint components must be a mapping")
    fingerprint = payload.get("model_fingerprint", {})
    if not isinstance(fingerprint, Mapping):
        raise TypeError("stream checkpoint model_fingerprint must be a mapping")
    unknown = set(components) - set(_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown stream checkpoint components: {sorted(unknown)}")
    return payload


def migrate_stream_checkpoint(payload: Any) -> dict[str, Any]:
    """Normalize unversioned legacy checkpoints with an unambiguous shape."""
    if not isinstance(payload, Mapping):
        raise TypeError("stream checkpoint must be a mapping")
    migrated = dict(payload)
    version = migrated.get("version")
    if version is None:
        if "components" not in migrated:
            raise ValueError("unsupported stream checkpoint version")
        migrated["version"] = CHECKPOINT_VERSION
    if migrated.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported stream checkpoint version")
    migrated.setdefault("metadata", {})
    migrated.setdefault("components", {})
    return migrated


def save_stream_checkpoint(
    destination: Any,
    *,
    tracker: Any | None = None,
    memory: Any | None = None,
    identity_store: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    model_fingerprint: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a stream checkpoint and return its path.

    The temporary file is created beside the destination and replaced with
    ``os.replace`` so readers never observe a partially-written JSON file.
    """
    path = Path(destination)
    payload = make_stream_checkpoint(
        tracker=tracker,
        memory=memory,
        identity_store=identity_store,
        metadata=metadata,
        model_fingerprint=model_fingerprint,
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_stream_checkpoint(
    source: Any,
    *,
    tracker: Any | None = None,
    memory: Any | None = None,
    identity_store: Any | None = None,
    expected_model_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and optionally restore a stream checkpoint.

    The parsed payload is returned so applications can inspect metadata or
    implement additional component restoration without losing forward fields.
    """
    with Path(source).open("r", encoding="utf-8") as handle:
        payload = _validate_checkpoint(json.load(handle))
    if expected_model_fingerprint is not None:
        if not isinstance(expected_model_fingerprint, Mapping):
            raise TypeError("expected_model_fingerprint must be a mapping")
        actual = dict(payload.get("model_fingerprint", {}))
        expected = dict(expected_model_fingerprint)
        if actual != expected:
            raise ValueError("checkpoint model fingerprint mismatch")
    components = payload["components"]
    for name, component in (
        ("tracker", tracker),
        ("memory", memory),
        ("identity_store", identity_store),
    ):
        if component is None:
            continue
        if name not in components:
            raise ValueError(f"checkpoint does not contain component '{name}'")
        method = getattr(component, "load_state_dict", None)
        if not callable(method):
            raise TypeError(f"{name} must expose load_state_dict(state)")
        method(components[name])
    return dict(payload)


__all__ = [
    "CHECKPOINT_VERSION",
    "load_stream_checkpoint",
    "make_stream_checkpoint",
    "migrate_stream_checkpoint",
    "save_stream_checkpoint",
]
