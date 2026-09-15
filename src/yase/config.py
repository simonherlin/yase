"""Declarative construction of Yase facades and semantic pipelines."""

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import Yase
from .limits import InputLimits
from .pipeline import PipelineStage, SemanticPipeline
from .registry import BackendRegistry, default_registry

_LIMIT_FIELDS = {
    "max_pixels",
    "max_width",
    "max_height",
    "max_channels",
    "max_bytes",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _options(value: Any, label: str) -> dict[str, Any]:
    options = dict(_mapping(value or {}, label))
    if any(not isinstance(key, str) or not key for key in options):
        raise ValueError(f"{label} keys must be non-empty strings")
    return options


def _limits(value: Any) -> InputLimits | None:
    if value is None:
        return None
    values = _options(value, "limits")
    unknown = set(values) - _LIMIT_FIELDS
    if unknown:
        raise ValueError(f"unknown input limits: {', '.join(sorted(unknown))}")
    return InputLimits(**values)


def _backend(config: Mapping[str, Any], registry: BackendRegistry, label: str) -> Any:
    name = config.get("backend", config.get("model"))
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} requires a non-empty backend name")
    options = _options(config.get("options"), f"{label}.options")
    return registry.create(name, **options)


def build_from_config(
    config: Mapping[str, Any],
    *,
    registry: BackendRegistry | None = None,
    include_plugins: bool = False,
) -> Yase | SemanticPipeline:
    """Build a facade or pipeline from a validated mapping.

    Supported top-level forms are ``backend``/``model`` for one extractor or
    ``stages=[...]`` for a named pipeline. Factories are resolved exclusively
    through :class:`BackendRegistry`; config files cannot import Python paths.
    """
    root = _mapping(config, "config")
    known = {
        "task",
        "backend",
        "model",
        "options",
        "limits",
        "stages",
        "conflict",
        "record_timings",
    }
    unknown = set(root) - known
    if unknown:
        raise ValueError(f"unknown config fields: {', '.join(sorted(unknown))}")
    active_registry = registry or default_registry(include_plugins=include_plugins)
    limits = _limits(root.get("limits"))
    stages = root.get("stages")
    if stages is not None:
        if not isinstance(stages, (list, tuple)) or not stages:
            raise ValueError("stages must be a non-empty list")
        parsed = []
        for index, value in enumerate(stages):
            stage_config = _mapping(value, f"stages[{index}]")
            allowed = {"name", "backend", "model", "options", "enabled"}
            extra = set(stage_config) - allowed
            if extra:
                raise ValueError(
                    f"unknown stages[{index}] fields: {', '.join(sorted(extra))}"
                )
            name = stage_config.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"stages[{index}].name must be non-empty")
            enabled = stage_config.get("enabled", True)
            if not isinstance(enabled, bool):
                raise TypeError(f"stages[{index}].enabled must be boolean")
            parsed.append(
                PipelineStage(
                    name,
                    _backend(stage_config, active_registry, f"stages[{index}]"),
                    enabled=enabled,
                )
            )
        conflict = root.get("conflict", "error")
        record_timings = root.get("record_timings", True)
        if not isinstance(conflict, str) or not isinstance(record_timings, bool):
            raise TypeError("conflict must be a string and record_timings a boolean")
        if limits is not None:
            raise ValueError("limits are only supported for a single Yase facade")
        return SemanticPipeline(
            parsed, conflict=conflict, record_timings=record_timings
        )

    if "backend" not in root and "model" not in root:
        raise ValueError("config requires backend/model or stages")
    task = root.get("task", "depth")
    if not isinstance(task, str):
        raise TypeError("task must be a string")
    name = root.get("backend", root.get("model"))
    options = _options(root.get("options"), "options")
    return Yase(
        task=task,
        model=name,
        registry=active_registry,
        input_limits=limits,
        **options,
    )


def load_config(
    source: str | Path | Mapping[str, Any],
    *,
    registry: BackendRegistry | None = None,
    include_plugins: bool = False,
) -> Yase | SemanticPipeline:
    """Load JSON/TOML config from a path or build directly from a mapping."""
    if isinstance(source, Mapping):
        return build_from_config(
            source, registry=registry, include_plugins=include_plugins
        )
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
    elif path.suffix.lower() in (".toml", ".tml"):
        try:
            toml_loader = importlib.import_module("tomllib")
        except ImportError:
            try:
                toml_loader = importlib.import_module("tomli")
            except ImportError as exc:
                raise ImportError(
                    "TOML loading on Python < 3.11 requires the tomli package"
                ) from exc
        parsed = toml_loader.loads(text)
    else:
        raise ValueError("config path must use .json, .toml, or .tml")
    return build_from_config(
        _mapping(parsed, "parsed config"),
        registry=registry,
        include_plugins=include_plugins,
    )


__all__ = ["build_from_config", "load_config"]
