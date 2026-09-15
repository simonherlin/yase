"""Contracts used to evolve pipelines into validated execution graphs."""

import math
import time
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .observation import FrameRef


def _fields(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not value for value in result):
        raise ValueError(f"{label} cannot contain empty field names")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} cannot contain duplicate field names")
    return result


@dataclass(frozen=True)
class StageSpec:
    """Declarative input/output contract for one semantic stage."""

    name: str
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    estimated_latency_ms: float | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        object.__setattr__(self, "requires", _fields(self.requires, "requires"))
        object.__setattr__(self, "provides", _fields(self.provides, "provides"))
        object.__setattr__(
            self, "capabilities", _fields(self.capabilities, "capabilities")
        )
        if set(self.requires).intersection(self.provides):
            raise ValueError("a stage cannot require and provide the same field")
        if self.estimated_latency_ms is not None and (
            not math.isfinite(self.estimated_latency_ms)
            or self.estimated_latency_ms < 0
        ):
            raise ValueError("estimated_latency_ms must be finite and non-negative")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "requires": list(self.requires),
            "provides": list(self.provides),
            "capabilities": list(self.capabilities),
            "estimated_latency_ms": self.estimated_latency_ms,
            "optional": self.optional,
        }


@dataclass
class StageContext:
    """Mutable per-invocation context shared by a stage and its adapters."""

    frame: FrameRef | None = None
    cache: MutableMapping[str, Any] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    cancel_event: Any | None = None
    deadline: float | None = None

    @property
    def cancelled(self) -> bool:
        """Whether the caller requested cancellation for the current stage."""
        return bool(self.cancel_event is not None and self.cancel_event.is_set())

    @property
    def deadline_exceeded(self) -> bool:
        """Whether the scheduler deadline has elapsed."""
        return self.deadline is not None and time.monotonic() >= self.deadline


class Stage(Protocol):
    """Protocol for DAG stages; legacy extractors remain supported."""

    spec: StageSpec

    def run(self, image: Any, context: StageContext) -> Any: ...


def validate_stage_specs(
    specs: Sequence[StageSpec], initial_fields: Sequence[str] = ("image", "frame")
) -> tuple[str, ...]:
    """Validate ordered stage dependencies and return the produced fields.

    The order is deliberately explicit for now. It gives current pipelines a
    safe migration path before a scheduler introduces fan-out/fan-in DAGs.
    """
    available = set(initial_fields)
    names = set()
    for spec in specs:
        if spec.name in names:
            raise ValueError(f"duplicate stage name: {spec.name}")
        names.add(spec.name)
        missing = sorted(set(spec.requires) - available)
        if missing:
            raise ValueError(
                f"stage '{spec.name}' requires unavailable fields: {', '.join(missing)}"
            )
        available.update(spec.provides)
    return tuple(sorted(available))


__all__ = ["Stage", "StageContext", "StageSpec", "validate_stage_specs"]
