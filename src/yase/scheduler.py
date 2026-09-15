"""Dependency-aware execution for semantic stages.

This module is intentionally small and framework-neutral.  It provides the
first production-shaped runtime boundary while leaving model loading to the
existing adapters: a graph can run synchronously, from an async application,
with a bounded result cache, explicit cancellation, and stage telemetry.
"""

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event
from typing import Any

import numpy as np

from .backends import CompositeExtractor
from .core import ImageInput, SemanticResult, load_image
from .errors import SchedulerError, StageCancelled
from .observability import RuntimeMetrics
from .observation import FrameRef
from .serialization import _json_value
from .stages import StageContext, StageSpec


@dataclass(frozen=True)
class SchedulerConfig:
    """Safety and performance policies for one scheduler instance."""

    cache_size: int = 128
    max_total_latency_ms: float | None = None
    max_stage_latency_ms: float | None = None
    on_error: str = "raise"
    on_budget: str = "raise"
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.cache_size < 0:
            raise ValueError("cache_size cannot be negative")
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or self.max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        for name in ("max_total_latency_ms", "max_stage_latency_ms"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        if self.on_budget not in ("raise", "skip"):
            raise ValueError("on_budget must be 'raise' or 'skip'")


@dataclass(frozen=True)
class StageExecution:
    """Telemetry for one attempted graph stage."""

    name: str
    status: str
    duration_seconds: float = 0.0
    cache_hit: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "cache_hit": self.cache_hit,
            "error": self.error,
        }


@dataclass(frozen=True)
class SchedulerReport:
    """Outputs and telemetry from one graph execution."""

    fields: Mapping[str, Any]
    stages: tuple[StageExecution, ...]
    elapsed_seconds: float
    cache_hits: int = 0

    def to_dict(self, include_arrays: bool = False) -> dict:
        return _json_value(
            {
                "fields": dict(self.fields),
                "stages": [stage.to_dict() for stage in self.stages],
                "elapsed_seconds": self.elapsed_seconds,
                "cache_hits": self.cache_hits,
            },
            include_arrays,
        )

    def as_result(self, timestamp: float | None = None) -> SemanticResult:
        """Project semantic fields from the report to the stable result API."""
        values = {
            name: self.fields[name]
            for name in CompositeExtractor._FIELDS
            if name in self.fields
        }
        values["metadata"] = {"scheduler": self.to_dict()}
        return SemanticResult(**values).with_timestamp(timestamp)


class ObservationScheduler:
    """Execute a validated DAG of ``PipelineStage``-compatible objects."""

    def __init__(
        self,
        stages: Sequence[Any],
        *,
        config: SchedulerConfig | None = None,
        on_stage: Callable[[StageExecution], None] | None = None,
        metrics: RuntimeMetrics | None = None,
        tracer: Any | None = None,
        initial_fields: Sequence[str] = ("image", "frame"),
    ) -> None:
        items = list(stages)
        if not items:
            raise ValueError("at least one stage is required")
        self.stages = tuple(items)
        self.config = config or SchedulerConfig()
        self.on_stage = on_stage
        self.metrics = metrics
        if tracer is not None and not (
            hasattr(tracer, "span") or hasattr(tracer, "start_as_current_span")
        ):
            raise TypeError(
                "tracer must expose span(name, attributes) or "
                "start_as_current_span(name)"
            )
        self.tracer = tracer
        self.initial_fields = frozenset(("image", "frame", *initial_fields))
        self._cache: OrderedDict[tuple, Any] = OrderedDict()
        self._plan = self._build_plan()

    @property
    def plan(self) -> tuple[Any, ...]:
        """Return the stable topological order selected for the graph."""
        return self._plan

    def clear_cache(self) -> None:
        """Drop all persistent stage results."""
        self._cache.clear()

    @contextmanager
    def _stage_span(self, stage: Any) -> Any:
        if self.tracer is None:
            yield None
            return
        attributes = {"yase.stage": stage.name}
        if hasattr(self.tracer, "span"):
            with self.tracer.span(f"yase.stage.{stage.name}", attributes):
                yield None
        else:
            with self.tracer.start_as_current_span(f"yase.stage.{stage.name}"):
                yield None

    def _invoke_traced(
        self, stage: Any, normalized: np.ndarray, context: StageContext
    ) -> Any:
        with self._stage_span(stage):
            return self._invoke(stage, normalized, context)

    def cache_info(self) -> dict[str, int]:
        """Return bounded-cache occupancy without exposing mutable internals."""
        return {"size": len(self._cache), "capacity": self.config.cache_size}

    def run(
        self,
        image: ImageInput,
        *,
        frame: FrameRef | None = None,
        initial_fields: Mapping[str, Any] | None = None,
        cancel_event: Event | None = None,
    ) -> SchedulerReport:
        """Run the graph and return all named fields plus stage telemetry."""
        if self.config.max_workers > 1:
            return self._run_parallel(
                image,
                frame=frame,
                initial_fields=initial_fields,
                cancel_event=cancel_event,
            )
        started = time.perf_counter()
        normalized = load_image(image)
        fields: dict[str, Any] = {"image": normalized}
        if frame is not None:
            fields["frame"] = frame
        if initial_fields:
            fields.update(initial_fields)
        deadline = (
            None
            if self.config.max_total_latency_ms is None
            else started + self.config.max_total_latency_ms / 1000.0
        )
        context = StageContext(
            frame=frame, cancel_event=cancel_event, deadline=deadline
        )
        executions: list[StageExecution] = []
        cache_hits = 0

        for stage in self._plan:
            spec = stage.spec
            assert isinstance(spec, StageSpec)
            if cancel_event is not None and cancel_event.is_set():
                execution = StageExecution(stage.name, "cancelled")
                self._record(executions, execution)
                raise StageCancelled(f"execution cancelled before stage '{stage.name}'")

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if (
                self.config.max_total_latency_ms is not None
                and elapsed_ms >= self.config.max_total_latency_ms
            ):
                execution = StageExecution(stage.name, "budget_exceeded")
                self._record(executions, execution)
                if self.config.on_budget == "raise" and not spec.optional:
                    raise SchedulerError(
                        f"total latency budget exceeded before stage '{stage.name}'"
                    )
                continue

            missing = sorted(set(spec.requires) - fields.keys())
            if missing:
                execution = StageExecution(
                    stage.name,
                    "skipped",
                    error=f"missing fields: {', '.join(missing)}",
                )
                self._record(executions, execution)
                if not spec.optional and self.config.on_error == "raise":
                    raise SchedulerError(
                        f"stage '{stage.name}' missing fields: {', '.join(missing)}"
                    )
                continue

            context.metadata["stage"] = stage.name
            context.metadata["inputs"] = {
                name: fields[name] for name in spec.requires if name in fields
            }
            key = self._cache_key(stage, normalized, fields)
            if key is not None and key in self._cache:
                output = self._cache.pop(key)
                self._cache[key] = output
                self._update_fields(fields, spec, output)
                execution = StageExecution(stage.name, "cached", cache_hit=True)
                cache_hits += 1
                self._record(executions, execution)
                continue

            stage_started = time.perf_counter()
            try:
                output = self._invoke_traced(stage, normalized, context)
                duration = time.perf_counter() - stage_started
                if (
                    self.config.max_stage_latency_ms is not None
                    and duration * 1000.0 > self.config.max_stage_latency_ms
                ):
                    execution = StageExecution(
                        stage.name, "budget_exceeded", duration_seconds=duration
                    )
                    self._record(executions, execution)
                    if self.config.on_budget == "raise" and not spec.optional:
                        raise SchedulerError(
                            f"stage '{stage.name}' exceeded its latency budget"
                        )
                    continue
                self._update_fields(fields, spec, output)
                if key is not None:
                    self._cache[key] = output
                    while len(self._cache) > self.config.cache_size:
                        self._cache.popitem(last=False)
                execution = StageExecution(
                    stage.name, "completed", duration_seconds=duration
                )
            except (StageCancelled, SchedulerError):
                raise
            except Exception as exc:
                execution = StageExecution(
                    stage.name,
                    "failed",
                    duration_seconds=time.perf_counter() - stage_started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._record(executions, execution)
                if self.config.on_error == "raise" and not spec.optional:
                    raise SchedulerError(f"stage '{stage.name}' failed: {exc}") from exc
                continue
            self._record(executions, execution)

        report = SchedulerReport(
            fields=fields,
            stages=tuple(executions),
            elapsed_seconds=time.perf_counter() - started,
            cache_hits=cache_hits,
        )
        if self.metrics is not None:
            self.metrics.record_report(report)
        return report

    def _run_parallel(
        self,
        image: ImageInput,
        *,
        frame: FrameRef | None,
        initial_fields: Mapping[str, Any] | None,
        cancel_event: Event | None,
    ) -> SchedulerReport:
        """Run independent ready stages concurrently with bounded workers.

        Dependency resolution and cache mutation stay on the caller thread.
        Workers only execute backends, so reports and callbacks remain stable
        even when stage completion order differs.
        """
        started = time.perf_counter()
        normalized = load_image(image)
        fields: dict[str, Any] = {"image": normalized}
        if frame is not None:
            fields["frame"] = frame
        if initial_fields:
            fields.update(initial_fields)
        deadline = (
            None
            if self.config.max_total_latency_ms is None
            else started + self.config.max_total_latency_ms / 1000.0
        )
        pending = list(self._plan)
        executions: list[StageExecution] = []
        cache_hits = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            while pending:
                if cancel_event is not None and cancel_event.is_set():
                    raise StageCancelled(
                        "execution cancelled before the next stage batch"
                    )

                ready = [
                    stage
                    for stage in pending
                    if set(stage.spec.requires).issubset(fields.keys())
                ]
                if not ready:
                    stage = pending.pop(0)
                    missing = sorted(set(stage.spec.requires) - fields.keys())
                    execution = StageExecution(
                        stage.name,
                        "skipped",
                        error=f"missing fields: {', '.join(missing)}",
                    )
                    self._record(executions, execution)
                    if not stage.spec.optional and self.config.on_error == "raise":
                        raise SchedulerError(
                            f"stage '{stage.name}' missing fields: {', '.join(missing)}"
                        )
                    continue

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if (
                    self.config.max_total_latency_ms is not None
                    and elapsed_ms >= self.config.max_total_latency_ms
                ):
                    for stage in ready:
                        pending.remove(stage)
                        execution = StageExecution(stage.name, "budget_exceeded")
                        self._record(executions, execution)
                        if self.config.on_budget == "raise" and not stage.spec.optional:
                            raise SchedulerError(
                                f"total latency budget exceeded before stage '{stage.name}'"
                            )
                    continue

                batch = ready[: self.config.max_workers]
                futures = {}
                batch_keys = {}
                for stage in batch:
                    key = self._cache_key(stage, normalized, fields)
                    if key is not None and key in self._cache:
                        output = self._cache.pop(key)
                        self._cache[key] = output
                        self._update_fields(fields, stage.spec, output)
                        pending.remove(stage)
                        execution = StageExecution(stage.name, "cached", cache_hit=True)
                        cache_hits += 1
                        self._record(executions, execution)
                        continue
                    context = StageContext(
                        frame=frame,
                        cancel_event=cancel_event,
                        deadline=deadline,
                        metadata={
                            "stage": stage.name,
                            "inputs": {
                                name: fields[name]
                                for name in stage.spec.requires
                                if name in fields
                            },
                        },
                    )
                    batch_keys[stage.name] = key
                    futures[stage.name] = (
                        stage,
                        time.perf_counter(),
                        executor.submit(
                            self._invoke_traced, stage, normalized, context
                        ),
                    )

                for stage in batch:
                    if stage.name not in futures:
                        continue
                    _, stage_started, future = futures[stage.name]
                    try:
                        output = future.result()
                        duration = time.perf_counter() - stage_started
                        if (
                            self.config.max_stage_latency_ms is not None
                            and duration * 1000.0 > self.config.max_stage_latency_ms
                        ):
                            execution = StageExecution(
                                stage.name,
                                "budget_exceeded",
                                duration_seconds=duration,
                            )
                            self._record(executions, execution)
                            if (
                                self.config.on_budget == "raise"
                                and not stage.spec.optional
                            ):
                                raise SchedulerError(
                                    f"stage '{stage.name}' exceeded its latency budget"
                                )
                            pending.remove(stage)
                            continue
                        self._update_fields(fields, stage.spec, output)
                        key = batch_keys[stage.name]
                        if key is not None:
                            self._cache[key] = output
                            while len(self._cache) > self.config.cache_size:
                                self._cache.popitem(last=False)
                        pending.remove(stage)
                        self._record(
                            executions,
                            StageExecution(
                                stage.name, "completed", duration_seconds=duration
                            ),
                        )
                    except (StageCancelled, SchedulerError):
                        raise
                    except Exception as exc:
                        pending.remove(stage)
                        execution = StageExecution(
                            stage.name,
                            "failed",
                            duration_seconds=time.perf_counter() - stage_started,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        self._record(executions, execution)
                        if self.config.on_error == "raise" and not stage.spec.optional:
                            raise SchedulerError(
                                f"stage '{stage.name}' failed: {exc}"
                            ) from exc

        report = SchedulerReport(
            fields=fields,
            stages=tuple(executions),
            elapsed_seconds=time.perf_counter() - started,
            cache_hits=cache_hits,
        )
        if self.metrics is not None:
            self.metrics.record_report(report)
        return report

    async def arun(self, image: ImageInput, **kwargs: Any) -> SchedulerReport:
        """Run without blocking an async event loop.

        The sync executor runs in a worker thread; model adapters can later
        provide native async implementations without changing this boundary.
        """
        cancel_event = kwargs.pop("cancel_event", None) or Event()
        try:
            return await asyncio.to_thread(
                self.run, image, cancel_event=cancel_event, **kwargs
            )
        except asyncio.CancelledError:
            cancel_event.set()
            raise

    def run_many(
        self,
        images: Sequence[ImageInput],
        *,
        frames: Sequence[FrameRef | None] | None = None,
        initial_fields: Sequence[Mapping[str, Any] | None] | None = None,
        cancel_event: Event | None = None,
    ) -> list[SchedulerReport]:
        """Run an ordered batch while preserving one report per input."""
        items = list(images)
        frame_items = [None] * len(items) if frames is None else list(frames)
        field_items = (
            [None] * len(items) if initial_fields is None else list(initial_fields)
        )
        if len(frame_items) != len(items):
            raise ValueError("frames must have the same length as images")
        if len(field_items) != len(items):
            raise ValueError("initial_fields must have the same length as images")
        return [
            self.run(
                image,
                frame=frame,
                initial_fields=fields,
                cancel_event=cancel_event,
            )
            for image, frame, fields in zip(items, frame_items, field_items)
        ]

    def _build_plan(self) -> tuple[Any, ...]:
        names: set[str] = set()
        providers: dict[str, Any] = {}
        for stage in self.stages:
            spec = getattr(stage, "spec", None)
            name = getattr(stage, "name", None)
            if not isinstance(spec, StageSpec) or not name:
                raise TypeError("scheduler stages require name and StageSpec")
            if name != spec.name:
                raise SchedulerError(
                    f"stage name '{name}' does not match spec name '{spec.name}'"
                )
            if not getattr(stage, "enabled", True):
                continue
            if name in names:
                raise SchedulerError(f"duplicate stage name: {name}")
            names.add(name)
            for field_name in spec.provides:
                if field_name in providers:
                    raise SchedulerError(
                        f"multiple stages provide field '{field_name}'"
                    )
                providers[field_name] = stage

        remaining = [stage for stage in self.stages if getattr(stage, "enabled", True)]
        available = set(self.initial_fields)
        ordered: list[Any] = []
        while remaining:
            progressed = False
            for stage in tuple(remaining):
                requires = set(stage.spec.requires)
                if requires.issubset(available):
                    ordered.append(stage)
                    remaining.remove(stage)
                    available.update(stage.spec.provides)
                    progressed = True
            if not progressed:
                unresolved = ", ".join(stage.name for stage in remaining)
                raise SchedulerError(f"cannot resolve stage dependencies: {unresolved}")
        return tuple(ordered)

    @staticmethod
    def _invoke(stage: Any, image: np.ndarray, context: StageContext) -> Any:
        if hasattr(stage.backend, "run"):
            return stage.backend.run(image, context)
        return CompositeExtractor._invoke(stage.backend, image)

    def _cache_key(
        self, stage: Any, image: np.ndarray, fields: Mapping[str, Any]
    ) -> tuple | None:
        if self.config.cache_size == 0:
            return None
        values = tuple(
            (name, _fingerprint(fields[name]))
            for name in stage.spec.requires
            if name in fields
        )
        return (stage.name, _fingerprint(image), values)

    def _record(
        self, executions: list[StageExecution], execution: StageExecution
    ) -> None:
        executions.append(execution)
        if self.on_stage is not None:
            self.on_stage(execution)

    @staticmethod
    def _update_fields(fields: dict[str, Any], spec: StageSpec, output: Any) -> None:
        if isinstance(output, SemanticResult):
            values = {
                name: getattr(output, name)
                for name in CompositeExtractor._FIELDS
                if getattr(output, name) is not None
            }
        elif isinstance(output, Mapping):
            values = dict(output)
        elif len(spec.provides) == 1:
            values = {spec.provides[0]: output}
        else:
            raise SchedulerError(
                f"stage '{spec.name}' must return a mapping or one provided field"
            )

        if spec.provides:
            missing = sorted(set(spec.provides) - values.keys())
            if missing:
                raise SchedulerError(
                    f"stage '{spec.name}' did not provide: {', '.join(missing)}"
                )
            fields.update({name: values[name] for name in spec.provides})
        else:
            fields.update(values)


def _fingerprint(value: Any) -> str:
    """Stable-enough process-local fingerprint for bounded stage caching."""
    digest = hashlib.sha256()
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            digest.update(_fingerprint(value[key]).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(_fingerprint(item).encode())
    elif isinstance(value, SemanticResult):
        digest.update(_fingerprint(value.metadata).encode())
        for field_name in CompositeExtractor._FIELDS:
            digest.update(_fingerprint(getattr(value, field_name)).encode())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


__all__ = [
    "ObservationScheduler",
    "SchedulerConfig",
    "SchedulerReport",
    "StageExecution",
]
