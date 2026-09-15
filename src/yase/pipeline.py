"""Composable semantic extraction pipelines."""

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .backends import CompositeExtractor
from .core import ImageInput, SemanticResult, load_image
from .stages import StageContext, StageSpec


@dataclass(frozen=True)
class PipelineStage:
    """A named model or post-processing stage."""

    name: str
    backend: Any
    enabled: bool = True
    spec: StageSpec | None = None


class SemanticPipeline:
    """Run independent semantic stages and merge their typed outputs.

    Stages are intentionally independent: a fast detector, a slower VLM, an
    OCR engine, and an embedding model can be developed and deployed
    separately. ``CompositeExtractor`` remains the lower-level compatibility
    API; this class adds names, enable/disable control, and timestamp handling.
    """

    def __init__(
        self,
        stages: Mapping[str, Any] | Iterable[PipelineStage],
        conflict: str = "error",
        record_timings: bool = True,
    ) -> None:
        if isinstance(stages, Mapping):
            items = [
                PipelineStage(str(name), backend) for name, backend in stages.items()
            ]
        else:
            items = list(stages)
        if not items:
            raise ValueError("at least one pipeline stage is required")
        if any(not isinstance(stage, PipelineStage) for stage in items):
            raise TypeError("iterable stages must contain PipelineStage values")
        self.stages = items
        self.conflict = conflict
        self.record_timings = record_timings
        self.validate()

    def validate(self) -> tuple[str, ...]:
        """Validate attached contracts while preserving legacy stages."""
        contracted = [stage.spec for stage in self.stages if stage.spec is not None]
        if not contracted:
            return ("frame", "image")
        from .scheduler import ObservationScheduler

        ObservationScheduler(self.stages)
        available = {"frame", "image"}
        for spec in contracted:
            available.update(spec.provides)
        return tuple(sorted(available))

    @property
    def active_stages(self) -> tuple[PipelineStage, ...]:
        return tuple(stage for stage in self.stages if stage.enabled)

    def extract(
        self, image: ImageInput, timestamp: float | None = None
    ) -> SemanticResult:
        active = self.active_stages
        if not active:
            raise ValueError("pipeline has no enabled stages")
        outputs = []
        timings = {}
        for stage in active:
            started = time.perf_counter()
            output = self._invoke_stage(stage, image)
            outputs.append((stage.name, CompositeExtractor._coerce(output, stage.name)))
            timings[stage.name] = time.perf_counter() - started
        result = self._merge(outputs)
        if self.record_timings:
            result = self._with_pipeline_metadata(result, timings)
        return result.with_timestamp(timestamp)

    def extract_many(
        self,
        images: Iterable[ImageInput],
        timestamps: Iterable[float | None] | None = None,
        error_policy: str = "raise",
        on_error: Callable[[Exception, int, str], Any | None] | None = None,
    ) -> list[SemanticResult | None]:
        """Run all active stages while preserving input order.

        Each stage uses ``extract_batch`` when it implements it, which avoids
        repeatedly preprocessing the same batch for embedding and detector
        backends.
        """
        if error_policy not in ("raise", "skip"):
            raise ValueError("error_policy must be raise or skip")
        items = list(images)
        stamps = [None] * len(items) if timestamps is None else list(timestamps)
        if len(stamps) != len(items):
            raise ValueError("timestamps must have the same length as images")
        active = self.active_stages
        if not active:
            raise ValueError("pipeline has no enabled stages")
        rows: list[list[tuple[str, SemanticResult]]] = [[] for _ in items]
        timings = {}
        for stage in active:
            started = time.perf_counter()
            stage_values: list[Any] = [None] * len(items)

            def recover(index: int) -> Any | None:
                try:
                    return self._invoke_stage(stage, items[index])
                except Exception as exc:
                    if on_error is not None:
                        return on_error(exc, index, stage.name)
                    if error_policy == "skip":
                        return None
                    raise

            if hasattr(stage.backend, "extract_batch"):
                arrays = []
                valid_indices = []
                for index, item in enumerate(items):
                    try:
                        arrays.append(load_image(item))
                        valid_indices.append(index)
                    except Exception as exc:
                        if on_error is not None:
                            stage_values[index] = on_error(exc, index, stage.name)
                        elif error_policy == "raise":
                            raise
                if arrays:
                    try:
                        values = list(stage.backend.extract_batch(arrays))
                        if len(values) != len(valid_indices):
                            raise ValueError(
                                f"stage '{stage.name}' must return one result per "
                                "valid input image"
                            )
                        for index, value in zip(valid_indices, values):
                            stage_values[index] = value
                    except Exception:
                        if error_policy == "raise":
                            raise
                        for index in valid_indices:
                            stage_values[index] = recover(index)
            else:
                for index in range(len(items)):
                    stage_values[index] = recover(index)
            timings[stage.name] = time.perf_counter() - started
            for index, value in enumerate(stage_values):
                if value is None:
                    continue
                try:
                    rows[index].append(
                        (stage.name, CompositeExtractor._coerce(value, stage.name))
                    )
                except Exception as exc:
                    if on_error is not None:
                        replacement = on_error(exc, index, stage.name)
                    elif error_policy == "skip":
                        replacement = None
                    else:
                        raise
                    if replacement is not None:
                        rows[index].append(
                            (
                                stage.name,
                                CompositeExtractor._coerce(replacement, stage.name),
                            )
                        )
        results: list[SemanticResult | None] = []
        for row, timestamp in zip(rows, stamps):
            if not row:
                results.append(None)
                continue
            result = self._merge(row).with_timestamp(timestamp)
            if self.record_timings:
                result = self._with_pipeline_metadata(result, timings)
            results.append(result)
        return results

    @staticmethod
    def _invoke_stage(stage: PipelineStage, image: ImageInput) -> Any:
        if stage.spec is not None and hasattr(stage.backend, "run"):
            return stage.backend.run(load_image(image), StageContext())
        return CompositeExtractor._invoke(stage.backend, image)

    def _merge(self, values: list[tuple[str, SemanticResult]]) -> SemanticResult:
        merged: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        timestamp = None
        for source, result in values:
            for field in CompositeExtractor._FIELDS:
                value = getattr(result, field)
                if value is None:
                    continue
                if field in merged:
                    if self.conflict == "error":
                        raise ValueError(
                            f"conflict for field '{field}' from '{source}'"
                        )
                    if self.conflict == "first":
                        continue
                merged[field] = value
            if result.timestamp is not None and timestamp is None:
                timestamp = result.timestamp
            for key, value in result.metadata.items():
                if key in metadata and self.conflict == "error":
                    raise ValueError(f"conflict for metadata '{key}' from '{source}'")
                if key not in metadata or self.conflict == "last":
                    metadata[key] = value
        return SemanticResult(
            depth=merged.get("depth"),
            segmentation=merged.get("segmentation"),
            detections=merged.get("detections"),
            tags=merged.get("tags"),
            embeddings=merged.get("embeddings"),
            timestamp=timestamp,
            metadata=metadata,
            ocr=merged.get("ocr"),
            caption=merged.get("caption"),
            scene=merged.get("scene"),
            events=merged.get("events"),
            keypoints=merged.get("keypoints"),
            relations=merged.get("relations"),
            document=merged.get("document"),
            depth_map=merged.get("depth_map"),
        )

    @staticmethod
    def _with_pipeline_metadata(
        result: SemanticResult, timings: Mapping[str, float]
    ) -> SemanticResult:
        metadata = dict(result.metadata)
        metadata["pipeline_timings_seconds"] = dict(timings)
        return SemanticResult(
            depth=result.depth,
            segmentation=result.segmentation,
            detections=result.detections,
            tags=result.tags,
            embeddings=result.embeddings,
            timestamp=result.timestamp,
            metadata=metadata,
            ocr=result.ocr,
            caption=result.caption,
            scene=result.scene,
            events=result.events,
            keypoints=result.keypoints,
            relations=result.relations,
            document=result.document,
            depth_map=result.depth_map,
        )

    def as_scheduler(
        self, config: Any = None, on_stage: Any = None, tracer: Any = None
    ) -> Any:
        """Return a dependency-aware scheduler for contracted active stages."""
        from .scheduler import ObservationScheduler

        active = self.active_stages
        if any(stage.spec is None for stage in active):
            raise ValueError("all active stages need StageSpec for scheduler execution")
        return ObservationScheduler(
            active, config=config, on_stage=on_stage, tracer=tracer
        )

    def extract_scheduled(
        self,
        image: ImageInput,
        timestamp: float | None = None,
        *,
        config: Any = None,
        tracer: Any = None,
    ) -> SemanticResult:
        """Execute contracted stages as a DAG and return a SemanticResult."""
        report = self.as_scheduler(config=config, tracer=tracer).run(image)
        return self._result_from_scheduler_report(report, timestamp)

    async def extract_scheduled_async(
        self,
        image: ImageInput,
        timestamp: float | None = None,
        *,
        config: Any = None,
        tracer: Any = None,
    ) -> SemanticResult:
        """Async counterpart of :meth:`extract_scheduled`."""
        report = await self.as_scheduler(config=config, tracer=tracer).arun(image)
        return self._result_from_scheduler_report(report, timestamp)

    def extract_bundle_scheduled(
        self,
        image: ImageInput,
        timestamp: float | None = None,
        *,
        frame: Any = None,
        frame_id: int = 0,
        source_id: str = "default",
        config: Any = None,
    ) -> Any:
        """Execute a DAG and promote its result to an ObservationBundle."""
        from .observation import FrameRef, ObservationBundle

        result = self.extract_scheduled(image, timestamp=timestamp, config=config)
        if frame is None:
            array = load_image(image)
            frame = FrameRef(
                frame_id=frame_id,
                source_id=source_id,
                timestamp=result.timestamp,
                width=array.shape[1],
                height=array.shape[0],
            )
        return ObservationBundle.from_result(result, frame=frame)

    @staticmethod
    def _result_from_scheduler_report(
        report: Any, timestamp: float | None
    ) -> SemanticResult:
        return report.as_result(timestamp=timestamp)

    def __call__(
        self, image: ImageInput, timestamp: float | None = None
    ) -> SemanticResult:
        return self.extract(image, timestamp=timestamp)


__all__ = ["PipelineStage", "SemanticPipeline"]
