"""Small, dependency-free benchmark harness for model comparisons."""

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from .core import ImageInput, SemanticResult
from .metrics import evaluate_tracking
from .schema import Detection


@dataclass(frozen=True)
class BenchmarkReport:
    """Latency and reliability report for one extractor."""

    name: str
    samples: int
    failures: int
    total_seconds: float
    latencies_seconds: tuple[float, ...]
    quality: Mapping[str, float] = field(default_factory=dict)
    phase_latencies_seconds: Mapping[str, tuple[float, ...]] = field(
        default_factory=dict
    )

    @property
    def throughput(self) -> float:
        return self.samples / self.total_seconds if self.total_seconds else 0.0

    def percentile(self, percentile: float) -> float:
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0, 100]")
        if not self.latencies_seconds:
            return 0.0
        ordered = sorted(self.latencies_seconds)
        index = (len(ordered) - 1) * percentile / 100
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = index - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    def to_dict(self) -> dict[str, Any]:
        phase_timings = {
            phase: {
                "samples": len(values),
                "mean_seconds": mean(values) if values else 0.0,
                "p50_seconds": self._percentile(values, 50),
                "p95_seconds": self._percentile(values, 95),
                "p99_seconds": self._percentile(values, 99),
            }
            for phase, values in self.phase_latencies_seconds.items()
        }
        return {
            "name": self.name,
            "samples": self.samples,
            "failures": self.failures,
            "total_seconds": self.total_seconds,
            "throughput": self.throughput,
            "latency_mean_seconds": mean(self.latencies_seconds)
            if self.latencies_seconds
            else 0.0,
            "latency_median_seconds": median(self.latencies_seconds)
            if self.latencies_seconds
            else 0.0,
            "latency_p50_seconds": self.percentile(50),
            "latency_p95_seconds": self.percentile(95),
            "latency_p99_seconds": self.percentile(99),
            "phase_timings": phase_timings,
            "quality": dict(self.quality),
        }

    @staticmethod
    def _percentile(values: tuple[float, ...], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile / 100
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = index - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    def phase_percentile(self, phase: str, percentile: float) -> float:
        """Return a percentile for an instrumented backend phase."""
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0, 100]")
        return self._percentile(
            tuple(self.phase_latencies_seconds.get(phase, ())), percentile
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


class BenchmarkRunner:
    """Compare extractors with identical inputs and an optional evaluator."""

    def __init__(self, warmup: int = 0, error_policy: str = "raise") -> None:
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if error_policy not in {"raise", "skip"}:
            raise ValueError("error_policy must be raise or skip")
        self.warmup = warmup
        self.error_policy = error_policy

    def run(
        self,
        extractor: Any,
        inputs: Iterable[ImageInput],
        name: str | None = None,
        evaluator: Callable[[SemanticResult, ImageInput], Mapping[str, float]]
        | None = None,
    ) -> BenchmarkReport:
        items = list(inputs)
        for item in items[: self.warmup]:
            self._invoke(extractor, item)
        latencies: list[float] = []
        phase_values: dict[str, list[float]] = {}
        quality_values: dict[str, list[float]] = {}
        failures = 0
        started_total = time.perf_counter()
        for item in items:
            started = time.perf_counter()
            try:
                result = self._invoke(extractor, item)
                elapsed = time.perf_counter() - started
                latencies.append(elapsed)
                timings = result.metadata.get("timings_seconds", {})
                if isinstance(timings, Mapping):
                    for phase, value in timings.items():
                        try:
                            duration = float(value)
                        except (TypeError, ValueError):
                            continue
                        if duration >= 0:
                            phase_values.setdefault(str(phase), []).append(duration)
                if evaluator is not None:
                    for key, value in evaluator(result, item).items():
                        quality_values.setdefault(key, []).append(float(value))
            except Exception:
                failures += 1
                if self.error_policy == "raise":
                    raise
        total = time.perf_counter() - started_total
        quality = {key: mean(values) for key, values in quality_values.items()}
        return BenchmarkReport(
            name=name or getattr(extractor, "__name__", extractor.__class__.__name__),
            samples=len(latencies),
            failures=failures,
            total_seconds=total,
            latencies_seconds=tuple(latencies),
            quality=quality,
            phase_latencies_seconds={
                phase: tuple(values) for phase, values in phase_values.items()
            },
        )

    def compare(
        self,
        extractors: Mapping[str, Any],
        inputs: Iterable[ImageInput],
        evaluator: Callable[[SemanticResult, ImageInput], Mapping[str, float]]
        | None = None,
    ) -> dict[str, BenchmarkReport]:
        """Run every extractor on the same materialized input sequence."""
        items = list(inputs)
        return {
            name: self.run(backend, items, name=name, evaluator=evaluator)
            for name, backend in extractors.items()
        }

    def run_tracking(
        self,
        tracker: Any,
        detections: Iterable[Iterable[Detection]],
        ground_truth: Iterable[Iterable[Detection]],
        name: str | None = None,
        iou_threshold: float = 0.5,
    ) -> BenchmarkReport:
        """Benchmark tracker latency and attach MOTA/IDF1 quality metrics."""
        predicted: list[list[Detection]] = []
        latencies: list[float] = []
        started_total = time.perf_counter()
        for frame in detections:
            started = time.perf_counter()
            predicted.append(list(tracker.update(list(frame))))
            latencies.append(time.perf_counter() - started)
        total = time.perf_counter() - started_total
        metrics = evaluate_tracking(
            predicted, ground_truth, iou_threshold=iou_threshold
        )
        return BenchmarkReport(
            name=name or tracker.__class__.__name__,
            samples=len(latencies),
            failures=0,
            total_seconds=total,
            latencies_seconds=tuple(latencies),
            quality={"mota": metrics.mota, "idf1": metrics.idf1},
        )

    @staticmethod
    def _invoke(extractor: Any, item: ImageInput) -> SemanticResult:
        output = (
            extractor.extract(item)
            if hasattr(extractor, "extract")
            else extractor(item)
        )
        if isinstance(output, SemanticResult):
            return output
        if isinstance(output, Mapping):
            excluded = {
                "depth",
                "segmentation",
                "mask",
                "detections",
                "tags",
                "embeddings",
                "ocr",
                "text",
                "caption",
                "scene",
                "events",
            }
            return SemanticResult(
                depth=output.get("depth"),
                segmentation=output.get("segmentation", output.get("mask")),
                detections=output.get("detections"),
                tags=output.get("tags"),
                embeddings=output.get("embeddings"),
                ocr=output.get("ocr", output.get("text")),
                caption=output.get("caption"),
                scene=output.get("scene"),
                events=output.get("events"),
                keypoints=output.get("keypoints", output.get("poses")),
                relations=output.get("relations"),
                document=output.get("document"),
                depth_map=output.get("depth_map"),
                metadata={
                    key: value for key, value in output.items() if key not in excluded
                },
            )
        return SemanticResult(metadata={"raw_output": output})


__all__ = ["BenchmarkReport", "BenchmarkRunner"]
