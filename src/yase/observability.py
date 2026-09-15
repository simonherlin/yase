"""Dependency-free metrics and optional OpenTelemetry tracing."""

from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from threading import Lock
from typing import Any


def _label_value(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(values: Mapping[str, Any]) -> str:
    if not values:
        return ""
    pairs = [f'{key}="{_label_value(values[key])}"' for key in sorted(values)]
    return "{" + ",".join(pairs) + "}"


class OpenTelemetryTracer:
    """Small optional bridge for OpenTelemetry spans.

    OpenTelemetry is imported only when this class is instantiated. A custom
    tracer object can be injected for tests or an application's configured
    provider; it must expose ``start_as_current_span``.
    """

    def __init__(self, tracer: Any = None, instrumentation_scope: str = "yase"):
        if not isinstance(instrumentation_scope, str) or not instrumentation_scope:
            raise ValueError("instrumentation_scope must be a non-empty string")
        self._status = None
        self._status_code = None
        if tracer is None:
            try:
                from opentelemetry import trace
                from opentelemetry.trace import Status, StatusCode
            except ImportError as exc:
                raise ImportError(
                    "install the observability extra to use OpenTelemetryTracer"
                ) from exc
            tracer = trace.get_tracer(instrumentation_scope)
            self._status = Status
            self._status_code = StatusCode
        if not hasattr(tracer, "start_as_current_span"):
            raise TypeError("tracer must expose start_as_current_span")
        self.tracer = tracer
        self.instrumentation_scope = instrumentation_scope

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Create a current span and annotate failures before re-raising."""
        if not isinstance(name, str) or not name:
            raise ValueError("span name must be a non-empty string")
        with self.tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                setter = getattr(span, "set_attribute", None)
                if callable(setter):
                    setter(
                        str(key),
                        value
                        if isinstance(value, (str, int, float, bool))
                        else str(value),
                    )
            try:
                yield span
            except Exception as exc:
                recorder = getattr(span, "record_exception", None)
                if callable(recorder):
                    recorder(exc)
                if self._status is not None and self._status_code is not None:
                    set_status = getattr(span, "set_status", None)
                    if callable(set_status):
                        set_status(self._status(self._status_code.ERROR, str(exc)))
                raise


class RuntimeMetrics:
    """Thread-safe scheduler metrics with Prometheus exposition.

    The collector accepts a scheduler report by protocol rather than importing
    the scheduler module, keeping this surface usable by custom runtimes.
    """

    def __init__(self, namespace: str = "yase") -> None:
        valid_tail = all(
            character.isascii() and (character.isalnum() or character == "_")
            for character in namespace[1:]
        )
        if (
            not namespace
            or not (
                namespace[0].isascii()
                and (namespace[0].isalpha() or namespace[0] == "_")
            )
            or not valid_tail
        ):
            raise ValueError(
                "namespace must start with a letter or _ and contain only letters, digits, or _"
            )
        self.namespace = namespace
        self._lock = Lock()
        self._runs = 0
        self._cache_hits = 0
        self._elapsed_sum = 0.0
        self._stage_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._stage_elapsed: defaultdict[str, float] = defaultdict(float)
        self._stage_observations: defaultdict[str, int] = defaultdict(int)
        self._video_processed = 0
        self._video_dropped = 0
        self._video_latency_sum = 0.0
        self._video_observations = 0
        self._extractions = 0
        self._extraction_failures = 0
        self._extraction_latency_sum = 0.0

    def record_report(self, report: Any) -> None:
        """Record one scheduler-like report."""
        with self._lock:
            self._runs += 1
            self._cache_hits += int(getattr(report, "cache_hits", 0))
            self._elapsed_sum += float(getattr(report, "elapsed_seconds", 0.0))
            for execution in getattr(report, "stages", ()):
                name = str(execution.name)
                status = str(execution.status)
                self._stage_counts[(name, status)] += 1
                self._stage_elapsed[name] += float(execution.duration_seconds)
                self._stage_observations[name] += 1

    def record_video(self, stats: Any) -> None:
        """Record one completed sequential or realtime video iteration."""
        with self._lock:
            processed = int(getattr(stats, "frames_processed", 0))
            self._video_processed += processed
            self._video_dropped += int(getattr(stats, "frames_dropped", 0))
            self._video_latency_sum += (
                float(getattr(stats, "mean_latency", 0.0)) * processed
            )
            self._video_observations += processed

    def record_extraction(self, duration_seconds: float, *, success: bool) -> None:
        """Record one direct image extraction attempt."""
        with self._lock:
            self._extractions += 1
            if not success:
                self._extraction_failures += 1
            self._extraction_latency_sum += float(duration_seconds)

    def snapshot(self) -> dict[str, Any]:
        """Return a copy suitable for JSON logging or dashboards."""
        with self._lock:
            return {
                "runs": self._runs,
                "cache_hits": self._cache_hits,
                "elapsed_seconds_sum": self._elapsed_sum,
                "stage_counts": {
                    f"{name}:{status}": count
                    for (name, status), count in self._stage_counts.items()
                },
                "stage_elapsed_seconds": dict(self._stage_elapsed),
                "stage_observations": dict(self._stage_observations),
                "video": {
                    "frames_processed": self._video_processed,
                    "frames_dropped": self._video_dropped,
                    "latency_seconds_sum": self._video_latency_sum,
                    "latency_observations": self._video_observations,
                },
                "extraction": {
                    "attempts": self._extractions,
                    "failures": self._extraction_failures,
                    "latency_seconds_sum": self._extraction_latency_sum,
                },
            }

    def prometheus_text(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        with self._lock:
            prefix = self.namespace
            lines = [
                f"# TYPE {prefix}_scheduler_runs_total counter",
                f"{prefix}_scheduler_runs_total {self._runs}",
                f"# TYPE {prefix}_scheduler_cache_hits_total counter",
                f"{prefix}_scheduler_cache_hits_total {self._cache_hits}",
                f"# TYPE {prefix}_scheduler_elapsed_seconds summary",
                f"{prefix}_scheduler_elapsed_seconds_sum {self._elapsed_sum:.9g}",
                f"{prefix}_scheduler_elapsed_seconds_count {self._runs}",
                f"# TYPE {prefix}_scheduler_stage_executions_total counter",
            ]
            for (name, status), count in sorted(self._stage_counts.items()):
                lines.append(
                    f"{prefix}_scheduler_stage_executions_total"
                    f"{_labels({'stage': name, 'status': status})} {count}"
                )
            lines.append(f"# TYPE {prefix}_scheduler_stage_duration_seconds summary")
            for name in sorted(self._stage_observations):
                labels = _labels({"stage": name})
                lines.append(
                    f"{prefix}_scheduler_stage_duration_seconds_sum{labels} "
                    f"{self._stage_elapsed[name]:.9g}"
                )
                lines.append(
                    f"{prefix}_scheduler_stage_duration_seconds_count{labels} "
                    f"{self._stage_observations[name]}"
                )
            lines.extend(
                [
                    f"# TYPE {prefix}_video_frames_processed_total counter",
                    f"{prefix}_video_frames_processed_total {self._video_processed}",
                    f"# TYPE {prefix}_video_frames_dropped_total counter",
                    f"{prefix}_video_frames_dropped_total {self._video_dropped}",
                    f"# TYPE {prefix}_video_frame_latency_seconds summary",
                    f"{prefix}_video_frame_latency_seconds_sum "
                    f"{self._video_latency_sum:.9g}",
                    f"{prefix}_video_frame_latency_seconds_count "
                    f"{self._video_observations}",
                    f"# TYPE {prefix}_extraction_attempts_total counter",
                    f"{prefix}_extraction_attempts_total {self._extractions}",
                    f"# TYPE {prefix}_extraction_failures_total counter",
                    f"{prefix}_extraction_failures_total {self._extraction_failures}",
                    f"# TYPE {prefix}_extraction_latency_seconds summary",
                    f"{prefix}_extraction_latency_seconds_sum "
                    f"{self._extraction_latency_sum:.9g}",
                    f"{prefix}_extraction_latency_seconds_count {self._extractions}",
                ]
            )
            return "\n".join(lines) + "\n"


__all__ = ["OpenTelemetryTracer", "RuntimeMetrics"]
