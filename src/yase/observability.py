"""Dependency-free runtime metrics for semantic pipelines."""

from collections import defaultdict
from collections.abc import Mapping
from threading import Lock
from typing import Any


def _label_value(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(values: Mapping[str, Any]) -> str:
    if not values:
        return ""
    pairs = [f'{key}="{_label_value(values[key])}"' for key in sorted(values)]
    return "{" + ",".join(pairs) + "}"


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
            return "\n".join(lines) + "\n"


__all__ = ["RuntimeMetrics"]
