"""Dependency-free observation sinks for archives and application plumbing."""

from collections.abc import Callable, Iterable
from pathlib import Path
from queue import Full, Queue
from typing import Any, Protocol, TextIO, cast

from .observation import ObservationBundle, observation_to_json


class ObservationSink(Protocol):
    """Minimal sink protocol used by stream applications."""

    def emit(self, observation: ObservationBundle) -> bool: ...

    def close(self) -> None: ...


class CallbackSink:
    """Adapt a synchronous callback into an observation sink."""

    def __init__(self, callback: Callable[[ObservationBundle], Any]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.callback = callback

    def emit(self, observation: ObservationBundle) -> bool:
        _validate_observation(observation)
        self.callback(observation)
        return True

    def close(self) -> None:
        return None


class MemorySink:
    """Collect observations for tests, notebooks, or small local jobs."""

    def __init__(self, max_items: int | None = None) -> None:
        if max_items is not None and max_items < 0:
            raise ValueError("max_items cannot be negative")
        self.observations: list[ObservationBundle] = []
        self.max_items = max_items

    def emit(self, observation: ObservationBundle) -> bool:
        _validate_observation(observation)
        if self.max_items is not None and len(self.observations) >= self.max_items:
            return False
        self.observations.append(observation)
        return True

    def close(self) -> None:
        return None


class JsonlObservationSink:
    """Incrementally write compact, deterministic observation JSONL."""

    def __init__(
        self,
        destination: str | Path | TextIO,
        *,
        include_arrays: bool = False,
        flush_each: bool = False,
    ) -> None:
        self.include_arrays = include_arrays
        self.flush_each = flush_each
        if isinstance(destination, (str, Path)):
            self._owns_handle = True
            self._handle: TextIO = cast(
                TextIO, open(str(destination), "w", encoding="utf-8")
            )
        else:
            self._owns_handle = False
            self._handle = destination
        self._closed = False

    def emit(self, observation: ObservationBundle) -> bool:
        _validate_observation(observation)
        if self._closed:
            raise RuntimeError("sink is closed")
        self._handle.write(
            observation_to_json(observation, include_arrays=self.include_arrays) + "\n"
        )
        if self.flush_each:
            self._handle.flush()
        return True

    def close(self) -> None:
        if not self._closed:
            self._handle.flush()
            if self._owns_handle:
                self._handle.close()
            self._closed = True

    def __enter__(self) -> "JsonlObservationSink":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class QueueSink:
    """Bounded queue sink with explicit backpressure or drop behavior."""

    def __init__(self, maxsize: int = 128, on_full: str = "block") -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        if on_full not in ("block", "drop"):
            raise ValueError("on_full must be 'block' or 'drop'")
        self.queue: Queue[ObservationBundle] = Queue(maxsize=maxsize)
        self.on_full = on_full
        self.dropped = 0
        self._closed = False

    def emit(self, observation: ObservationBundle) -> bool:
        _validate_observation(observation)
        if self._closed:
            raise RuntimeError("sink is closed")
        try:
            if self.on_full == "drop":
                self.queue.put_nowait(observation)
            else:
                self.queue.put(observation)
        except Full:
            self.dropped += 1
            return False
        return True

    def get(self, timeout: float | None = None) -> ObservationBundle:
        """Consume one observation, raising ``queue.Empty`` when configured."""
        return self.queue.get(timeout=timeout)

    def task_done(self) -> None:
        self.queue.task_done()

    def close(self) -> None:
        self._closed = True


class FanoutSink:
    """Send one observation to every sink and report partial backpressure."""

    def __init__(self, sinks: Iterable[ObservationSink]) -> None:
        self.sinks = tuple(sinks)
        if not self.sinks:
            raise ValueError("at least one sink is required")

    def emit(self, observation: ObservationBundle) -> bool:
        statuses = [sink.emit(observation) for sink in self.sinks]
        return all(statuses)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


def _validate_observation(observation: ObservationBundle) -> None:
    if not isinstance(observation, ObservationBundle):
        raise TypeError("sink expects an ObservationBundle")


__all__ = [
    "CallbackSink",
    "FanoutSink",
    "JsonlObservationSink",
    "MemorySink",
    "ObservationSink",
    "QueueSink",
]
