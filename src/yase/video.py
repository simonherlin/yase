"""Real-time video iteration utilities."""

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Union

from .core import SemanticResult, _normalise_output, load_image
from .limits import InputLimits
from .observability import RuntimeMetrics


@dataclass(frozen=True)
class FrameResult:
    """Result for one processed video frame."""

    frame_index: int
    timestamp: float
    result: SemanticResult
    processing_time: float


@dataclass(frozen=True)
class VideoStats:
    """Counters and rates collected during one iteration."""

    frames_read: int
    frames_processed: int
    frames_dropped: int
    elapsed: float
    input_fps: float
    output_fps: float
    mean_latency: float
    max_latency: float


class VideoStream:
    """Read frames and run an extractor with predictable frame pacing.

    OpenCV is optional and imported only when iteration starts. stride processes
    every Nth frame. max_fps drops frames that exceed the output rate.
    Iteration is intentionally sequential: drop_frames documents that skipped
    frames are not queued, while callers needing a worker thread can provide a
    capture-like source that implements that policy.
    """

    def __init__(
        self,
        source: Union[str, int, Any],
        extractor: Any,
        stride: int = 1,
        max_fps: Optional[float] = None,
        drop_frames: bool = True,
        color_order: str = "BGR",
        max_frames: Optional[int] = None,
        batch_size: int = 1,
        on_error: Optional[Callable[[Exception, int], Optional[SemanticResult]]] = None,
        error_policy: str = "raise",
        tracker: Optional[Any] = None,
        memory: Optional[Any] = None,
        identity_store: Optional[Any] = None,
        camera_id: str = "default",
        event_engine: Optional[Any] = None,
        sink: Optional[Any] = None,
        source_id: str = "default",
        input_limits: Optional[InputLimits] = None,
        metrics: Optional[RuntimeMetrics] = None,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if max_fps is not None and max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be >= 0")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if error_policy not in ("raise", "skip"):
            raise ValueError("error_policy must be raise or skip")
        self.source = source
        self.extractor = extractor
        self.stride = stride
        self.max_fps = max_fps
        self.drop_frames = drop_frames
        self.color_order = color_order
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.on_error = on_error
        self.error_policy = error_policy
        self.tracker = tracker
        self.memory = memory
        self.identity_store = identity_store
        self.camera_id = str(camera_id)
        self.event_engine = event_engine
        self.sink = sink
        self.source_id = str(source_id)
        self.input_limits = input_limits
        self.metrics = metrics
        self._capture = None
        self._fps = 0.0
        self._stats = VideoStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def fps(self) -> float:
        """Source FPS, or zero when the source does not report one."""
        return self._fps

    @property
    def stats(self) -> VideoStats:
        """Counters from the most recent iteration."""
        return self._stats

    def _open(self) -> Any:
        if hasattr(self.source, "read"):
            capture = self.source
        else:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError(
                    "opencv-python is required for video sources"
                ) from exc
            capture = cv2.VideoCapture(self.source)
        if hasattr(capture, "isOpened") and not capture.isOpened():
            raise RuntimeError("could not open video source")
        try:
            import cv2

            self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        except (ImportError, AttributeError):
            self._fps = 0.0
        self._capture = capture
        return capture

    def _timestamp(self, capture: Any, index: int) -> float:
        try:
            import cv2

            value = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if value > 0:
                return value
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        if self._fps > 0:
            return index / self._fps
        return time.monotonic()

    def _track(self, result: SemanticResult, timestamp: float) -> SemanticResult:
        if self.tracker is None or result.detections is None:
            return result
        if not hasattr(self.tracker, "update"):
            raise TypeError("tracker must expose update(detections, timestamp=...)")
        try:
            detections = self.tracker.update(result.detections, timestamp=timestamp)
        except TypeError:
            detections = self.tracker.update(result.detections)
        return replace(result, detections=detections)

    def _events(self, result: SemanticResult, timestamp: float) -> SemanticResult:
        if self.event_engine is None:
            return result
        if not hasattr(self.event_engine, "update"):
            raise TypeError("event_engine must expose update(result, timestamp)")
        emitted = self.event_engine.update(result, timestamp)
        if not emitted:
            return result
        existing = list(result.events or ())
        return replace(result, events=existing + list(emitted))

    def _identity(self, result: SemanticResult, timestamp: float) -> SemanticResult:
        if self.identity_store is None or result.detections is None:
            return result
        if not hasattr(self.identity_store, "update"):
            raise TypeError("identity_store must expose update(detections, ...)")
        try:
            detections = self.identity_store.update(
                result.detections, camera_id=self.camera_id, timestamp=timestamp
            )
        except TypeError:
            detections = self.identity_store.update(result.detections)
        return replace(result, detections=detections)

    def _memory(self, result: SemanticResult, timestamp: float) -> SemanticResult:
        if self.memory is None:
            return result
        if not hasattr(self.memory, "update"):
            raise TypeError("memory must expose update(result, timestamp=...)")
        try:
            return self.memory.update(result, timestamp=timestamp)
        except TypeError:
            return self.memory.update(result)

    @staticmethod
    def _coerce_output(output: Any, timestamp: float) -> SemanticResult:
        task = "semantic" if isinstance(output, Mapping) else "depth"
        return _normalise_output(output, task, timestamp)

    def _extract_batch(
        self, images: list[Any], indices: list[int], timestamps: list[float]
    ) -> list[Optional[SemanticResult]]:
        """Extract a frame batch while keeping per-frame error semantics."""
        backend = self.extractor
        try:
            if hasattr(backend, "extract_batch"):
                outputs = list(backend.extract_batch(images))
                if len(outputs) != len(images):
                    raise ValueError(
                        "extract_batch must return one result per video frame"
                    )
            else:
                outputs = [
                    backend.extract(image)
                    if hasattr(backend, "extract")
                    else backend(image)
                    for image in images
                ]
            results: list[Optional[SemanticResult]] = []
            for output, timestamp in zip(outputs, timestamps):
                results.append(self._coerce_output(output, timestamp))
            return results
        except Exception:
            if self.error_policy == "raise" and self.on_error is None:
                raise

        # A failed batch must be retried frame by frame: a single corrupt frame
        # should not discard otherwise valid frames when skip/recovery is used.
        recovered: list[Optional[SemanticResult]] = []
        for image, index, timestamp in zip(images, indices, timestamps):
            try:
                output = (
                    backend.extract(image)
                    if hasattr(backend, "extract")
                    else backend(image)
                )
                recovered.append(self._coerce_output(output, timestamp))
            except Exception as exc:
                if self.on_error is not None:
                    recovered.append(self.on_error(exc, index))
                else:
                    recovered.append(None)
        return recovered

    def _emit_observation(
        self, result: SemanticResult, frame_index: int, timestamp: float, image: Any
    ) -> None:
        if self.sink is None:
            return
        from .observation import FrameRef, ObservationBundle

        if not hasattr(self.sink, "emit"):
            raise TypeError("sink must expose emit(observation)")
        array = load_image(image, limits=self.input_limits)
        self.sink.emit(
            ObservationBundle.from_result(
                result,
                frame=FrameRef(
                    frame_id=frame_index,
                    source_id=self.source_id,
                    timestamp=timestamp,
                    width=array.shape[1],
                    height=array.shape[0],
                ),
            )
        )

    def __iter__(self) -> Iterator[FrameResult]:
        capture = self._open()
        started_at = time.perf_counter()
        frames_read = frames_processed = frames_dropped = 0
        yielded = 0
        last_timestamp: Optional[float] = None
        index = 0
        latencies = []
        pending: list[tuple[int, float, Any]] = []

        def process_pending() -> Iterator[FrameResult]:
            nonlocal yielded, frames_dropped, frames_processed, last_timestamp
            if not pending:
                return
            batch = list(pending)
            pending.clear()
            indices = [item[0] for item in batch]
            timestamps = [item[1] for item in batch]
            raw_frames = [item[2] for item in batch]
            images: list[Optional[Any]] = [None] * len(batch)
            results: list[Optional[SemanticResult]] = [None] * len(batch)
            valid_positions = []
            for position, frame in enumerate(raw_frames):
                try:
                    images[position] = load_image(
                        frame,
                        color_order=self.color_order,
                        limits=self.input_limits,
                    )
                    valid_positions.append(position)
                except Exception as exc:
                    if self.on_error is not None:
                        results[position] = self.on_error(exc, indices[position])
                    elif self.error_policy == "skip":
                        continue
                    else:
                        raise
            started = time.perf_counter()
            if valid_positions:
                extracted = self._extract_batch(
                    [images[position] for position in valid_positions],
                    [indices[position] for position in valid_positions],
                    [timestamps[position] for position in valid_positions],
                )
                for position, result in zip(valid_positions, extracted):
                    results[position] = result
            batch_latency = time.perf_counter() - started
            per_frame_latency = batch_latency / max(1, len(valid_positions))
            for position, (index, timestamp, image, result) in enumerate(
                zip(indices, timestamps, images, results)
            ):
                if result is None:
                    frames_dropped += 1
                    continue
                result = self._track(result, timestamp)
                result = self._memory(result, timestamp)
                result = self._identity(result, timestamp)
                result = self._events(result, timestamp)
                if image is not None:
                    self._emit_observation(result, index, timestamp, image)
                latencies.append(per_frame_latency)
                yield FrameResult(index, timestamp, result, per_frame_latency)
                last_timestamp = timestamp
                yielded += 1
                frames_processed += 1

        try:
            while self.max_frames is None or yielded < self.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frames_read += 1
                if index % self.stride:
                    frames_dropped += 1
                    index += 1
                    continue
                timestamp = self._timestamp(capture, index)
                pacing_timestamp = pending[-1][1] if pending else last_timestamp
                if (
                    self.max_fps is not None
                    and pacing_timestamp is not None
                    and timestamp - pacing_timestamp < 1.0 / self.max_fps
                ):
                    frames_dropped += 1
                    index += 1
                    continue
                pending.append((index, timestamp, frame))
                index += 1
                if len(pending) >= self.batch_size or (
                    self.max_frames is not None
                    and yielded + len(pending) >= self.max_frames
                ):
                    yield from process_pending()
            yield from process_pending()
        finally:
            elapsed = time.perf_counter() - started_at
            output_fps = frames_processed / elapsed if elapsed else 0.0
            mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
            max_latency = max(latencies) if latencies else 0.0
            self._stats = VideoStats(
                frames_read,
                frames_processed,
                frames_dropped,
                elapsed,
                self._fps,
                output_fps,
                mean_latency,
                max_latency,
            )
            if self.metrics is not None:
                self.metrics.record_video(self._stats)
            if hasattr(capture, "release"):
                capture.release()

    def close(self) -> None:
        if self._capture is not None and hasattr(self._capture, "release"):
            self._capture.release()
            self._capture = None


def process_video(
    source: Union[str, int, Any], extractor: Any, **kwargs: Any
) -> Iterator[FrameResult]:
    """Convenience generator equivalent to VideoStream."""
    return iter(VideoStream(source, extractor, **kwargs))


__all__ = ["FrameResult", "VideoStats", "VideoStream", "process_video"]


class RealtimeVideoStream(VideoStream):
    """Capture frames in a worker while inference consumes the latest frame.

    With drop_frames=True (the default), the one-element buffer is overwritten
    by the reader and stale frames are counted as dropped. With drop_frames=False
    the reader waits for the consumer, applying backpressure instead. close()
    is idempotent and joins the worker before releasing the capture.
    """

    def __init__(
        self,
        source: Union[str, int, Any],
        extractor: Any,
        queue_size: int = 1,
        drop_frames: bool = True,
        color_order: str = "BGR",
        max_frames: Optional[int] = None,
        on_error: Optional[Callable[[Exception, int], Optional[SemanticResult]]] = None,
        error_policy: str = "raise",
        tracker: Optional[Any] = None,
        memory: Optional[Any] = None,
        identity_store: Optional[Any] = None,
        camera_id: str = "default",
        event_engine: Optional[Any] = None,
        sink: Optional[Any] = None,
        source_id: str = "default",
        input_limits: Optional[InputLimits] = None,
        metrics: Optional[RuntimeMetrics] = None,
    ) -> None:
        if queue_size != 1:
            raise ValueError("latest-frame mode requires queue_size=1")
        super().__init__(
            source,
            extractor,
            stride=1,
            max_fps=None,
            drop_frames=drop_frames,
            color_order=color_order,
            max_frames=max_frames,
            on_error=on_error,
            error_policy=error_policy,
            tracker=tracker,
            memory=memory,
            identity_store=identity_store,
            camera_id=camera_id,
            event_engine=event_engine,
            sink=sink,
            source_id=source_id,
            input_limits=input_limits,
            metrics=metrics,
        )
        import threading

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._worker = None
        self._latest = None
        self._reader_done = False
        self._reader_error: Optional[BaseException] = None
        self._read_count = 0
        self._drop_count = 0

    def _reader(self, capture: Any) -> None:
        index = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = self._timestamp(capture, index)
                with self._condition:
                    while (
                        not self.drop_frames
                        and self._latest is not None
                        and not self._stop_event.is_set()
                    ):
                        self._condition.wait(timeout=0.1)
                    if self._stop_event.is_set():
                        break
                    if self._latest is not None:
                        self._drop_count += 1
                    self._latest = (index, timestamp, frame)
                    self._read_count += 1
                    index += 1
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._reader_error = exc
        finally:
            with self._condition:
                self._reader_done = True
                self._condition.notify_all()

    def __iter__(self) -> Iterator[FrameResult]:
        capture = self._open()
        import threading

        self._stop_event.clear()
        self._reader_done = False
        self._reader_error = None
        self._read_count = 0
        self._drop_count = 0
        self._latest = None
        self._worker = threading.Thread(
            target=self._reader, args=(capture,), name="yase-capture", daemon=True
        )
        self._worker.start()
        started_at = time.perf_counter()
        frames_processed = 0
        latencies = []
        try:
            while self.max_frames is None or frames_processed < self.max_frames:
                with self._condition:
                    while (
                        self._latest is None
                        and not self._reader_done
                        and self._reader_error is None
                    ):
                        self._condition.wait(timeout=0.1)
                    if self._reader_error is not None:
                        raise RuntimeError(
                            "video capture worker failed"
                        ) from self._reader_error
                    if self._latest is None and self._reader_done:
                        break
                    item = self._latest
                    self._latest = None
                    self._condition.notify_all()
                index, timestamp, frame = item
                started = time.perf_counter()
                try:
                    image = load_image(
                        frame,
                        color_order=self.color_order,
                        limits=self.input_limits,
                    )
                    backend = self.extractor
                    result = (
                        backend.extract(image, timestamp=timestamp)
                        if hasattr(backend, "extract")
                        else backend(image)
                    )
                except Exception as exc:
                    if self.on_error is not None:
                        result = self.on_error(exc, index)
                        if result is None:
                            continue
                    elif self.error_policy == "skip":
                        continue
                    else:
                        raise
                if not isinstance(result, SemanticResult):
                    result = SemanticResult(depth=result, timestamp=timestamp)
                else:
                    result = result.with_timestamp(timestamp)
                result = self._track(result, timestamp)
                result = self._memory(result, timestamp)
                result = self._identity(result, timestamp)
                result = self._events(result, timestamp)
                self._emit_observation(result, index, timestamp, image)
                latency = time.perf_counter() - started
                latencies.append(latency)
                frames_processed += 1
                yield FrameResult(index, timestamp, result, latency)
        finally:
            self.close()
            elapsed = time.perf_counter() - started_at
            output_fps = frames_processed / elapsed if elapsed else 0.0
            mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
            max_latency = max(latencies) if latencies else 0.0
            self._stats = VideoStats(
                self._read_count,
                frames_processed,
                self._drop_count,
                elapsed,
                self._fps,
                output_fps,
                mean_latency,
                max_latency,
            )
            if self.metrics is not None:
                self.metrics.record_video(self._stats)

    def close(self) -> None:
        """Signal the reader and release resources; safe to call repeatedly."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._capture is not None and hasattr(self._capture, "release"):
            self._capture.release()
        self._capture = None
        self._worker = None


__all__ = [
    "FrameResult",
    "RealtimeVideoStream",
    "VideoStats",
    "VideoStream",
    "process_video",
]
