"""Real-time video iteration utilities."""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional, Union

from .core import SemanticResult, load_image


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
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if max_fps is not None and max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be >= 0")
        self.source = source
        self.extractor = extractor
        self.stride = stride
        self.max_fps = max_fps
        self.drop_frames = drop_frames
        self.color_order = color_order
        self.max_frames = max_frames
        self._capture = None
        self._fps = 0.0
        self._stats = VideoStats(0, 0, 0, 0.0, 0.0, 0.0)

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

    def __iter__(self) -> Iterator[FrameResult]:
        capture = self._open()
        started_at = time.perf_counter()
        frames_read = frames_processed = frames_dropped = 0
        yielded = 0
        last_timestamp: Optional[float] = None
        index = 0
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
                if (
                    self.max_fps is not None
                    and last_timestamp is not None
                    and timestamp - last_timestamp < 1.0 / self.max_fps
                ):
                    frames_dropped += 1
                    index += 1
                    continue
                started = time.perf_counter()
                image = load_image(frame, color_order=self.color_order)
                backend = self.extractor
                if hasattr(backend, "extract"):
                    result = backend.extract(image, timestamp=timestamp)
                else:
                    result = backend(image)
                if not isinstance(result, SemanticResult):
                    result = SemanticResult(depth=result, timestamp=timestamp)
                elif result.timestamp is None:
                    result = SemanticResult(
                        depth=result.depth,
                        segmentation=result.segmentation,
                        detections=result.detections,
                        tags=result.tags,
                        embeddings=result.embeddings,
                        timestamp=timestamp,
                        metadata=result.metadata,
                    )
                yield FrameResult(
                    index, timestamp, result, time.perf_counter() - started
                )
                last_timestamp = timestamp
                yielded += 1
                frames_processed += 1
                index += 1
        finally:
            elapsed = time.perf_counter() - started_at
            output_fps = frames_processed / elapsed if elapsed else 0.0
            self._stats = VideoStats(
                frames_read,
                frames_processed,
                frames_dropped,
                elapsed,
                self._fps,
                output_fps,
            )
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
