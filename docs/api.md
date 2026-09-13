# API reference

## Yase

- Yase(task=depth, extractor=...) constructs a facade.
- extract(image, timestamp=None) accepts a path, Pillow image, or NumPy
  array and returns SemanticResult.
- run_inference and __call__ are compatibility aliases.

## SemanticResult

Typed fields include optional depth, segmentation, detections, tags,
embeddings, timestamp, and arbitrary metadata.

## VideoStream

VideoStream(source, extractor, stride=1, max_fps=None, max_frames=None)
yields FrameResult. stats exposes read/processed/dropped counts, source and
output FPS, elapsed time, and mean/max inference latency.

Use error_policy=skip to discard failed frames, or pass
on_error(exception, frame_index) to return a recovery SemanticResult.
Exceptions are raised by default.

See examples/image.py and examples/video.py for runnable minimal adapters.
