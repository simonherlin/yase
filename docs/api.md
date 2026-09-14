# API reference

## Yase

- Yase(task=depth, extractor=...) constructs a facade.
- extract(image, timestamp=None) accepts a path, Pillow image, or NumPy
  array and returns SemanticResult.
- extract_many(images, timestamps=None, error_policy=raise, on_error=None)
  preserves input order and uses a backend's native extract_batch method when
  available. With error_policy=skip, failed positions contain None.
- run_inference and __call__ are compatibility aliases.

## SemanticResult

Typed fields include optional depth, segmentation, detections, tags,
embeddings, timestamp, and arbitrary metadata.

## Backends

CallableExtractor wraps a Python callable. CompositeExtractor combines named
backends and supports error, first, or last conflict policies while preserving
all SemanticResult fields and timestamps. TorchScriptExtractor loads a local
TorchScript file with the torch extra. OnnxRuntimeExtractor loads a local ONNX
file with the onnx extra, prepares NCHW float32 input, and supports depth,
segmentation, or paired outputs. Inject a session object in tests to avoid the
optional runtime dependency.

## VideoStream and RealtimeVideoStream

VideoStream(source, extractor, stride=1, max_fps=None, max_frames=None) yields
FrameResult sequentially. RealtimeVideoStream uses a worker and one-item
latest-frame buffer; drop_frames=True overwrites stale frames, while False
applies backpressure. Both expose stats with read/processed/dropped counts,
source/output FPS, elapsed time, and mean/max inference latency.

Use error_policy=skip to discard failed frames, or pass
on_error(exception, frame_index) to return a recovery SemanticResult.
Exceptions are raised by default.

See examples/image.py, examples/batch.py, examples/composite.py,
examples/onnx.py, and examples/video.py for runnable adapters.
