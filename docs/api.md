# API reference

## Yase

- Yase(task=depth, extractor=...) constructs a facade.
- extract(image, timestamp=None) accepts a path, Pillow image, or NumPy
  array and returns SemanticResult.
- run_inference and __call__ are compatibility aliases.

## SemanticResult

Typed fields include optional depth, segmentation, detections, tags,
embeddings, timestamp, and arbitrary metadata.

## Backends

CallableExtractor wraps a Python callable. TorchScriptExtractor loads a local
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

See examples/image.py and examples/video.py for runnable minimal adapters.
