# Yase

Yase (Yet Another Semantic Extractor) is a Python library for extracting
semantic signals from images and video streams. The core API is lightweight:
NumPy and Pillow are enough for image extraction, while video capture and
model backends are optional extras.

## Install

~~~bash
uv sync --dev
# Optional live video capture:
uv sync --extra video
# Optional local TorchScript backend:
uv sync --extra torch
~~~

The package can also be installed from a built wheel with
uv pip install dist/yase-*.whl. Python 3.9 and newer are supported.

## Image extraction

Inject any callable backend in applications and tests. A backend can return a
SemanticResult, a mapping with depth/segmentation keys, or a NumPy array.

~~~python
import numpy as np
from yase import Yase

class MeanDepth:
    def extract(self, image):
        return {"depth": image.mean(axis=2, dtype=np.float32)}

extractor = Yase(extractor=MeanDepth())
result = extractor.extract(np.zeros((480, 640, 3), dtype=np.uint8))
print(result.depth.shape)
~~~
Yase accepts paths, Pillow images, and NumPy arrays. Arrays are treated as RGB
by default; pass color_order="BGR" when adapting OpenCV frames.

For a real model, use the local TorchScript adapter. It loads no weights from
the network and is initialized only when explicitly supplied:

~~~python
from yase import Yase
model = Yase(model="torchscript", model_path="model.pt")
depth = model.extract("photo.jpg").depth
~~~

This backend requires the `torch` extra and a model file supplied locally by
the application. Yase never downloads weights implicitly. For ONNX Runtime,
install the `onnx` extra and provide a local model (also without downloads):

~~~python
from yase import OnnxRuntimeExtractor
backend = OnnxRuntimeExtractor("model.onnx", task="depth", size=(640, 192))
result = backend.extract("photo.jpg")
~~~

## Batch extraction

`extract_many` preserves input order and timestamps. Backends that implement
`extract_batch(images)` are called once for efficient vectorized inference;
all other extractors use the same stable API with an image-by-image fallback.

~~~python
results = extractor.extract_many(
    ["frame-001.jpg", "frame-002.jpg"],
    error_policy="skip",
)
for result in results:
    if result is not None:
        print(result.depth.shape)
~~~

Use `CompositeExtractor` to merge depth, segmentation, detections, tags, and
embeddings from independent model backends into one `SemanticResult`.

## Real-time video

VideoStream yields typed FrameResult values with frame index, source timestamp,
processing duration, and SemanticResult. It supports frame stride, max output
FPS, and bounded output count.

~~~python
from yase import RealtimeVideoStream, Yase

extractor = Yase(extractor=MeanDepth())
for item in RealtimeVideoStream(0, extractor, drop_frames=True):
    print(item.frame_index, item.timestamp, item.result.depth.shape)
~~~
OpenCV is imported only when a path or camera source is opened. A capture-like
object implementing read() and release() can be supplied for custom pipelines.
For file processing use VideoStream; for live cameras use RealtimeVideoStream,
which runs capture in a worker and keeps a bounded latest-frame buffer. Set
drop_frames=False to apply backpressure instead of discarding stale frames.

## Development

~~~bash
uv sync --dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build
~~~

The test suite uses fake backends and captures, so it never downloads model
weights. See docs/devguide for release and quality guidance.

## Project status

The injectable core, generic result schema, TorchScript adapter, and video
iterator are stable. Segmentation, ONNX, and specialized detection models are
backend integrations; use CallableExtractor or a custom Extractor until those
backends are configured for your deployment.
