# Architecture

Yase is split into three layers:

1. core.py defines the stable Yase facade, SemanticResult, image
   conversion, and the Extractor protocol. It imports only NumPy and Pillow.
2. backends.py contains explicit adapters. CallableExtractor is useful for
   tests and custom inference; TorchScriptExtractor loads a local model only
   when constructed and requires the torch extra.
3. video.py owns capture iteration. VideoStream reads sequentially, applies
   stride/FPS policies, emits FrameResult, and records VideoStats.

A backend may return a typed result, a mapping, or an array. The result schema
supports depth, segmentation, detections, tags, embeddings, timestamps, and
backend metadata without coupling the core to a particular model family.

OpenCV is imported lazily when a path or camera source is used. A capture-like
object with read, get, and release methods enables deterministic testing
and integration with other camera systems.

Video processing is deliberately sequential. stride and max_fps discard
frames before inference; error_policy=skip or on_error can keep a stream
alive after backend failures. This avoids hidden worker threads and unbounded
queues. Applications needing a latest-frame worker can wrap their camera source
with a one-item queue and pass that capture-like object to VideoStream.
