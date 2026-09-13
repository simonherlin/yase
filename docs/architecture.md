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

VideoStream processing is sequential: stride and max_fps discard frames
before inference. RealtimeVideoStream uses a capture worker and one-item
condition-protected buffer; drop_frames=True overwrites stale frames, while
False applies backpressure. error_policy=skip or on_error can keep either
stream alive after backend failures. close() signals and joins the worker.
