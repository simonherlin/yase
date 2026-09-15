# Architecture

Yase is split into a lightweight core and optional capability layers:

1. `core.py` defines the stable Yase facade, `SemanticResult`, image
   conversion, and the `Extractor` protocol. It imports only NumPy and Pillow.
2. `schema.py` contains model-neutral boxes, detections, poses, metric depth,
   scene-graph relations, OCR regions, and temporal events.
3. `backends.py` contains explicit adapters. `CallableExtractor` is useful for
   tests and custom inference; TorchScript and ONNX adapters load local model
   artifacts only when constructed.
4. `pipeline.py` composes named independent stages without importing model
   frameworks.
5. `index.py` provides a NumPy cosine-search baseline for embeddings.
6. `events.py` provides stream-local context and deterministic presence rules.
7. `tracking.py` provides a dependency-free IoU tracker baseline.
8. `serialization.py` exports results to JSON and JSONL.
9. `video.py` owns capture iteration. `VideoStream` reads sequentially,
   applies stride/FPS policies, emits `FrameResult`, and records `VideoStats`.
10. `reid.py` provides an optional NumPy-only global identity store for
    cross-camera appearance matching.
11. `metrics.py` provides deterministic detection and tracking regression
    metrics; `models.py` records model provenance and redistribution notes.
12. `observation.py` defines the versioned `FrameRef`, `ObservationBundle`,
    `ModelProvenance`, `Uncertainty`, and `EmbeddingRecord` contracts.
13. `stages.py` defines the declarative `StageSpec`/`StageContext` protocol and
    validates ordered dependencies before the future DAG scheduler runs.
14. `scheduler.py` executes contracted stages as a topological graph with
    bounded caching, cancellation, deadlines, async entry points, and stage
    telemetry.
15. `sinks.py` provides dependency-free callback, memory, JSONL, fan-out, and
    bounded-queue boundaries for downstream systems.
16. `diagnostics.py` provides lazy runtime capability reports and readiness
    checks without importing optional frameworks.
17. `native.py` exposes optional C++17 acceleration for tracking hot paths;
    the public Python fallback remains deterministic when no compiler artifact
    is installed.

A backend may return a typed result, a mapping, or an array. The result schema
supports depth, segmentation, detections, tags, embeddings, OCR, captions,
scene data, events, timestamps, and backend metadata without coupling the core
to a particular model family. Heavy model integrations remain explicit plugins
so a CPU-only installation stays small and deterministic.

`SemanticResult` remains the compatibility object for low-latency one-shot
calls. Long-lived systems should promote it to an `ObservationBundle`: this
prevents frame identity, model artifact identity, calibration state, and
uncertainty from being lost between a video reader, a stage, and a sink.

OpenCV is imported lazily when a path or camera source is used. A capture-like
object with `read`, `get`, and `release` methods enables deterministic testing
and integration with other camera systems.

VideoStream processing is sequential: stride and max_fps discard frames before
inference. RealtimeVideoStream uses a capture worker and one-item
condition-protected buffer; `drop_frames=True` overwrites stale frames, while
`False` applies backpressure. `error_policy=skip` or `on_error` can keep either
stream alive after backend failures. `close()` signals and joins the worker.

Event rules are stream-local and dependency-free. Zone and line rules consume
tracked bounding boxes, while external trackers can be plugged through
`ExternalTrackerAdapter` when an application chooses official ByteTrack,
BoT-SORT, or a hardware-specific implementation.
