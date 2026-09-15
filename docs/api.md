# API reference

## Yase

- Yase(task=depth, extractor=...) constructs a facade. `model="torchscript"`,
  `model="onnx"`, `model="openvino"`, and `model="tensorrt"` are explicit
  local-artifact shortcuts.
- extract(image, timestamp=None) accepts a path, Pillow image, or NumPy
  array and returns SemanticResult.
- extract_many(images, timestamps=None, error_policy=raise, on_error=None)
  preserves input order and uses a backend's native extract_batch method when
  available. With error_policy=skip, malformed batch inputs are isolated so
  valid images still use native batching; failed positions contain None.
- `extract_many(..., max_workers=N)` can run non-batch backends concurrently
  with bounded workers while preserving result order; the default is serial.
- The `extract` CLI command exposes this as `--max-workers`; benchmark
  execution remains serial by default for reproducible latency comparisons.
- run_inference and __call__ are compatibility aliases.
- extract_bundle(image, ...) returns an `ObservationBundle` that binds the
  result to a `FrameRef` and accepts explicit model provenance and uncertainty.
- `InputLimits(max_pixels=..., max_width=..., max_height=..., max_channels=...,
  max_bytes=...)` can be passed to `Yase(..., input_limits=...)` or used
  directly with `load_image(..., limits=...)`. Violations raise `InputError`
  after decoding and before backend execution.
- The `extract`, `video`, and `benchmark` CLI commands expose the same
  controls through `--max-pixels`, `--max-width`, `--max-height`,
  `--max-channels`, and `--max-bytes`.

## SemanticResult

Typed fields include optional depth, segmentation, detections, tags,
embeddings, OCR regions, captions, scene information, temporal events,
timestamp, keypoints, poses, relations, document payloads, metric depth, and
arbitrary metadata. `BoundingBox`, `OrientedBoundingBox`, `Detection`,
`Keypoint`, `Pose`, `DepthMap`, `Relation`, `TextRegion`, and `SemanticEvent`
are model-neutral typed primitives.

`RuntimeInfo`, `HealthReport`, `collect_runtime_info()`, and `health_check()`
provide lazy readiness diagnostics. They inspect optional packages with
`find_spec` and never download weights or import heavy runtimes just to answer
whether a service can start.

The CLI exposes the same check through `yase diagnostics`; repeat
`--require PACKAGE` to make optional runtime packages readiness requirements.
Scores are normalized to `[0, 1]` by typed primitives, but their calibration is
model-specific and should be documented in backend metadata.

## ObservationBundle and stage contracts

`ObservationBundle` is the versioned transport contract for durable pipelines.
It contains a `FrameRef` (source, frame ID, timestamp, dimensions, color
order), the legacy `SemanticResult`, zero or more `ModelProvenance` records,
named `Uncertainty` values, and application metadata. `to_dict()` is safe for
JSON by default and represents large arrays by shape and dtype; pass
`include_arrays=True` only for deliberate small-result exports.

`EmbeddingRecord` prevents anonymous vectors from silently mixing incompatible
model spaces. It records the model, revision, modality, dimension, dtype, and
whether unit normalization has been applied.

`observation_to_json()` and `write_observation_jsonl()` provide deterministic
JSON/JSONL exports for archives and message sinks. Large arrays remain compact
shape/dtype descriptors unless `include_arrays=True` is explicitly requested.
Standalone `result_to_dict()` payloads now carry `schema_version="1.0"`; use
`result_from_dict()` to restore payloads containing real array values. Legacy
payloads without a version remain accepted, while shape/dtype summaries are
intentionally rejected during reconstruction.

`StageSpec`, `StageContext`, and the `Stage` protocol provide a migration path
from independent named stages to a validated execution graph. Attach a
`StageSpec` as the fourth argument of `PipelineStage`; current stages without a
spec keep their existing behavior. `validate_stage_specs()` rejects duplicate
stage names and missing ordered dependencies before inference begins.

`ObservationScheduler` is the DAG runtime. It reorders contracted stages
topologically, exposes a bounded LRU result cache, propagates cancellation and
deadline state through `StageContext`, and supports `run()` plus non-blocking
`arun()`. `SchedulerReport` contains produced fields and one
`StageExecution` per stage, including duration, cache hits, skips, failures,
and budget violations. `SemanticPipeline.as_scheduler()`,
`extract_scheduled()`, and `extract_bundle_scheduled()` expose the same runtime
without a second pipeline definition.

`ObservationScheduler.run_many()` preserves batch order and validates aligned
frames or per-item external fields. `SchedulerReport.as_result()` is the
canonical conversion back to `SemanticResult`, including a compact scheduler
telemetry record in metadata.

`SchedulerConfig(max_workers=N)` executes up to `N` dependency-free ready
stages concurrently. Cache mutation, dependency promotion, stage callbacks,
and report ordering remain deterministic; the default `N=1` preserves serial
execution.

`CallbackSink`, `MemorySink`, `JsonlObservationSink`, `FanoutSink`, and
`QueueSink` consume `ObservationBundle` values. `QueueSink(on_full="block")`
provides backpressure; `on_full="drop"` preserves the producer and increments
`dropped` for realtime applications where freshness is more important than
completeness.

`RuntimeMetrics` is a thread-safe, dependency-free collector. Pass
`metrics=RuntimeMetrics()` to `ObservationScheduler` and expose
`prometheus_text()` from the host service; it tracks run totals, cache hits,
per-stage status counts, duration summaries, processed video frames, dropped
frames, and video latency summaries. Pass the same collector as `metrics=` to
`VideoStream` or `RealtimeVideoStream` to record stream statistics.
Pass the same collector as `metrics=` to `Yase` to record direct image
extraction attempts, failures, and latency summaries. Install the optional
`observability` extra and pass `tracer=OpenTelemetryTracer()` to emit a
`yase.extract` span with task/model attributes; the bridge records exceptions
and re-raises them without changing extraction semantics.
Direct `CallableExtractor`, TorchScript, ONNX Runtime, OpenVINO, and TensorRT
adapters also accept `input_limits=InputLimits(...)`.

`yase.native.iou_matrix()` uses the compiled C++17 extension when available
and transparently falls back to Python. Run `make native` to build it in place;
`RuntimeInfo.optional_packages["yase._native"]` reports availability.
The generic wheel intentionally remains ABI-independent; compile the optional
extension in-place from the sdist when native acceleration is required.
`yase.native.nms_indices()` provides deterministic greedy non-maximum
suppression with optional class-aware filtering and the same fallback contract.
`non_maximum_suppression()` applies that operation to typed `Detection` values
while preserving masks, track IDs, and custom attributes.

`BackendRegistry.discover_entry_points()` explicitly loads third-party
factories from the `yase.backends` entry-point group. Use
`default_registry(include_plugins=True)` when plugin discovery is desired;
the default registry remains built-in-only and lazy.
Pass a registry to `Yase(model="plugin-name", registry=registry)` to resolve
custom factories through the same facade.

## Backends

CallableExtractor wraps a Python callable. CompositeExtractor combines named
backends and supports error, first, or last conflict policies while preserving
all SemanticResult fields and timestamps. TorchScriptExtractor loads a local
TorchScript file with the torch extra. OnnxRuntimeExtractor loads a local ONNX
file with the onnx extra, prepares NCHW float32 input, and supports depth,
segmentation, or paired outputs. Inject a session object in tests to avoid the
optional runtime dependency. `OpenVINOExtractor` compiles a local IR/ONNX
artifact for CPU/GPU/NPU/AUTO. `TensorRTExtractor` runs a local CUDA plan or a
custom `runner` exposing `infer(array)` for Polygraphy/CuPy/Triton integration.
`CompositeExtractor.extract_batch()` invokes each batch-capable child once and
merges outputs by input position; non-batch children retain the compatible
per-image fallback.
`OnnxRuntimeExtractor(use_io_binding=True)` uses the runtime I/O binding
contract when available and binds outputs on CPU for portable post-processing;
the CLI exposes this as `--io-binding`.

`OpenVINOExtractor.extract_batch_async()` uses OpenVINO's `AsyncInferQueue` to
overlap per-image submissions while restoring input order from callback
userdata. Inject `async_queue=` in tests or managed runtimes; otherwise the
adapter lazily constructs the queue from the installed OpenVINO package.

`TensorRTContextPool(factory, size=N)` owns independent runners/contexts for
concurrent inference. Pass it as `runner_pool=` and call
`TensorRTExtractor.extract_batch_parallel()` when latency overlap is preferred
over the engine's vectorized batch path; call `close()` during worker shutdown.

`normalise_detections()` converts common detector mappings, columnar outputs,
numeric ``(x1, y1, x2, y2, score, class_id)`` rows, and normalized coordinates
into typed `Detection` objects. `inspect_artifact()`, `sha256_file()`, and
`verify_artifact()` provide reproducible local model-artifact metadata without
downloading or caching weights.

## VideoStream and RealtimeVideoStream

VideoStream(source, extractor, stride=1, max_fps=None, max_frames=None,
batch_size=1) yields FrameResult sequentially. When the extractor implements
`extract_batch`, selected frames are grouped without changing output order;
tracking, memory, identity, events, and sinks still run causally one frame at a
time. RealtimeVideoStream uses a worker and one-item latest-frame buffer;
drop_frames=True overwrites stale frames, while False applies backpressure.
Both expose stats with read/processed/dropped counts, source/output FPS,
elapsed time, and mean/max inference latency.
Frames rejected by `error_policy="skip"` or a recovery callback returning
`None` are included in `frames_dropped`, including in the realtime worker.

Use error_policy=skip to discard failed frames, or pass
on_error(exception, frame_index) to return a recovery SemanticResult.
Exceptions are raised by default.

Both stream classes accept an optional `sink=` and `source_id=`. After tracking,
memory, identity, and event enrichment, each processed frame is emitted as an
`ObservationBundle`; the iterator still yields the backward-compatible
`FrameResult`.
Pass `input_limits=InputLimits(...)` to either stream class to enforce the
same pixel, dimension, channel, and byte limits before a callable backend runs.

Optional `tracker` and `memory` stages run before `event_engine`. The tracker
adds stable IDs, while `SemanticTrackMemory` enriches detections with lifetime,
label-history, and exponentially smoothed confidence metadata.
`identity_store` and `camera_id` add optional cross-camera `global_id` matching
from appearance embeddings stored in detection attributes.
`GlobalIdentityStore.state_dict()` and `load_state_dict()` make those
cross-camera identities resumable across worker restarts.
`SemanticTrackMemory` exposes the same checkpoint contract for confidence EMA,
label history, lifetime counters, and missed-frame state.
`VideoStream.save_checkpoint()` and `load_checkpoint()` wrap the same atomic
checkpoint envelope for the state components attached to a stream and include
source/camera context in its metadata. `RealtimeVideoStream` inherits these
methods.

See examples/image.py, examples/batch.py, examples/composite.py,
examples/onnx.py, and examples/video.py for runnable adapters.

## SemanticPipeline and NumpyVectorIndex

`SemanticPipeline` names and composes independent stages such as a detector,
OCR engine, embedding model, and VLM. `NumpyVectorIndex` provides local cosine
search for small or offline embedding collections; `add_record()` and
`search_record()` bind provenance-aware `EmbeddingRecord` values to an
embedding space and persist that namespace. Production deployments can
implement the same boundary with a vector database adapter.
`SemanticPipeline.extract_many()` accepts the same `error_policy="skip"`
approach as `Yase.extract_many()` and invokes `on_error(exception, index, stage)`
for malformed inputs or recoverable stage failures while preserving alignment.
`QdrantVectorIndex` exposes the same record methods; it stores the space,
model ID, and optional revision in each point payload and rejects cross-space
queries before they reach the collection.
Its `search()` and `search_record()` methods also support `min_score`; when a
space is bound, all searches automatically add the corresponding payload
filter.
For multi-modal collections, pass `vector_name="image"` to use a Qdrant named
vector; `payload_indexes={"camera_id": "keyword"}` creates filter indexes at
startup, and `ensure_payload_index()` can add one later.

`make_stream_checkpoint()`, `save_stream_checkpoint()`, and
`load_stream_checkpoint()` provide one versioned JSON checkpoint for the
tracker, `SemanticTrackMemory`, and `GlobalIdentityStore`. The file writer is
atomic and fsyncs its temporary file before replacement, making it suitable
for worker failover and scheduled recovery jobs.

`Tracker` is the common online tracking protocol. `IoUTracker` adds stable
stream-local `track_id` values to typed detections, and both built-in trackers
support JSON-compatible `state_dict()` / `load_state_dict()` checkpoints for
worker failover and resumable video jobs.
`PresenceRule` and `EventEngine` turn frame-level detections into enter/exit
`SemanticEvent` values. Pass `tracker=` and `event_engine=` to either video
stream class to activate them.

`BackendRegistry` and `default_registry()` provide explicit named factories for
optional model adapters. Applications can register their own backends and
record capabilities and the optional installation extra they require.
The built-in registry includes `image-embedding`, `vlm`, `tesseract`, and
`paddleocr` in addition to the detection, segmentation, and accelerator
backends; all remain lazy until `create()` or `Yase` actually instantiates one.
The CLI uses this same registry for image, batch, video, and benchmark
commands; registered VLM/SAM3 backends accept `--prompt`, and Grounding DINO
accepts `--labels`.

`build_from_config()` and `load_config()` construct a `Yase` facade or
`SemanticPipeline` from JSON/TOML or a mapping. Backend names are resolved only
through `BackendRegistry`; arbitrary import paths are rejected. The schema
supports `options`, `limits`, stage `enabled`, `conflict`, and
`record_timings`.
Video CLI processing additionally exposes `--batch-size`; it is ignored for
the latest-frame realtime worker, whose purpose is latency control.

`YaseASGI` and `create_asgi_app()` provide a dependency-free ASGI boundary for
hosts such as Uvicorn: `/health` and `/ready` return readiness JSON,
`/metrics` exposes the facade's Prometheus text when metrics are attached, and
POST `/extract` accepts `{ "image_base64": "...", "timestamp": ... }` without
ever accepting a server-side path. The body limit defaults to 16 MiB.

Optional adapters include `TesseractExtractor`, `PaddleOCRExtractor`,
`TransformersImageEmbeddingExtractor`, and `TransformersVLMExtractor`. They
load their heavy dependencies only when instantiated and default to local-only
Transformers model loading.

`StructuredQuery` and `parse_structured_output()` provide dependency-free JSON
schema validation for VLM answers. `TransformersVLMExtractor.ask_structured()`
preserves the raw answer as `caption`, returns the parsed object as `scene`,
and records the prompt/schema in metadata.
`ask_batch()`/`extract_batch()` perform one processor/model generation for an
ordered image batch; `ask_structured_batch()` validates each JSON answer while
retaining the same provenance metadata.

`RFDETRExtractor` adapts an externally installed RF-DETR model and normalizes
its detections. `ByteTrackLite` is a dependency-free two-stage tracker for
low-confidence recovery; it is a practical fallback for official ByteTrack or
BoT-SORT integrations.

`PromptableSegmentationExtractor` provides a stable boundary for SAM2, SAM3,
or another promptable segmentation runtime without forcing its model license
or framework into the Yase core.

`TransformersSAM3Extractor` is the concrete Transformers implementation for
SAM3 text prompts. It returns instance masks, boxes, scores, and provenance.
`TransformersSAM3VideoExtractor.extract_video()` adds preloaded video
propagation and preserves SAM3 object IDs as `Detection.track_id`.
`ModelCard`, `ModelCatalog`, and `default_model_catalog()` expose model
capabilities and license notes without downloading weights.

`TransformersObjectDetectionExtractor` handles DETR-style models, while
`TransformersGroundingDinoExtractor` handles open-vocabulary models through a
list of text concepts. Both normalize model-specific tensors into
`Detection` values.

`RFDETRExtractor` adapts an externally installed RF-DETR model and normalizes
its detections. `ByteTrackLite` is a dependency-free two-stage tracker for
low-confidence recovery; it is a practical fallback for official ByteTrack or
BoT-SORT integrations.

`AdaptiveSemanticCascade`, `SemanticTrackMemory`, and
`MultimodalConsensus` provide model-independent temporal routing, track-level
memory, and auditable multi-model fusion. `BenchmarkRunner` and
`BenchmarkReport` provide latency percentiles, throughput, failures, and
optional quality metrics for comparing backends on identical inputs.

`evaluate_detections()` provides a deterministic IoU-based precision, recall,
F1, and mean-IoU evaluator suitable for small regression datasets. It is not a
replacement for COCO mAP or MOTChallenge HOTA, but makes local quality gates
possible without adding a dataset framework dependency.

`evaluate_average_precision()` provides a dependency-free, COCO-inspired
101-point interpolated AP at one IoU. `evaluate_mean_average_precision()` adds
the 0.50:0.05:0.95 sweep and returns AP/AR per threshold. The implementation
does not pretend to implement COCO area ranges, ignore regions, or maxDets;
use `pycocotools` for a complete official COCO evaluation.

`evaluate_tracking()` adds approximate MOTA, IDF1, false positives, false
negatives, and identity-switch counts for frame-aligned sequences. `ZoneRule`
and `LineCrossingRule` provide geometry-based events for tracked detections.
`load_mot_sequence()` and `write_mot_sequence()` handle the common
MOTChallenge CSV representation. The CLI command `yase evaluate-mot` reports
single-threshold HOTA, the standard multi-alpha HOTA curve, and tracking
metrics.

`load_coco_dataset()` reads COCO instance JSON into `CocoDataset`, preserving
image order and category IDs. `CocoDataset.frames()` returns aligned typed
detections suitable for AP/mAP evaluation. Polygon masks and uncompressed RLE
work without `pycocotools`; compressed RLE uses that optional package.
`write_coco_predictions()` exports bbox results and optional uncompressed RLE
masks.

`load_coco_predictions()` reads result JSON and aligns predictions to the same
image order. The CLI equivalent is:

~~~bash
yase evaluate-coco predictions.json instances.json --masks
~~~

Image extraction and benchmarking accept files, directories, and glob patterns;
`discover_images()` performs deterministic recursive expansion and filters to
common image extensions.

The `yase artifact PATH --sha256` and `--verify DIGEST` commands expose the
same local artifact integrity helpers for deployment pipelines.

`AdaptiveSemanticCascade`, `SemanticTrackMemory`, and
`MultimodalConsensus` provide model-independent temporal routing, track-level
memory, and auditable multi-model fusion. `BenchmarkRunner` and
`BenchmarkReport` provide latency percentiles, throughput, failures, and
optional quality metrics for comparing backends on identical inputs.
