# Yase autonomous development status

This file is the execution ledger for the project. Each cycle ends with tests,
lint, packaging, and a short gap analysis before the next action list is
selected.

## Cycle 1 — core contract

- [x] Audit the initial repository and existing tests.
- [x] Preserve the lightweight NumPy/Pillow installation.
- [x] Add typed boxes, detections, OCR regions, and temporal events.
- [x] Add composable pipelines and an explicit backend registry.
- [x] Add a local vector-search baseline.
- [x] Write the technology and licensing study.

Result: version 0.3.0, 37 tests passing.

## Cycle 2 — usable application surface

- [x] Add JSON/JSONL serialization with large-array-safe defaults.
- [x] Add dependency-free IoU tracking with stable IDs and expiry.
- [x] Integrate optional tracking into sequential and real-time video streams.
- [x] Add Tesseract and PaddleOCR normalization adapters.
- [x] Add local Transformers image-embedding and VLM adapters.
- [x] Add a CLI for local model extraction and backend inspection.
- [x] Add vector-index filtering and NPZ persistence.

Result: adapters and application surface implemented; 43 deterministic tests
pass.

## Cycle 3 — current action list

- [x] Add stream context and event-rule API for temporal analytics.
- [x] Add batch-capable `SemanticPipeline` execution and per-stage timings.
- [x] Add Qdrant adapter behind the retrieval extra.
- [x] Add model provenance and structured error objects to every stage.
- [x] Add CLI video processing and JSONL output.
- [x] Add deterministic integration tests for pipeline, tracking, OCR, CLI, and
  persisted retrieval.
- [x] Rebuild wheels and verify a clean installation in a fresh environment.

## Cycle 4 — final hardening action list

- [x] Add structured extraction errors with stage/backend context.
- [x] Validate all public imports and optional extras in a clean wheel install.
- [x] Add model provenance defaults and document confidence semantics.
- [x] Run packaging, CLI, coverage, and smoke tests from the built artifact.
- [x] Perform final gap analysis and freeze the first release boundary.

## Cycle 5 — adaptive semantic engine

- [x] Add budget-aware fast/accurate semantic cascade with novelty routing.
- [x] Add track-level semantic memory with confidence smoothing and expiry.
- [x] Add multimodal consensus with auditable evidence and abstention.
- [x] Add dependency-free ByteTrack-inspired two-stage tracker.
- [x] Add RF-DETR adapter boundary and registry metadata.
- [x] Add benchmark reports with p50/p95/p99, throughput, failures, and quality hooks.
- [x] Integrate memory into sequential and real-time video streams.
- [x] Expose benchmark workflow through the CLI.

Result: version 0.4.0, 54 deterministic tests passing before final validation.

## Cycle 6 — ecosystem and evaluation hardening

- [x] Add a concrete Transformers SAM3 image segmentation adapter.
- [x] Add model cards, provenance, capabilities, and license catalog.
- [x] Add deterministic detection precision/recall/F1/IoU metrics.
- [x] Add SAM3, Grounding DINO, and RF-DETR entries to the backend registry.
- [x] Add `yase models` to inspect available model cards without downloading weights.
- [x] Add preloaded SAM3 video propagation with preserved object IDs.

Result: version 0.5.0, all new features remain optional and lazy-loaded.

## Cycle 7 — multi-camera analytics

- [x] Add polygon zone enter/exit events.
- [x] Add line-crossing events with direction metadata.
- [x] Add NumPy-only cross-camera global identity matching.
- [x] Add external tracker normalization boundary.
- [x] Add approximate MOTA/IDF1 and identity-switch metrics.
- [x] Integrate identity matching into sequential and realtime video streams.

Result: version 0.6.0, 64 deterministic tests passing before final validation.

## Cycle 8 — benchmark science and MOT interoperability

- [x] Add maximum-assignment single-threshold HOTA evaluation.
- [x] Add instance-mask IoU and mask quality metrics.
- [x] Add MOTChallenge sequence reader and writer.
- [x] Add duration-of-presence/dwell events.
- [x] Add `yase evaluate-mot` for reproducible CLI evaluation.

Result: version 0.7.0, 69 deterministic tests passing before final validation.

## Cycle 9 — accelerator runtimes and HOTA curve

- [x] Add standard multi-alpha HOTA curve evaluation and JSON serialization.
- [x] Add OpenVINO CPU/GPU/NPU/AUTO adapter with lazy compilation.
- [x] Add TensorRT CUDA plan adapter with native execution-context support and
  a custom runner boundary for Polygraphy/CuPy/Triton integrations.
- [x] Register accelerator runtimes without importing their dependencies.
- [x] Add CI coverage for Python 3.9 through 3.13.
- [x] Add deterministic fake-runtime tests so accelerator contracts are tested
  on CPU-only development machines.

Result: version 0.8.0, accelerator support remains optional and local-artifact
only; no model weights are downloaded by Yase.

## Cycle 10 — portable output contracts and artifact integrity

- [x] Add framework-neutral detector-output normalization for mappings,
  columnar tensors, numeric rows, class maps, thresholds, and normalized boxes.
- [x] Add ONNX Runtime provider options, graph optimization/profiling controls,
  NCHW/NHWC input layouts, and provider discovery.
- [x] Add SHA-256 artifact inspection and verification helpers for reproducible
  local model deployments.
- [x] Expose artifact inspection and verification through the CLI.
- [x] Add deterministic tests for provider configuration, normalization edge
  cases, and artifact integrity.

Result: version 0.9.0, 77 deterministic tests passing before final validation.

## Cycle 11 — detection and instance-mask AP

- [x] Add deterministic 101-point interpolated average precision at one IoU.
- [x] Add mean AP/AR over the standard 0.50:0.05:0.95 sweep.
- [x] Add instance-mask AP using the same score-ranked protocol.
- [x] Preserve explicit scope: no COCO area ranges, ignore regions, or maxDets
  are silently approximated.
- [x] Add perfect, false-positive, mask, and frame-alignment regression tests.

Result: version 0.10.0, 80 deterministic tests passing before final validation.

## Cycle 12 — COCO interoperability

- [x] Add COCO image/category/annotation records with ordered detection frames.
- [x] Load COCO bbox annotations into typed `Detection` values.
- [x] Rasterize polygon segmentation with Pillow/NumPy and decode uncompressed
  RLE without requiring `pycocotools`.
- [x] Delegate compressed RLE to optional `pycocotools` with an explicit error
  when that dependency is absent.
- [x] Export aligned bbox/mask predictions as COCO result JSON.
- [x] Add end-to-end COCO-to-mask-AP regression tests.

Result: version 0.11.0, 81 deterministic tests passing before final validation.

## Cycle 13 — COCO evaluation and batch discovery

- [x] Load COCO result JSON into frames aligned with ground truth image order.
- [x] Add multi-threshold instance-mask mAP/mAR.
- [x] Add `yase evaluate-coco` for bbox evaluation and optional mask evaluation.
- [x] Add deterministic file/directory/glob image discovery for extract and
  benchmark workflows.
- [x] Add end-to-end CLI, loader, export, and batch-discovery tests.

Result: version 0.12.0, 82 deterministic tests passing before final validation.

## Cycle 14 — versioned observations and stage contracts

- [x] Add `FrameRef` with source, timestamp, dimensions, and color-order
  invariants.
- [x] Add `ModelProvenance` with artifact SHA-256 and runtime/device metadata.
- [x] Add named `Uncertainty` and validated `EmbeddingRecord` contracts.
- [x] Add versioned `ObservationBundle` serialization without breaking
  `SemanticResult` consumers.
- [x] Add `Yase.extract_bundle()` for image and frame workflows.
- [x] Add `StageSpec`, `StageContext`, and a structural `Stage` protocol.
- [x] Validate ordered stage dependencies while keeping legacy pipelines
  compatible.
- [x] Add negative and positive tests for all new public invariants.
- [x] Update API and architecture documentation.

Result: version 0.13.0, 87 deterministic tests passing before final validation;
versioned observations are now the migration boundary for the next scheduler/
DAG cycle.

## Cycle 15 — scheduler and dependency-aware execution

- [x] Add topological DAG planning for contracted `PipelineStage` values.
- [x] Add bounded LRU stage-result caching and cache introspection.
- [x] Add total and per-stage latency budgets with explicit skip/raise policy.
- [x] Add cancellation propagation and deadline state in `StageContext`.
- [x] Add structured `StageExecution` telemetry and callbacks.
- [x] Add synchronous and async scheduler entry points.
- [x] Integrate scheduler execution with `SemanticPipeline` and
  `ObservationBundle`.
- [x] Add graph, cache, async, failure, cancellation, and compatibility tests.

Result: version 0.14.0, 92 deterministic tests passing before final validation.

## Cycle 16 — composable observation sinks

- [x] Add callback, memory, JSONL, fan-out, and bounded queue sinks.
- [x] Make backpressure versus dropping an explicit queue policy.
- [x] Keep sinks dependency-free and compatible with `ObservationBundle`.
- [x] Add lifecycle, close, flush, fan-out, and queue overflow tests.

Result: version 0.14.0, 93 deterministic tests passing before final validation.

## Cycle 17 — video-to-observation integration

- [x] Emit processed sequential and realtime frames as `ObservationBundle`
  values through the sink protocol.
- [x] Preserve `FrameResult` compatibility and post-process sink emissions
  after tracking, memory, identity, and event stages.
- [x] Add source/frame identity to video observations.
- [x] Add a stride-one regression test for frame-index continuity.

Result: version 0.14.0, 94 deterministic tests passing before final validation;
video, scheduler, and sink paths now share the same versioned observation
boundary.

## Cycle 18 — generic scheduler batch projection

- [x] Add ordered `ObservationScheduler.run_many()` with frame and external
  field alignment checks.
- [x] Add `SchedulerReport.as_result()` as the canonical DAG-to-result
  projection.
- [x] Reuse that projection from `SemanticPipeline` sync, async, and bundle
  APIs.
- [x] Add batch/cache/projection regression tests.

Result: version 0.15.0, 94 deterministic tests passing before final validation.

## Cycle 19 — typed rich-vision outputs

- [x] Add validated `OrientedBoundingBox`, `Keypoint`, and `Pose` contracts.
- [x] Add `DepthMap` with unit, scale, invalid-value, and camera metadata.
- [x] Add scored `Relation` for scene-graph assertions and evidence.
- [x] Add first-class `depth_map`, `keypoints`, `relations`, and `document`
  fields to `SemanticResult`.
- [x] Preserve the new fields through normalization, composition, fusion,
  memory, cascade, benchmark, scheduler, and serialization paths.
- [x] Add regression tests for validation and propagation.

Result: version 0.16.0, 95 deterministic tests passing before final validation.

## Cycle 20 — runtime diagnostics and readiness

- [x] Add lazy `RuntimeInfo` capability inspection.
- [x] Add machine-readable `HealthReport` with ready/degraded status.
- [x] Add extractor-interface and required-package readiness checks.
- [x] Keep diagnostics free of weight downloads and heavy imports.
- [x] Add tests for healthy and degraded environments.

Result: version 0.17.0, 96 deterministic tests passing before final validation.

## Cycle 21 — diagnostics CLI and readiness surface

- [x] Add `yase diagnostics` with repeatable required-package checks.
- [x] Return JSON suitable for container probes and deployment dashboards.
- [x] Test CLI readiness output and preserve non-zero degraded semantics.

Result: version 0.17.0, 97 deterministic tests passing before final validation.

## Cycle 22 — bounded image inputs

- [x] Add configurable pixel, dimension, channel, and byte limits.
- [x] Apply limits consistently to path, Pillow, NumPy, batch, and bundle inputs.
- [x] Expose `InputLimits` and raise the stable `InputError` contract.
- [x] Add regression tests for rejection and valid extraction.

Result: version 0.18.0, 98 deterministic tests passing before final validation.

## Cycle 23 — bounded DAG parallelism

- [x] Add `SchedulerConfig.max_workers` with strict positive-integer validation.
- [x] Execute independent ready stages concurrently with a bounded thread pool.
- [x] Preserve deterministic dependency promotion, cache writes, callbacks, and
  report ordering.
- [x] Cover fan-out execution with a synchronization-barrier regression test.

Result: version 0.19.0, 99 deterministic tests passing before final validation.

## Cycle 24 — dependency-free runtime observability

- [x] Add thread-safe `RuntimeMetrics` with bounded in-process state.
- [x] Integrate scheduler reports without making Prometheus mandatory.
- [x] Export counters and duration summaries in Prometheus text format.
- [x] Add snapshot and exposition regression tests.

Result: version 0.20.0, 100 deterministic tests passing before final validation.

## Cycle 25 — release artifact validation

- [x] Validate sdist and wheel creation locally for version `0.20.0`.
- [x] Verify critical public modules are included in the wheel.
- [x] Add a CI release job with artifact and diagnostics smoke checks.

Result: release checks pass locally; CI now gates publication-shaped artifacts
on the full test matrix.

## Cycle 26 — CLI input safety controls

- [x] Expose all `InputLimits` dimensions on image, video, and benchmark CLI
  commands.
- [x] Reject non-positive values before model initialization.
- [x] Propagate limits through the existing `Yase` API boundary.
- [x] Add parser regression coverage and update the API reference.

Result: version 0.21.0, 101 deterministic tests passing before final validation.

## Cycle 27 — bounded direct video streams

- [x] Apply `InputLimits` to sequential and realtime video streams.
- [x] Protect callable/custom backends that do not instantiate `Yase`.
- [x] Preserve frame error policies and add a skip-path regression test.

Result: version 0.22.0, 102 deterministic tests passing before final validation.

## Cycle 28 — video runtime observability

- [x] Record processed and dropped frames from sequential and realtime streams.
- [x] Export aggregate video latency summaries through `RuntimeMetrics`.
- [x] Keep metrics dependency-free and bounded by process lifetime.
- [x] Add stream integration and Prometheus exposition tests.

Result: version 0.23.0, 103 deterministic tests passing before final validation.

## Cycle 29 — bounded generic batch parallelism

- [x] Add opt-in `Yase.extract_many(max_workers=N)` for non-batch backends.
- [x] Preserve input order, timestamps, input limits, and error callbacks.
- [x] Keep native `extract_batch` backends on their optimized path.
- [x] Add a barrier-based concurrency regression test.

Result: version 0.24.0, 104 deterministic tests passing before final validation.

## Cycle 30 — batch parallelism from CLI

- [x] Expose `--max-workers` on the `extract` command.
- [x] Preserve serial benchmark behavior for comparable latency reports.
- [x] Reuse the same positive-integer validation as all CLI limits.
- [x] Add parser coverage for the new execution control.

Result: version 0.25.0, 104 deterministic tests passing before final validation.

## Cycle 31 — explicit backend plugin discovery

- [x] Support Python entry points in the `yase.backends` group.
- [x] Preserve lazy, built-in-only defaults unless discovery is requested.
- [x] Record plugin provenance in `BackendSpec.metadata`.
- [x] Support modern and Python 3.9 entry-point APIs with regression tests.

Result: version 0.26.0, 105 deterministic tests passing before final validation.

## Cycle 32 — registry-backed Yase facade

- [x] Allow `Yase` to resolve custom models through an injected registry.
- [x] Preserve native backend shortcuts and lazy initialization.
- [x] Validate registry shape before inference starts.
- [x] Add end-to-end facade coverage for a registered backend.

Result: version 0.27.0, 106 deterministic tests passing before final validation.

## Cycle 33 — direct image observability

- [x] Record successful and failed `Yase.extract` attempts.
- [x] Export direct extraction latency summaries in `RuntimeMetrics`.
- [x] Keep input validation failures visible to service operators.
- [x] Add nominal and failure-path regression coverage.

Result: version 0.28.0, 107 deterministic tests passing before final validation.

## Cycle 34 — backend-level input safety

- [x] Apply `InputLimits` to the five core direct backend families.
- [x] Preserve resizing and batching behavior after validation.
- [x] Keep optional runtime dependencies lazy and unchanged.
- [x] Add direct callable regression coverage.

Result: version 0.29.0, 108 deterministic tests passing before final validation.

## Cycle 35 — optional C++ acceleration

- [x] Add a C++17 extension for pairwise IoU matrices.
- [x] Integrate the native matrix into `IoUTracker` without changing contracts.
- [x] Provide a portable Python fallback and explicit in-place build script.
- [x] Add native compilation CI and benchmark the measured hot-path speedup.
- [x] Include compiled `.so`, `.pyd`, and `.dylib` artifacts in native wheels.

Result: version 0.30.0, 109 tests passing with native code enabled locally.

## Cycle 36 — native ByteTrack matching

- [x] Reuse the native IoU matrix for ByteTrackLite high/low-score matching.
- [x] Preserve score ordering, class-aware filtering, and track IDs.
- [x] Add a regression test for low-confidence track recovery.

Result: version 0.31.0, 110 tests passing with native code enabled locally.

## Cycle 37 — native greedy NMS

- [x] Add deterministic C++17 greedy NMS with optional class-aware filtering.
- [x] Keep a validated Python fallback for environments without a compiler.
- [x] Expose the utility as a package-level API for detector integrations.

Result: version 0.32.0, pending final validation.

## Cycle 38 — typed detector post-processing

- [x] Expose class-aware `non_maximum_suppression()` for `Detection` values.
- [x] Preserve masks, track IDs, attributes, and deterministic score ordering.
- [x] Reuse the native implementation without coupling normalisation to a model framework.

Result: version 0.33.0, pending final validation.

## Cycle 39 — Python-only NMS fallback

- [x] Remove per-comparison temporary IoU matrices from the fallback NMS path.
- [x] Reuse one scalar IoU implementation for matrix and suppression operations.
- [x] Keep the no-compiler installation path deterministic and dependency-light.

Result: version 0.34.0, pending final validation.

## Cycle 40 — ABI-safe native packaging

- [x] Prevent Python-ABI-specific binaries from entering `py3-none-any` wheels.
- [x] Keep native source and the in-place builder available in sdists.
- [x] Add CI assertions for portable wheel contents and source rebuildability.

Result: version 0.35.0, pending final validation.

## Cycle 41 — portable NMS parity fix

- [x] Treat omitted class IDs as one shared suppression class in the fallback.
- [x] Add an installed-wheel regression for Python-only NMS behavior.
- [x] Release the correction as version 0.35.1.

Result: version 0.35.1, pending final validation.

## Cycle 42 — native input validation

- [x] Reject non-finite raw box coordinates before native or fallback execution.
- [x] Keep direct low-level APIs aligned with typed `BoundingBox` invariants.
- [x] Add a regression test for `NaN` input.

Result: version 0.35.2, pending final validation.

## Cycle 43 — native box geometry validation

- [x] Reject reversed raw box coordinates before dispatching to C++.
- [x] Match the invariants enforced by the public `BoundingBox` schema.
- [x] Add a regression test for malformed geometry.

Result: version 0.35.3, pending final validation.

## Cycle 44 — resumable tracker state

- [x] Define a shared `Tracker` protocol for online update/reset/checkpoint APIs.
- [x] Add versioned JSON-compatible state checkpoints to IoUTracker and ByteTrackLite.
- [x] Validate IDs, boxes, labels, counters, velocity, and state versions on restore.

Result: version 0.36.0, pending final validation.

## Cycle 45 — validated structured VLM extraction

- [x] Add a dependency-free `StructuredQuery` and JSON-schema subset validator.
- [x] Add fenced/prefix JSON recovery for common VLM formatting behavior.
- [x] Add `TransformersVLMExtractor.ask_structured()` with raw-answer retention.
- [x] Allow injected torch modules for deterministic adapter tests.

Result: version 0.37.0, pending final validation.

## Cycle 46 — complete built-in backend registry

- [x] Register image embeddings and VLMs with capability metadata.
- [x] Register Tesseract and PaddleOCR behind their explicit extras.
- [x] Keep all newly registered frameworks lazy-loaded.

Result: version 0.38.0, pending final validation.

## Cycle 47 — RF-DETR distribution coherence

- [x] Define the `rfdetr` optional extra referenced by the backend registry.
- [x] Restrict the extra to Python versions supported by upstream RF-DETR.
- [x] Test that every registry extra is declared by the installed package metadata.

Result: version 0.38.1, pending final validation.

## Cycle 48 — resumable cross-camera identity

- [x] Add versioned JSON-compatible checkpoints to `GlobalIdentityStore`.
- [x] Validate identity IDs, embeddings, timestamps, camera history, and counts.
- [x] Add a restart simulation preserving cross-camera matching.

Result: version 0.39.0, pending final validation.

## Cycle 49 — embedding-space safety

- [x] Add provenance-aware add/search methods to `NumpyVectorIndex`.
- [x] Reject mixed embedding spaces while retaining the raw-vector API.
- [x] Persist and restore the index namespace in NPZ checkpoints.

Result: version 0.40.0, pending final validation.

## Cycle 50 — resumable semantic memory

- [x] Add versioned checkpoints to `SemanticTrackMemory`.
- [x] Preserve EMA confidence, label history, lifetime, missed frames, and frame counter.
- [x] Add restart/recovery regression coverage.

Result: version 0.41.0, pending final validation.

## Cycle 51 — batched video extraction

- [x] Add an opt-in `VideoStream(batch_size=...)` path for batch-capable backends.
- [x] Preserve frame order and causal tracker/memory/identity/event processing.
- [x] Keep per-frame skip/recovery behavior when a batch call fails.
- [x] Enforce `max_frames` without over-processing a final partial batch.
- [x] Apply input limits before batching and retain legacy callable/mapping outputs.

Result: version 0.42.0, pending final validation.

## Cycle 52 — provenance-safe Qdrant retrieval

- [x] Add `EmbeddingRecord`-aware `add_record()` and `search_record()` methods.
- [x] Bind a Qdrant adapter to one embedding space and reject mismatches.
- [x] Persist `space`, `model_id`, and optional revision in point payloads.
- [x] Reject empty IDs, non-finite vectors, and zero-norm queries.
- [x] Add deterministic injected-client coverage without requiring Qdrant.

Result: version 0.43.0, pending final validation.

## Cycle 53 — atomic stream checkpoints

- [x] Add a versioned checkpoint envelope for all online semantic state.
- [x] Restore tracker, temporal memory, and cross-camera identity together.
- [x] Write checkpoints atomically beside the destination and fsync before replace.
- [x] Validate component contracts, unknown components, metadata, and versions.
- [x] Add restart simulation coverage with real built-in stateful components.

Result: version 0.44.0, pending final validation.

## Cycle 54 — retrieval API parity

- [x] Add Qdrant `min_score`/score-threshold support.
- [x] Apply the bound embedding-space payload filter to every Qdrant search.
- [x] Reject conflicting caller-provided space filters and non-finite thresholds.
- [x] Extend injected-client coverage while keeping Qdrant lazy and optional.

Result: version 0.45.0, pending final validation.

## Cycle 55 — registry-complete CLI

- [x] Expose all built-in lazy backends through `extract`, `video`, and `benchmark`.
- [x] Add model identifier/path, VLM/SAM3 prompt, and Grounding DINO label options.
- [x] Preserve local runtime behavior and fail clearly when required model options are absent.
- [x] Add parser coverage for modern registered backends without importing heavy runtimes.

Result: version 0.46.0, pending final validation.

## Cycle 56 — batched video CLI

- [x] Expose `VideoStream(batch_size=...)` as `yase video --batch-size`.
- [x] Keep realtime latest-frame semantics explicit and separate.
- [x] Validate the option as a positive integer and document backend behavior.

Result: version 0.47.0, pending final validation.

## Cycle 57 — native batched VLM generation

- [x] Add one-call VLM caption/question answering for ordered image batches.
- [x] Add one-call structured JSON extraction with per-answer validation.
- [x] Reuse the path from `VideoStream(batch_size=...)` automatically.
- [x] Preserve raw captions, prompts, schemas, and backend provenance.
- [x] Add deterministic injected processor/model coverage.

Result: version 0.48.0, pending final validation.

## Cycle 58 — stream checkpoint integration

- [x] Add `VideoStream.save_checkpoint()` and `load_checkpoint()` convenience APIs.
- [x] Include source and camera provenance in checkpoint metadata.
- [x] Make the same API available to `RealtimeVideoStream` through inheritance.
- [x] Add integration coverage with tracker and semantic memory recovery.

Result: version 0.49.0, pending final validation.

## Cycle 59 — batched composite extraction

- [x] Add `CompositeExtractor.extract_batch()` with ordered row-wise fusion.
- [x] Invoke native child batches once and retain per-image fallback children.
- [x] Preserve conflict policies, timestamps, metadata, and contextual errors.
- [x] Add regression coverage for mixed batch/non-batch composites.

Result: version 0.50.0, pending final validation.

## Cycle 60 — ONNX accelerator I/O binding

- [x] Add opt-in `OnnxRuntimeExtractor(use_io_binding=True)` execution.
- [x] Bind CPU inputs and named outputs through `run_with_iobinding()`.
- [x] Fail clearly when an injected/runtime session lacks the binding contract.
- [x] Expose the feature as `--io-binding` in all model-processing CLI commands.
- [x] Add deterministic session and parser coverage.

Result: version 0.51.0, pending final validation.

## Cycle 61 — realtime loss accounting

- [x] Count realtime backend failures as dropped frames under skip policy.
- [x] Count input-limit/preprocessing failures in the same metric.
- [x] Cover deterministic backpressure-mode error accounting.

Result: version 0.52.0, pending final validation.

## Cycle 62 — full production audit

- [x] Audit every public subsystem, optional runtime, persistence boundary, and release gate.
- [x] Cross-check ONNX Runtime, OpenVINO, TensorRT, Transformers, and Qdrant official APIs.
- [x] Separate delivered contracts, hardware-gated validation, and remaining P0/P1/P2 work.
- [x] Publish the prioritized package-wide roadmap in `research/full-package-audit.md`.
- [x] Repair the documentation index so every linked guide exists.

Result: package-wide roadmap refreshed for the `0.52.0` release line.

## Cycle 63 — resilient native batch extraction

- [x] Load batch inputs independently so `error_policy="skip"` can preserve
  valid images when another item is malformed or exceeds input limits.
- [x] Preserve original indices and timestamps when only valid batch members
  are sent to a native backend.
- [x] Route pre-processing failures through the public `on_error` callback.
- [x] Record one extraction metric per batch item, including load failures.
- [x] Retry valid items individually when a provider-level batch call fails and
  the request is skip-tolerant.
- [x] Add regression tests for partial batches, callback indices, and metrics.

Result: version `0.53.0`, pending final validation.

## Cycle 64 — OpenVINO asynchronous inference queue

- [x] Add an injectable `AsyncInferQueue` boundary without importing OpenVINO
  on base-package import.
- [x] Submit one request per image and reconstruct results using callback
  userdata, preserving public input order independent of completion order.
- [x] Support OpenVINO request `results`, `outputs`, and
  `get_output_tensor()` result surfaces.
- [x] Expose `extract_batch_async()` and `extract_async()` with explicit queue
  validation and metadata.
- [x] Add deterministic fake-queue tests that run without accelerator hardware.

Result: version `0.54.0`, pending final validation.

## Cycle 65 — optional OpenTelemetry tracing

- [x] Add a lazy `OpenTelemetryTracer` bridge that keeps the base install
  dependency-free.
- [x] Support injection of an application's configured OpenTelemetry tracer for
  deterministic tests and custom providers.
- [x] Instrument `Yase.extract()` with task/model attributes and exception
  recording while preserving existing metrics and error semantics.
- [x] Export the bridge from the top-level package and document its usage.

Result: version `0.55.0`, pending final validation.

## Cycle 66 — scalable Qdrant retrieval

- [x] Add optional Qdrant named-vector collection configuration through
  `vector_name=` while preserving legacy unnamed vectors.
- [x] Query named vectors through both modern `query_points(using=...)` and
  legacy `search(query_vector=(name, vector))` client surfaces.
- [x] Add explicit `payload_indexes=` bootstrap and public
  `ensure_payload_index()` for high-cardinality filtered fields.
- [x] Preserve embedding-space provenance checks and normalized cosine vectors
  across both collection layouts.
- [x] Add fake-client regression coverage for collection, point, query, and
  payload-index contracts.

Result: version `0.56.0`, pending final validation.

## Cycle 67 — dependency-free ASGI service surface

- [x] Add `YaseASGI` and `create_asgi_app()` without requiring FastAPI or an
  ASGI server in the base install.
- [x] Expose `/health`, `/ready`, `/metrics`, and a POST `/extract` endpoint.
- [x] Accept only base64-encoded image bytes, never server filesystem paths.
- [x] Enforce request body limits and return structured 400/413/404/500 JSON
  errors while preserving client disconnects.
- [x] Add end-to-end fake-ASGI tests for health, extraction, invalid input, and
  request-size rejection.

Result: version `0.57.0`, pending final validation.

## Cycle 68 — resilient semantic pipeline batches

- [x] Add `error_policy="raise|skip"` and a stage-aware `on_error` callback to
  `SemanticPipeline.extract_many()`.
- [x] Preprocess batch members independently so malformed inputs do not cancel
  valid stage outputs.
- [x] Preserve aligned result positions and retry valid members individually
  after a provider-level batch failure in skip mode.
- [x] Keep dependency-ordered stages and timing metadata compatible with the
  existing scheduler path.
- [x] Add regression coverage for invalid inputs and callback context.

Result: version `0.58.0`, validated with 139 tests, 82.52% coverage, wheel /
sdist builds, and an isolated Python 3.9 wheel smoke test. See the [release
readiness report](research/release-readiness-0.58.md).

## Final gap analysis

The first release boundary is complete. Official ByteTrack, BoT-SORT, and full
Paddle runtime integration remain optional follow-on adapters because they
require separate repositories, hardware, model weights, or licensing decisions.
TensorRT and OpenVINO now have explicit runtime adapters, while the public
contracts remain open for further model integrations.

The 0.58.0 release boundary also includes resilient native batches, OpenVINO
queue execution, optional OpenTelemetry, named-vector Qdrant retrieval, a
dependency-free ASGI surface, and package-level artifact validation. Live
provider CI remains the next operational investment because this CPU-only
environment cannot exercise accelerator hardware.

## Cycle 69 — TensorRT context pool

- [x] Add a thread-safe `TensorRTContextPool` around independent callable or
  `.infer()` runners.
- [x] Add `TensorRTExtractor.extract_batch_parallel()` for overlap across
  independent contexts while preserving input order.
- [x] Keep ordinary vectorized `extract_batch()` as the default for engines
  whose optimized batch dimension is faster.
- [x] Add explicit lifecycle shutdown and closed-pool errors.
- [x] Add deterministic pool tests without TensorRT or CUDA installed.

Result: version `0.59.0`, pending final validation.

## Cycle 70 — declarative configuration

- [x] Add safe JSON/TOML loading through `load_config()` and direct mapping
  construction through `build_from_config()`.
- [x] Resolve all backends through `BackendRegistry`; configuration files never
  import arbitrary Python paths.
- [x] Support one `Yase` facade or a named `SemanticPipeline`, including input
  limits, stage enablement, conflict policy, and timing controls.
- [x] Validate unknown fields, backend names, options, limits, and file suffixes
  with actionable errors.
- [x] Add JSON-file and custom-registry regression tests.

Result: version `0.60.0`, pending final validation.

## Cycle 71 — versioned semantic result serialization

- [x] Add `RESULT_SCHEMA_VERSION = "1.0"` to standalone result serialization.
- [x] Add `result_from_dict()` with legacy payload acceptance and explicit
  rejection of unsupported versions.
- [x] Reject shape/dtype-only summaries when a caller requests reconstruction,
  avoiding silent fake tensors.
- [x] Keep `ObservationBundle` serialization compatible while making nested
  result schemas self-describing.
- [x] Add round-trip, legacy, summary-rejection, and future-version tests.

Result: version `0.61.0`, pending final validation.

## Cycle 72 — ASGI batch extraction

- [x] Add POST `/extract/batch` with ordered base64 images and aligned
  timestamps.
- [x] Reuse `Yase.extract_many()` so native backend batching and `skip/raise`
  behavior remain identical between Python and HTTP clients.
- [x] Enforce a configurable `max_batch_size` and close all decoded images on
  success or failure.
- [x] Return JSON-safe per-item results, including aligned `null` values for
  skip-tolerated inference failures.
- [x] Add end-to-end ASGI tests for successful batches and size rejection.

Result: version `0.62.0`, pending final validation.

## Cycle 73 — second full package audit

- [x] Re-audit all public surfaces after the TensorRT pool, declarative config,
  versioned results, and ASGI batch work.
- [x] Reconcile the roadmap with official TensorRT, Transformers, ONNX Runtime,
  OpenVINO, and Qdrant runtime constraints.
- [x] Separate delivered platform capabilities from hardware-gated work still
  requiring live provider CI.
- [x] Update the full audit and release checklist to the 0.62.0 state.

Result: architecture roadmap synchronized; pending final 0.62.0 validation.

## Cycle 74 — 0.62.0 release validation

- [x] Run 142 tests and lint/format checks on Python 3.12.
- [x] Run the coverage gate on Python 3.9: 82.38% total coverage.
- [x] Build and inspect exact 0.62.0 wheel and sdist artifacts.
- [x] Install the wheel in a fresh Python 3.9 environment with only base
  dependencies and run an extraction plus serialization smoke test.
- [x] Record provider and hardware limitations in the release report.

Result: version `0.62.0` release gates passed; provider-specific CI remains the
next operational requirement.

## Cycle 75 — scheduler trace propagation

- [x] Accept the same optional tracer bridge on `ObservationScheduler` and
  `SemanticPipeline.as_scheduler()`.
- [x] Emit one `yase.stage.<name>` span per executed stage, including stages
  dispatched through the parallel worker pool.
- [x] Preserve cache, error, cancellation, timing, and callback semantics.
- [x] Add deterministic span-attribute coverage without OpenTelemetry
  installed.

Result: version `0.63.0`, pending final validation.

## Cycle 76 — 0.63.0 release validation

- [x] Run 143 tests and lint/format checks on Python 3.12.
- [x] Run the coverage gate on Python 3.9: 82.38% total coverage.
- [x] Build and inspect exact 0.63.0 wheel and sdist artifacts.
- [x] Install the wheel in a fresh Python 3.9 environment and run extraction
  plus versioned-result serialization smoke tests.
- [x] Record live-provider limits and remaining operational roadmap.

Result: version `0.63.0` release gates passed; provider-specific CI remains
the next operational requirement.

## Cycle 77 — Python runtime policy

- [x] Audit the complete dependency/runtime intersection instead of selecting
  Python from the base package alone.
- [x] Set Python 3.12 as the repository development/reference interpreter via
  `.python-version` and the CI matrix.
- [x] Support the practical modern range Python 3.10–3.13 and reject Python
  3.9/3.14 for the current release line until the optional ecosystem matrix
  proves them safe.
- [x] Align Ruff and RF-DETR metadata with the new package boundary.
- [x] Record the hardware finding: the local Quadro M3000M is compute 5.2 and
  cannot validate current TensorRT releases that require Turing-class or newer
  GPUs; OpenVINO CPU and ONNX CPU remain valid local targets.

Result: version `0.64.0`; Python 3.12 is the recommended Yase environment,
and the provider smoke matrix is recorded in
`docs/research/runtime-matrix-0.64.md`.

## Cycle 78 — live runtime matrix

- [x] Recreate the project environment with Python 3.12.3 through `uv` without
  changing the system Python installation.
- [x] Install and execute OpenVINO 2026.3.1 on CPU, GPU, and AUTO with both
  synchronous and `AsyncInferQueue` extraction paths.
- [x] Install and execute ONNX Runtime 1.30.0 on CPU with normal execution and
  I/O binding, including a real temporary ONNX graph.
- [x] Instantiate the OpenTelemetry bridge against the installed API.
- [x] Compile and execute the optional C++17 IoU/NMS extension using temporary
  Python development headers, without changing system packages.
- [x] Inspect NVIDIA/CUDA capabilities and explicitly classify TensorRT as
  hardware-gated on the local Quadro M3000M (compute 5.2).
- [x] Publish exact versions, provider lists, commands, and limitations in the
  runtime matrix report.

Result: version `0.64.0` has validated CPU and OpenVINO GPU/AUTO execution
paths; TensorRT remains ready for supported deployment hosts but cannot be
truthfully validated on this Maxwell workstation.

## Cycle 79 — safe image decoding

- [x] Validate Pillow image dimensions from the header before pixel conversion.
- [x] Keep final NumPy byte/channel checks after conversion.
- [x] Add regression coverage for path and in-memory Pillow inputs exceeding
  configured limits.

Result: untrusted image paths now fail before the expensive decode/conversion
step when they exceed configured dimensions, reducing decompression-bomb risk.

## Cycle 80 — provider diagnostics

- [x] Keep default diagnostics dependency-light and non-invasive.
- [x] Add explicit ONNX Runtime provider discovery with versions.
- [x] Add explicit OpenVINO device discovery with device names.
- [x] Add optional PyTorch CUDA and TensorRT probes with structured errors.
- [x] Expose the probe through `yase diagnostics --providers` and the public
  `collect_provider_info()` API.
- [x] Add machine-readable regression coverage without making optional
  runtimes mandatory for the base installation.

Result: deployments can now distinguish an installed package from an actual
provider/device exposed by the host.

## Cycle 81 — reproducible provider smoke tests

- [x] Add a non-published `runtime-test` dependency group for provider CI.
- [x] Add a real ONNX graph smoke script covering batch execution and I/O
  binding.
- [x] Add a real OpenVINO graph smoke script covering CPU batch execution and
  `AsyncInferQueue` ordering.
- [x] Add a dedicated Linux/Python 3.12 runtime-smoke CI job.

Result: optional runtime regressions now have a reproducible CI path without
downloading model weights or making heavy providers part of the base package.

## Cycle 82 — RF-DETR dependency contract

- [x] Recheck the current upstream RF-DETR requirements against the optional
  dependency declarations.
- [x] Require RF-DETR 1.6+, PyTorch 2.2+, torchvision 0.17+, and Transformers
  5.1 through 5.x in the dedicated extra.
- [x] Keep the generic Transformers extra independent for older compatible
  vision/VLM integrations.

Result: RF-DETR resolution now fails early for an unsupported dependency mix
instead of failing later during model construction.

## Cycle 83 — batch tracing

- [x] Add a `yase.extract_batch` span around native backend batch execution.
- [x] Include task, selected model, batch size, and backend class attributes.
- [x] Preserve raw tracer compatibility by applying attributes when the caller
  provides `start_as_current_span` directly.
- [x] Add deterministic regression coverage for ordered batch tracing.

Result: batch inference now has the same minimum observability contract as
single-image extraction and DAG stages.

## Cycle 84 — ASGI decoded-input limits

- [x] Add an optional `InputLimits` contract directly to `YaseASGI`.
- [x] Validate image dimensions and decoded RGB byte estimates before custom
  backends receive a request.
- [x] Preserve body-size and batch-size limits and add an HTTP regression test.

Result: the service boundary remains safe even when it wraps a backend that is
not itself a `Yase` facade.

## Cycle 85 — observation deserialization

- [x] Add `observation_from_dict()` for versioned bundle reconstruction.
- [x] Reconstruct frame references, model provenance, uncertainty, metadata,
  and real serialized result arrays.
- [x] Reject future schema versions and shape/dtype-only tensor summaries.
- [x] Add archive round-trip and invalid-payload coverage.

Result: durable observation JSON can now be consumed by a separate worker or
resume process without bypassing the typed contract.

## Cycle 86 — native build portability

- [x] Detect Python development headers from the active interpreter first.
- [x] Fall back to the matching system `/usr/include/pythonX.Y` headers when
  `uv` manages the interpreter separately.
- [x] Keep an explicit environment override for custom header layouts.
- [x] Validate the resulting ABI-specific extension locally with C++17.

Result: the Linux native CI job now installs and discovers the headers it needs
instead of relying on an implicit system/interpreter configuration match.

## Cycle 87 — 0.65.0 release validation

- [x] Bump the package and runtime metadata to 0.65.0.
- [x] Validate the lockfile, dependency consistency, lint, formatting, and
  the complete 147-test suite.
- [x] Validate the real provider smoke script for OpenVINO and ONNX Runtime.
- [x] Build the 0.65.0 wheel and source distribution.
- [x] Verify the portable wheel in a clean Python 3.12 environment and ensure
  the optional native extension is not embedded in the generic wheel.
- [x] Publish the release readiness report and remaining hardware-gated limits.

Result: 0.65.0 is ready as a portable release candidate; CUDA/TensorRT remains
explicitly hardware-gated by the local Maxwell GPU.

## Cycle 88 — supported Python wheel matrix

- [x] Install the 0.65.0 wheel in clean Python 3.10, 3.11, 3.12, and 3.13
  environments.
- [x] Verify import, public result construction, and version metadata in every
  supported interpreter.
- [x] Keep optional provider smoke tests concentrated on the validated Python
  3.12 reference environment.

Result: the published pure wheel imports across the complete advertised
Python range; heavy provider combinations remain separately matrixed.

## Exit criteria

The package is considered ready for a first public release when the core and
all optional adapters have clear dependency boundaries, every public feature
has a deterministic test or an explicit hardware-gated integration test, the
CLI and Python APIs cover image/batch/video/live workflows, and the benchmark
report records latency and quality separately.
