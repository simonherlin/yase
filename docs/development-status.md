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

## Final gap analysis

The first release boundary is complete. Official ByteTrack, BoT-SORT, and full
Paddle runtime integration remain optional follow-on adapters because they
require separate repositories, hardware, model weights, or licensing decisions.
TensorRT and OpenVINO now have explicit runtime adapters, while the public
contracts remain open for further model integrations.

## Exit criteria

The package is considered ready for a first public release when the core and
all optional adapters have clear dependency boundaries, every public feature
has a deterministic test or an explicit hardware-gated integration test, the
CLI and Python APIs cover image/batch/video/live workflows, and the benchmark
report records latency and quality separately.
