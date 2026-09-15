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
# Optional local Transformers VLM/embedding adapters:
uv sync --extra transformers
# Optional Tesseract OCR adapter:
uv sync --extra ocr
# Optional OpenVINO runtime:
uv sync --extra openvino
# CPU-oriented complete stack (does not install CUDA/TensorRT):
uv sync --extra all
# GPU-oriented stack (requires compatible CUDA/TensorRT components):
uv sync --extra all-gpu
# Alternative ONNX Runtime provider; choose this instead of `onnx`:
uv sync --extra onnx-gpu
# NVIDIA TensorRT plan execution:
uv sync --extra tensorrt
# Optional RF-DETR real-time detector (Python >= 3.10, Transformers 5.x):
uv sync --extra rfdetr

# Optional C++17 tracking acceleration in the source checkout:
make native
~~~

The package can also be installed from a built wheel with
`uv pip install dist/yase-*.whl`. Python 3.10–3.13 are supported; Python 3.12
is the recommended development and deployment baseline. `make build` produces
the portable `py3-none-any` wheel using the current `uv` environment;
`make build-isolated` performs the strict PEP 517 isolated build. The
`make build-native` target produces an
OS/Python-ABI-specific wheel containing the optional C++ extension; it requires
the matching Python development headers and a C++17 compiler. For a complete
multi-platform native wheel matrix, run cibuildwheel from a dedicated release
environment with the checked-in configuration.

The native extension is intentionally small: model inference already runs in
the selected ONNX Runtime, OpenVINO, TensorRT, or PyTorch engine. Yase keeps
Python for orchestration, schemas, I/O, plugins, and error policies, and uses
C++ only for reusable CPU hot paths (IoU and greedy NMS). This avoids a second
inference engine, keeps the portable wheel universal, and guarantees the same
deterministic Python fallback when no compiler or native wheel is available.
The provider results for the reference workstation are recorded in
[`docs/research/runtime-matrix-0.64.md`](docs/research/runtime-matrix-0.64.md).
The complete 0.66.0 release gates are recorded in
[`docs/research/release-readiness-0.66.md`](docs/research/release-readiness-0.66.md).

The CLI reports available local adapters with `yase info`. Extraction from a
local ONNX model is available with `yase extract image.jpg --model onnx
--model-path model.onnx`.
Use `yase diagnostics --providers` when deploying to inspect actual ONNX
providers, OpenVINO devices, and optional CUDA/TensorRT availability. Run
`uv run python tools/runtime_smoke.py --all-openvino-devices` to compile and
execute the deterministic smoke graph on every discovered OpenVINO device;
for ONNX GPU validation, add `--onnx-provider CUDAExecutionProvider` so a CPU
fallback fails loudly.
The extraction CLI exposes the same control with repeated
`--provider CUDAExecutionProvider --strict-providers` flags.
Use `--device GPU` for OpenVINO or `--device cuda` for TorchScript/TensorRT
when selecting their runtime device from the CLI.
The same commands also accept registered semantic backends such as `vlm`,
`image-embedding`, `sam3`, `rf-detr`, `tesseract`, and `paddleocr`; use
`--model-path` for a local/Hugging Face model identifier, `--prompt` for VLM
or SAM3, and `--labels` for Grounding DINO. Optional dependencies remain
lazy.
For ONNX accelerator sessions, add `--io-binding` to reduce host/device copy
overhead when the installed execution provider supports it.
The same interface supports `yase video input.mp4 --model onnx
--model-path model.onnx --output results.jsonl` and `--realtime` for a live
source. File processing can use `--batch-size 8` when the selected backend
implements `extract_batch`.
For repeatable local measurements, `yase benchmark image-*.jpg --model onnx
--model-path model.onnx --warmup 2` reports p50/p95/p99 latency and throughput.
For phase attribution, construct an ONNX/OpenVINO/TensorRT/TorchScript backend
with `record_timings=True`; each result then contains JSON-safe
`metadata["timings_seconds"]` for `preprocess`, `inference`, and `postprocess`,
and `BenchmarkReport.to_dict()` exports their p50/p95/p99 summaries.
Use `yase models` to inspect supported model families and license notes without
loading or downloading any weights.

To measure the optional local C++ kernels independently from model inference,
run `uv run python benchmarks/native.py --boxes 256 --iterations 10`. The
benchmark reports IoU and NMS latency for both implementations when the native
extension is installed; it is intentionally a diagnostic, not a claim about
end-to-end model speed.

Detector outputs from external runtimes can be made framework-neutral with
`normalise_detections()`, and local model files can be checked with
`inspect_artifact()`/`verify_artifact()` before deployment.

COCO instance annotations are available through `load_coco_dataset()` and can
be exported with `write_coco_predictions()` without installing `pycocotools`
for polygon or uncompressed-RLE workflows. This follows the standard COCO
`bbox: [x, y, width, height]` and polygon conventions described by the
[COCO API](https://github.com/cocodataset/cocoapi).

Evaluate a COCO result file directly:

~~~bash
yase evaluate-coco predictions.json instances.json --masks
~~~

The `extract` and `benchmark` commands also accept image directories and glob
patterns, for example `yase extract 'frames/**/*.jpg' --model onnx ...`.

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

When a deployment must fail rather than silently fall back to CPU, configure
the requested execution provider explicitly and enable `strict_providers`:

~~~python
backend = OnnxRuntimeExtractor(
    "model.onnx",
    providers=[("CUDAExecutionProvider", {"device_id": "0"})],
    strict_providers=True,
)
~~~

The adapter records the active provider list in every result metadata payload;
strict mode checks that each requested provider is actually active in the
created ONNX Runtime session.

For Intel CPU/GPU/NPU targets, use the lazy OpenVINO adapter with a local
IR/ONNX artifact. For NVIDIA CUDA plans, `TensorRTExtractor` supports the
native execution-context path or a custom runner managed by the application:

~~~python
from yase import OpenVINOExtractor, TensorRTContextPool, TensorRTExtractor

depth = OpenVINOExtractor(
    "model.xml",
    device="AUTO",
    compile_config={"PERFORMANCE_HINT": "LATENCY"},
    cache_dir=".yase-openvino-cache",
).extract("photo.jpg")
depth = TensorRTExtractor("model.plan").extract("photo.jpg")

# Independent contexts for concurrent latency-sensitive inference:
pool = TensorRTContextPool(lambda: make_runner(), size=2)
results = TensorRTExtractor(runner_pool=pool).extract_batch_parallel(images)

# OpenVINO asynchronous queue (one request per image, ordered results):
images = ["one.jpg", "two.jpg"]
results = OpenVINOExtractor("model.xml", async_jobs=4).extract_batch_async(images)
~~~

Install accelerator runtimes separately; Yase never downloads weights or
imports these frameworks unless their adapters are instantiated.

All four local model adapters accept the opt-in `record_timings=True` flag.
Instrumentation is disabled by default and reports wall-clock adapter phases;
provider-specific device transfer or asynchronous kernel timings should be
validated separately on the target hardware.

For long-lived services, pass a shared bounded `RuntimeCache(capacity=8)` to
TorchScript, ONNX Runtime, or OpenVINO adapters to reuse model/session
resources. The cache never downloads artifacts, exposes `info()`, and closes
owned resources on eviction or `clear()`; call `cache.close()` during service
shutdown. TensorRT contexts should generally be supplied through an explicit
`TensorRTContextPool` when concurrent inference is required.

The repository pins Python 3.12 in `.python-version` because it is the most
conservative intersection of the supported vision runtimes.

Tracing is optional and lazy:

~~~python
from yase import OpenTelemetryTracer, Yase

model = Yase(extractor=my_backend, tracer=OpenTelemetryTracer())
~~~

Qdrant deployments can keep multiple embedding spaces in one collection with
`QdrantVectorIndex(..., vector_name="image")`; pass `payload_indexes=` for
fields used frequently in filtered search.

For a minimal HTTP deployment, install an ASGI server separately and expose
the built-in service:

~~~python
from yase import Yase, create_asgi_app

api = Yase(model="onnx", model_path="model.onnx")
app = create_asgi_app(api, max_concurrency=4, timeout_seconds=10.0)
~~~

Deployments can also construct the same facade from a checked-in JSON/TOML
configuration using `load_config("yase.toml")`; only names registered in
`BackendRegistry` are allowed.
Use `with Yase(...) as api:` or call `api.close()` when a backend owns runtime
contexts that must be released at shutdown.

Serialized `SemanticResult` payloads include a `schema_version` and can be
restored with `result_from_dict()` when arrays were exported explicitly.
`migrate_result_payload()` handles legacy `mask`, `text`, and `poses` aliases;
stream checkpoints expose the analogous `migrate_stream_checkpoint()` helper
and can carry a model/configuration fingerprint to prevent unsafe resumption.
The ASGI service also exposes `/extract/batch` for ordered base64 image lots.
Synchronous model calls run in a bounded worker pool; call `app.close()` when
the host shuts down.
Scheduler-based pipelines accept the same optional tracer to correlate each
named stage with the parent extraction span.

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
FPS, bounded output count, and optional `batch_size` acceleration for backends
implementing `extract_batch`; post-processing remains in frame order.

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
Pass tracker=ByteTrackLite() and memory=SemanticTrackMemory() to keep
identities and semantic confidence stable across frames.
Use `stream.save_checkpoint("state.json")` and
`stream.load_checkpoint("state.json")` to resume attached tracker, memory, and
cross-camera identity state after a worker restart.

For analytics across several cameras, attach a `GlobalIdentityStore` and set
`camera_id`; detections carrying an appearance vector in
`attributes["embedding"]` receive a stable `global_id`.

Video events can be configured without a framework dependency:

~~~python
from yase import EventEngine, LineCrossingRule, ZoneRule

events = EventEngine([
    ZoneRule("person", [(0, 0), (500, 0), (500, 400), (0, 400)], zone_name="secure"),
    LineCrossingRule("person", (250, 0), (250, 400)),
])
~~~

Use `evaluate_detections()` and `evaluate_tracking()` for deterministic local
quality gates before running a full COCO or MOTChallenge evaluation.
Use `evaluate_average_precision()` or `evaluate_mean_average_precision()` for
framework-free score-ranked AP/AR regression checks across IoU thresholds.
For MOTChallenge-style CSV files, the same evaluation is available from the
CLI:

~~~bash
yase evaluate-mot predictions.txt groundtruth.txt --iou 0.5
~~~

The MOT JSON output includes the requested single-threshold decomposition and
the standard 19-threshold HOTA curve.

Inspect or verify a local model artifact before deployment:

~~~bash
yase artifact model.onnx --sha256
yase artifact model.onnx --verify <sha256-digest>
~~~

## Development

~~~bash
uv sync --dev
make check                 # lock, lint, format, tests and coverage
make runtime-smoke         # optional ONNX/OpenVINO runtime probes
make build                 # portable sdist + py3-none-any wheel
make build-native          # ABI-specific wheel with C++ acceleration
~~~

GitHub Actions are intentionally disabled to avoid consuming hosted-runner
credits. The test suite uses fake backends and captures, so it never downloads
model weights; `make runtime-smoke` only probes runtimes already installed on
the local machine. See [`docs/README.md`](docs/README.md) for the architecture,
API, release audits, and local quality gates.

## Composable semantic pipelines

The stable result schema also supports model-agnostic detections, OCR regions,
captions, scenes, temporal events, and embeddings. A pipeline can combine
independent model adapters without making any of them mandatory:

~~~python
from yase import Detection, PipelineStage, SemanticPipeline, TextRegion

pipeline = SemanticPipeline([
    PipelineStage("detector", detector_backend),
    PipelineStage("ocr", ocr_backend),
])
result = pipeline.extract("photo.jpg")
~~~

For durable storage or video analytics, use the versioned observation contract:

~~~python
from yase import ModelProvenance, Uncertainty, Yase

bundle = Yase(extractor=detector_backend).extract_bundle(
    frame,
    frame_id=42,
    source_id="camera-1",
    provenance=(ModelProvenance(model_id="detector", revision="v1"),),
    uncertainty={"detections": Uncertainty(confidence=0.91)},
)
payload = bundle.to_dict()  # JSON-friendly, arrays summarized by default
~~~

`FrameRef`, `ObservationBundle`, `ModelProvenance`, and `Uncertainty` make
provenance and confidence first-class instead of backend-specific conventions.

Contracted stages can be executed as a dependency graph:

~~~python
from yase import ObservationScheduler, PipelineStage, SchedulerConfig, StageSpec

detector_spec = StageSpec("detector", provides=("detections",))
caption_spec = StageSpec(
    "caption", requires=("detections",), provides=("caption",)
)
scheduler = ObservationScheduler(
    [
        PipelineStage("caption", caption_backend, spec=caption_spec),
        PipelineStage("detector", detector_backend, spec=detector_spec),
    ],
    config=SchedulerConfig(cache_size=128, max_total_latency_ms=100),
)
report = scheduler.run(frame)
~~~

The scheduler reorders the graph, caches repeatable stage outputs, propagates
cancellation and deadlines, and exposes stage-level telemetry.

For deployment probes, `yase diagnostics` reports Python, NumPy, CPU, optional
runtime availability, and installed distribution versions without loading
model weights. Add `--providers` when the target machine should also be probed
for ONNX Runtime, OpenVINO, PyTorch CUDA, and TensorRT devices.
ASGI uploads are header-checked and verified before inference; malformed images
and Pillow decompression bombs are rejected as client input errors.

For small collections, `NumpyVectorIndex` provides local cosine retrieval and
can ingest `SemanticResult.embeddings`. For larger collections, keep this API
as the application boundary and replace it with a Qdrant/FAISS adapter.

`docs/research/semantic-extraction-landscape.md` records the current model,
runtime, licensing, and architecture study behind these choices.

## Adaptive semantic intelligence

Yase includes model-agnostic orchestration primitives. `AdaptiveSemanticCascade`
uses a fast extractor on stable frames and invokes an accurate extractor when
confidence or visual novelty requires it. `SemanticTrackMemory` accumulates
stable labels and confidence for tracker-provided IDs. `MultimodalConsensus`
fuses overlapping detections from several models and records abstentions and
evidence sources instead of hiding disagreement.

~~~python
from dataclasses import replace

from yase import AdaptiveSemanticCascade, ByteTrackLite, SemanticTrackMemory

cascade = AdaptiveSemanticCascade(fast_detector, accurate_detector)
tracker = ByteTrackLite()
memory = SemanticTrackMemory()
result = cascade.extract(frame)
result = memory.update(replace(result, detections=tracker.update(result.detections)))
~~~

`RFDETRExtractor` is an optional adapter for the externally installed RF-DETR
package. Heavy models remain lazy and are never downloaded by Yase implicitly.
Review the selected model's license before redistributing weights or a product
built around it.

`TransformersSAM3Extractor` implements SAM3 text-prompted image segmentation
through the official Transformers API and returns both instance masks and
normalized `Detection` values. Install the `transformers` extra and explicitly
manage model weights according to the model card.
`TransformersSAM3VideoExtractor` provides offline video propagation with
preserved object IDs; use `VideoStream` plus a causal tracker for strict
real-time processing.

## Project status

The injectable core, rich result schema, composable pipeline, local vector
index, TorchScript/ONNX adapters, and video iterators are stable. Modern
detection, OCR, segmentation, VLM, and tracking models remain optional
integrations: configure them explicitly for the target hardware, model
license, and latency budget. Yase never downloads weights implicitly.
