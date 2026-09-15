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
# Optional RF-DETR real-time detector (Python >= 3.10):
uv sync --extra rfdetr

# Optional C++17 tracking acceleration (requires g++ or clang++):
make native
~~~

The package can also be installed from a built wheel with
uv pip install dist/yase-*.whl. Python 3.9 and newer are supported.

The CLI reports available local adapters with `yase info`. Extraction from a
local ONNX model is available with `yase extract image.jpg --model onnx
--model-path model.onnx`.
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
Use `yase models` to inspect supported model families and license notes without
loading or downloading any weights.

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

For Intel CPU/GPU/NPU targets, use the lazy OpenVINO adapter with a local
IR/ONNX artifact. For NVIDIA CUDA plans, `TensorRTExtractor` supports the
native execution-context path or a custom runner managed by the application:

~~~python
from yase import OpenVINOExtractor, TensorRTContextPool, TensorRTExtractor

depth = OpenVINOExtractor("model.xml", device="AUTO").extract("photo.jpg")
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

app = create_asgi_app(Yase(model="onnx", model_path="model.onnx"))
~~~

Deployments can also construct the same facade from a checked-in JSON/TOML
configuration using `load_config("yase.toml")`; only names registered in
`BackendRegistry` are allowed.

Serialized `SemanticResult` payloads include a `schema_version` and can be
restored with `result_from_dict()` when arrays were exported explicitly.
The ASGI service also exposes `/extract/batch` for ordered base64 image lots.
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
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build
~~~

The test suite uses fake backends and captures, so it never downloads model
weights. See docs/devguide for release and quality guidance.

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

For deployment probes, `yase diagnostics` reports Python, NumPy, CPU and
optional runtime availability without loading model weights.

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
