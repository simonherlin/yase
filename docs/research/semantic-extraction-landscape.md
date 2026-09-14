# Semantic extraction landscape

## Executive recommendation

Yase should be a small orchestration layer for computer vision inference, not
another monolithic model zoo. The durable product is a single typed contract
for images, batches, videos, and live streams, with optional adapters for
models, runtimes, storage, and observability.

The recommended architecture is:

1. A dependency-light core that normalizes inputs and emits typed semantic
   observations.
2. Independent stages for detection, segmentation, tracking, OCR, VLM
   reasoning, embeddings, depth, and temporal events.
3. Runtime adapters for CPU, CUDA, TensorRT, OpenVINO, and edge targets.
4. A local NumPy retrieval baseline plus production vector-store adapters.
5. Evaluation and provenance as first-class data, because confidence, model
   version, latency, and license are as important as a label.

This design is the best compromise between state-of-the-art capability and a
package that remains installable, testable, and deployable on hardware that
does not have PyTorch or a GPU.

## What the current repository already provides

The initial Yase implementation already had several good decisions:

- an injectable extractor protocol;
- path, Pillow, NumPy, and OpenCV-frame normalization;
- ordered batch extraction with native batch inference when available;
- TorchScript and ONNX Runtime adapters without implicit weight downloads;
- sequential and latest-frame video iterators with drop/backpressure policies;
- a generic `SemanticResult` able to carry depth, masks, detections, tags, and
  embeddings.

Its main limitation was not the inference code itself but the semantic contract:
objects, OCR, tracks, events, provenance, and retrieval had no stable shape.
The first implementation step therefore adds `BoundingBox`, `Detection`,
`TextRegion`, `SemanticEvent`, `SemanticPipeline`, and `NumpyVectorIndex` while
preserving the existing API.

## Capability survey

### Detection and open-vocabulary grounding

Closed-set real-time detectors remain the right default when the class list is
known and throughput matters. RT-DETR is an end-to-end real-time detector and
its paper reports a detector designed specifically for the real-time regime
([CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf)).
RF-DETR is a newer real-time DETR family with detection and instance
segmentation variants; its project documents a DINOv2 backbone and Apache 2.0
for the core models, with separate licensing for some larger variants
([RF-DETR repository](https://github.com/roboflow/rf-detr)).

Ultralytics remains convenient and widely deployed, but it must not be made a
hard dependency for a permissively licensed Yase distribution. Its official
licensing page states that AGPL-3.0 applies by default and that commercial or
closed deployments generally require an Enterprise license
([Ultralytics licensing](https://www.ultralytics.com/license)).

For arbitrary user concepts, Grounding DINO provides language-conditioned
open-set detection and a natural bridge to promptable segmentation
([official repository](https://github.com/IDEA-Research/GroundingDINO),
[paper](https://arxiv.org/abs/2303.05499)). It should be an optional, slower
"search/annotation" stage rather than the default per-frame detector.

### Segmentation and tracking

SAM 2 established a promptable image-and-video segmentation foundation model
with temporal memory ([paper](https://arxiv.org/abs/2408.00714),
[repository](https://github.com/facebookresearch/sam2)). SAM 3 extends the
concept to text and exemplar prompts and unifies detection, segmentation, and
tracking for all matching instances in images and videos
([Meta research page](https://ai.meta.com/research/sam3/),
[repository](https://github.com/facebookresearch/sam3)).

SAM 3 is a capability to support, not a license to assume. Its repository
contains a dedicated SAM License rather than an ordinary Apache/MIT license
([SAM 3 license](https://github.com/facebookresearch/sam3/blob/main/LICENSE)).
Yase should expose the adapter only when the user explicitly opts into the
model and accepts its terms.

For a detector-first production pipeline, ByteTrack is an efficient and
simple baseline that associates low- and high-confidence detections
([official repository](https://github.com/FoundationVision/ByteTrack)).
BoT-SORT adds appearance cues and camera-motion compensation and reports
strong MOTChallenge results ([repository](https://github.com/NirAharon/BoT-SORT),
[paper](https://arxiv.org/abs/2206.14651)). The tracker interface should accept
and return the neutral `Detection` shape with `track_id`, so either family can
be used without changing application code.

### Embeddings and semantic search

CLIP makes image/text similarity and zero-shot labeling possible using a shared
embedding space ([OpenAI overview](https://openai.com/index/clip/),
[official code](https://github.com/openai/CLIP)). DINOv2 provides robust visual
features that can be reused for classification, retrieval, and downstream
pixel-level tasks without fine-tuning ([model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md),
[paper](https://arxiv.org/abs/2304.07193)).

These representations serve different purposes: CLIP is the better choice for
text-to-image search and zero-shot concepts; DINOv2 is a strong choice for
visual similarity, clustering, and domain adaptation. Yase should not collapse
them into one unnamed `embedding`; provenance must record model name, version,
dimension, normalization, and modality.

Qdrant supports named vectors, metadata filtering, local mode, and multimodal
search ([Python client](https://github.com/qdrant/qdrant-client),
[multimodal guide](https://qdrant.tech/documentation/tutorials-basics/multimodal-search/)).
The package now ships a NumPy baseline for small/offline collections and should
add a Qdrant adapter as an optional integration rather than forcing a server on
every user.

### OCR and documents

PaddleOCR is the stronger direction for multilingual and document-oriented
pipelines; its documentation describes multilingual recognition and document
structure components ([multilingual documentation](https://www.paddleocr.ai/v2.10.0/en/ppocr/blog/multi_languages.html),
[PaddleOCR 3.0 report](https://arxiv.org/abs/2507.05595)). Tesseract remains a
valuable lightweight Apache-2.0 fallback with broad language support and
coordinate-aware hOCR output ([official manual](https://tesseract-ocr.github.io/tessdoc/)).

The stable abstraction should be a sequence of text regions, not only a single
string. That is why Yase exposes `TextRegion(text, score, box, polygon,
language)` and leaves table/layout/document graphs to an adapter-specific
metadata payload.

### VLMs and temporal understanding

Vision-language models are useful for captions, question answering, document
interpretation, attribute extraction, and long-tail concepts, but they are too
expensive and nondeterministic to run on every live frame. Transformers exposes
a common image-text-to-text API and supports current Qwen-VL families
([official task guide](https://huggingface.co/docs/transformers/main/tasks/image_text_to_text),
[Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/v5.3.0/en/model_doc/qwen2_5_vl)).

The correct pattern is a cascade: cheap per-frame perception, temporal
aggregation, then VLM calls only on keyframes, events, or user queries. The VLM
output should be attached as a caption/scene/metadata result with model and
prompt provenance, never silently treated as ground truth.

For action and clip-level understanding, VideoMAE is an established video
transformer exposed by Transformers, while PyTorchVideo provides reusable
SlowFast, X3D, and related video components
([VideoMAE docs](https://huggingface.co/docs/transformers/model_doc/videomae),
[PyTorchVideo model guide](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/models.md)).
These belong behind a clip sampler and temporal stage rather than inside the
image extractor interface.

## Runtime and deployment strategy

There is no universally best inference runtime. The adapter should choose based
on hardware and artifact format:

| Target | Recommended first runtime | Reason |
|---|---|---|
| Python CPU | ONNX Runtime or OpenVINO | Portable execution providers and easy packaging |
| NVIDIA GPU | TensorRT, optionally through ONNX Runtime | FP16/INT8 and optimized GPU engines |
| CUDA development | PyTorch/Transformers | Fastest iteration and broadest model support |
| Intel CPU/GPU/NPU | OpenVINO | CPU/GPU/NPU and automatic/heterogeneous device modes |
| Mobile/embedded PyTorch path | ExecuTorch | Export/compile/runtime path for constrained devices |
| Small offline Python collection | NumPy index | No service and no vector database dependency |
| Production vector retrieval | Qdrant/FAISS adapter | Filtering, indexing, persistence, and scale |

ONNX Runtime exposes CPU, CUDA, TensorRT, OpenVINO, DirectML, XNNPACK, QNN,
and other execution providers ([official provider matrix](https://onnxruntime.ai/docs/execution-providers/)).
TensorRT supports ONNX parsing, engine building, dynamic shapes, and mixed
precision ([NVIDIA Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)).
OpenVINO explicitly supports CPU, GPU, and NPU plus automatic and heterogeneous
inference ([supported devices](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)).
ExecuTorch provides a separate export/compile/runtime workflow for mobile,
embedded, and constrained hardware ([PyTorch documentation](https://docs.pytorch.org/executorch/stable/)).

The package must never hide runtime selection or download weights implicitly.
Every adapter should report `model_id`, `model_revision`, `runtime`, `device`,
precision, input shape, and latency in metadata.

## Proposed public contract

The semantic contract is intentionally richer than a tensor but simpler than a
model-specific object graph:

```text
SemanticResult
├── depth: ndarray | None
├── segmentation: ndarray | None
├── detections: list[Detection] | None
│   ├── label, score, BoundingBox
│   ├── optional mask and track_id
│   └── arbitrary attributes
├── ocr: list[TextRegion] | None
├── tags: list[...] | None
├── caption / scene: optional high-level semantics
├── embeddings: ndarray | None + provenance in metadata
├── events: list[SemanticEvent] | None
└── timestamp + metadata/provenance
```

`SemanticPipeline` composes independent stages. It should later grow the
following optional concepts without breaking the core:

- `StreamContext` for tracker and temporal-model state;
- keyframe policies and asynchronous VLM execution;
- backpressure-aware bounded queues between decode, inference, and sinks;
- a model registry with explicit local/Hugging Face artifact resolution;
- adapters for Qdrant, FAISS, and dataset export;
- OpenTelemetry spans and metrics around decode, preprocessing, inference,
  postprocessing, queue depth, dropped frames, and model errors.

OpenTelemetry's Python API covers traces and metrics and is suitable for this
observability layer ([official Python docs](https://opentelemetry.io/docs/languages/python/)).

## What should be implemented next

### Phase 1 — foundation (implemented in this repository)

- Keep the minimal NumPy/Pillow core.
- Add typed boxes, detections, OCR regions, and temporal events.
- Add a named `SemanticPipeline` over the existing composite backend.
- Add a local cosine index for embeddings.
- Preserve explicit model loading and deterministic tests.

### Phase 2 — high-value adapters

1. `TransformersVLMExtractor` for local Qwen-VL/Florence-like models.
2. `GroundingDinoExtractor` with a neutral detection output.
3. `Sam2VideoSegmenter` and an explicit SAM 3 adapter when its license is
   acceptable to the deployment.
4. `ByteTrackTracker` and `BoTSORTTracker` with stream state.
5. `PaddleOCRExtractor` and `TesseractExtractor`.
6. CLIP and DINOv2 embedding adapters.

### Phase 3 — production

- Qdrant/FAISS adapters and incremental indexing.
- Keyframe/event cascades for VLM calls.
- ONNX Runtime provider selection, TensorRT engine cache, OpenVINO device
  selection, and ExecuTorch export helpers.
- Prometheus/OpenTelemetry instrumentation.
- Reproducible benchmark suite on representative CPU, CUDA, and edge devices.

## Evaluation plan

"Best" must be measured as a Pareto frontier, not a single leaderboard number.
For each adapter and target device record:

- quality: detection mAP/recall, mask IoU, OCR CER/WER, retrieval Recall@K,
  tracking HOTA/IDF1, event F1;
- performance: p50/p95 latency, throughput, startup time, memory, and dropped
  frames;
- robustness: resolution changes, low light, blur, camera motion, occlusion,
  long-tail labels, multilingual text, and corrupted input;
- operational behavior: deterministic output, error recovery, cache behavior,
  model provenance, and license compatibility.

The benchmark runner should export JSON/Parquet records so model and runtime
comparisons remain reproducible. Results from one GPU should never be presented
as universal speed claims.

## Risks and non-goals

- No single foundation model is optimal for every domain or latency budget.
- Open-vocabulary models can hallucinate labels or produce unstable scores;
  thresholds and calibration belong to the application.
- VLM text is an interpretation, not a pixel-accurate annotation.
- Tracking IDs are stream-local and must not be treated as permanent identity.
- Model weights and datasets have separate licenses from package code; Yase
  must keep adapters explicit and ship license metadata.
- The core package should not depend on CUDA, a vector database, a web API, or
  a particular model vendor.

## Sources

The links embedded above are the primary papers, official repositories, and
official runtime documentation consulted for this design. They should be
rechecked before each release because model versions, availability, and license
terms change quickly.

1. Meta AI, [SAM 3 research page](https://ai.meta.com/research/sam3/).
2. Meta FAIR, [SAM 3 repository](https://github.com/facebookresearch/sam3) and
   [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).
3. Meta FAIR, [SAM 2 paper](https://arxiv.org/abs/2408.00714) and
   [SAM 2 repository](https://github.com/facebookresearch/sam2).
4. Roboflow, [RF-DETR repository](https://github.com/roboflow/rf-detr) and
   [RF-DETR paper](https://arxiv.org/abs/2511.09554).
5. Ultralytics, [official licensing page](https://www.ultralytics.com/license).
6. Lyuwenyu et al., [RT-DETR CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf).
7. IDEA Research, [Grounding DINO repository](https://github.com/IDEA-Research/GroundingDINO)
   and [paper](https://arxiv.org/abs/2303.05499).
8. FoundationVision, [ByteTrack repository](https://github.com/FoundationVision/ByteTrack).
9. Aharon et al., [BoT-SORT paper](https://arxiv.org/abs/2206.14651) and
   [repository](https://github.com/NirAharon/BoT-SORT).
10. OpenAI, [CLIP overview](https://openai.com/index/clip/) and
    [official repository](https://github.com/openai/CLIP).
11. Meta FAIR, [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
    and [paper](https://arxiv.org/abs/2304.07193).
12. PaddlePaddle, [multilingual PaddleOCR documentation](https://www.paddleocr.ai/v2.10.0/en/ppocr/blog/multi_languages.html)
    and [PaddleOCR 3.0 report](https://arxiv.org/abs/2507.05595).
13. Tesseract OCR, [official user manual](https://tesseract-ocr.github.io/tessdoc/).
14. Hugging Face, [image-text-to-text task guide](https://huggingface.co/docs/transformers/main/tasks/image_text_to_text)
    and [Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/v5.3.0/en/model_doc/qwen2_5_vl).
15. Hugging Face, [VideoMAE documentation](https://huggingface.co/docs/transformers/model_doc/videomae).
16. Meta FAIR, [PyTorchVideo model guide](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/models.md).
17. ONNX Runtime, [execution provider documentation](https://onnxruntime.ai/docs/execution-providers/).
18. NVIDIA, [TensorRT Python API documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).
19. Intel, [OpenVINO supported devices](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html).
20. PyTorch, [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/).
21. Qdrant, [Python client](https://github.com/qdrant/qdrant-client) and
    [multimodal search guide](https://qdrant.tech/documentation/tutorials-basics/multimodal-search/).
22. OpenTelemetry, [Python documentation](https://opentelemetry.io/docs/languages/python/).
