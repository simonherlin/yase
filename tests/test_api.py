import asyncio
import base64
import json
import sys
from contextlib import contextmanager
from importlib.metadata import metadata
from io import BytesIO, StringIO
from threading import Barrier, Event

import numpy as np
import pytest

from yase import (
    AdaptiveSemanticCascade,
    ArtifactInfo,
    AveragePrecisionResult,
    BackendError,
    BackendRegistry,
    BenchmarkRunner,
    BoundingBox,
    ByteTrackLite,
    CallableExtractor,
    CallbackSink,
    CocoDataset,
    CocoImage,
    CompositeExtractor,
    DepthMap,
    Detection,
    DetectionMetrics,
    DwellRule,
    EmbeddingRecord,
    EventEngine,
    ExternalTrackerAdapter,
    FanoutSink,
    FrameRef,
    GlobalIdentityStore,
    HealthReport,
    HOTACurveResult,
    HOTAResult,
    InputError,
    InputLimits,
    IoUTracker,
    JsonlObservationSink,
    Keypoint,
    LineCrossingRule,
    MaskMetrics,
    MeanAveragePrecisionResult,
    MemorySink,
    ModelProvenance,
    MultimodalConsensus,
    NumpyVectorIndex,
    ObservationBundle,
    ObservationScheduler,
    OnnxRuntimeExtractor,
    OpenTelemetryTracer,
    OpenVINOExtractor,
    OrientedBoundingBox,
    PipelineStage,
    Pose,
    PresenceRule,
    PromptableSegmentationExtractor,
    QdrantVectorIndex,
    QueueSink,
    RealtimeVideoStream,
    Relation,
    RFDETRExtractor,
    RuntimeInfo,
    RuntimeMetrics,
    SchedulerConfig,
    SchedulerError,
    SemanticEvent,
    SemanticPipeline,
    SemanticResult,
    SemanticTrackMemory,
    StageCancelled,
    StageContext,
    StageExecution,
    StageSpec,
    StructuredQuery,
    TemperatureScaler,
    TensorRTContextPool,
    TensorRTExtractor,
    TesseractExtractor,
    TextRegion,
    Tracker,
    TrackingMetrics,
    TransformersSAM3Extractor,
    TransformersSAM3VideoExtractor,
    TransformersVLMExtractor,
    Uncertainty,
    VideoStats,
    Yase,
    YaseASGI,
    ZoneRule,
    build_from_config,
    collect_runtime_info,
    create_asgi_app,
    default_model_catalog,
    default_registry,
    discover_images,
    evaluate_average_precision,
    evaluate_detections,
    evaluate_hota,
    evaluate_hota_curve,
    evaluate_mask_average_precision,
    evaluate_masks,
    evaluate_mean_average_precision,
    evaluate_mean_mask_average_precision,
    evaluate_tracking,
    health_check,
    inspect_artifact,
    load_coco_dataset,
    load_coco_predictions,
    load_config,
    load_image,
    load_mot_sequence,
    load_stream_checkpoint,
    make_stream_checkpoint,
    nms_indices,
    non_maximum_suppression,
    normalise_detections,
    observation_to_json,
    parse_structured_output,
    result_from_dict,
    result_to_dict,
    result_to_json,
    save_stream_checkpoint,
    sha256_file,
    validate_stage_specs,
    verify_artifact,
    write_coco_predictions,
    write_mot_sequence,
    write_observation_jsonl,
)
from yase.video import VideoStream


class FakeBackend:
    def __init__(self):
        self.inputs = []

    def extract(self, image):
        self.inputs.append(image)
        return {"depth": image[..., 0].astype(np.float32)}


class FakeCapture:
    def __init__(self, count=5):
        self.frames = [
            np.full((2, 3, 3), index, dtype=np.uint8) for index in range(count)
        ]
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True

    def isOpened(self):
        return True

    def get(self, _property):
        return 0.0


def test_import_and_fake_depth_backend():
    backend = FakeBackend()
    api = Yase(extractor=backend)
    result = api.extract(np.zeros((2, 3, 3), dtype=np.uint8), timestamp=1.5)
    assert isinstance(result, SemanticResult)
    assert result.depth.shape == (2, 3)
    assert result.timestamp == 1.5
    assert len(backend.inputs) == 1


def test_load_image_converts_grayscale_and_bgr():
    gray = np.array([[1, 2]], dtype=np.uint8)
    assert load_image(gray).shape == (1, 2, 3)
    bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)
    assert load_image(bgr, color_order="BGR").tolist() == [[[3, 2, 1]]]


def test_result_mapping_and_backward_compatibility():
    api = Yase(task="segmentation", extractor=lambda image: {"mask": image[..., 0]})
    result = api.run_inference(np.ones((2, 2, 3), dtype=np.uint8))
    assert result.segmentation.shape == (2, 2)
    assert result.depth is None


def test_invalid_task_is_rejected():
    with pytest.raises(ValueError, match="task"):
        Yase(task="classification", extractor=lambda image: image)


def test_video_stride_and_release():
    capture = FakeCapture(5)
    backend = FakeBackend()
    stream = VideoStream(capture, Yase(extractor=backend), stride=2)
    frames = list(stream)
    assert [item.frame_index for item in frames] == [0, 2, 4]
    assert all(item.result.depth.shape == (2, 3) for item in frames)
    assert capture.released


def test_video_stride_one_processes_every_frame_and_emits_observations():
    sink = MemorySink()
    stream = VideoStream(
        FakeCapture(4),
        lambda image: np.zeros(image.shape[:2]),
        sink=sink,
        source_id="camera-7",
    )
    frames = list(stream)
    assert [item.frame_index for item in frames] == [0, 1, 2, 3]
    assert [
        item.frame_id for item in [bundle.frame for bundle in sink.observations]
    ] == [
        0,
        1,
        2,
        3,
    ]
    assert sink.observations[0].frame.source_id == "camera-7"


def test_video_batch_backend_preserves_order_and_respects_max_frames():
    class BatchBackend:
        def __init__(self):
            self.calls = []

        def extract_batch(self, images):
            self.calls.append(len(images))
            return [{"depth": image[..., 0].astype(np.float32)} for image in images]

    backend = BatchBackend()
    stream = VideoStream(FakeCapture(6), backend, batch_size=2, max_frames=5)
    frames = list(stream)
    assert [item.frame_index for item in frames] == [0, 1, 2, 3, 4]
    assert [int(item.result.depth[0, 0]) for item in frames] == [0, 1, 2, 3, 4]
    assert backend.calls == [2, 2, 1]
    assert stream.stats.frames_processed == 5


def test_video_batch_failure_recovers_frame_by_frame_when_skipping():
    class FailingBatch:
        def __init__(self):
            self.batch_calls = 0
            self.single_calls = 0

        def extract_batch(self, images):
            self.batch_calls += 1
            raise RuntimeError("batch unavailable")

        def extract(self, image):
            self.single_calls += 1
            if int(image[0, 0, 0]) == 1:
                raise ValueError("bad frame")
            return {"depth": image[..., 0]}

    backend = FailingBatch()
    stream = VideoStream(FakeCapture(3), backend, batch_size=3, error_policy="skip")
    frames = list(stream)
    assert [item.frame_index for item in frames] == [0, 2]
    assert backend.batch_calls == 1
    assert backend.single_calls == 3
    assert stream.stats.frames_dropped == 1


def test_video_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        VideoStream(FakeCapture(1), lambda image: image, batch_size=0)


def test_video_max_frames_and_metadata():
    capture = FakeCapture(4)
    stream = VideoStream(capture, lambda image: np.zeros(image.shape[:2]))
    frames = list(VideoStream(capture, stream.extractor, max_frames=2))
    assert len(frames) == 2
    assert frames[0].processing_time >= 0
    assert frames[0].result.timestamp is not None


def test_generic_semantic_fields_are_preserved():
    result = Yase(
        task="depth",
        extractor=lambda image: {
            "depth": image[..., 0],
            "detections": [{"label": "object", "score": 0.9}],
            "tags": ["scene"],
            "embeddings": [1.0, 2.0],
            "backend_version": "test",
        },
    ).extract(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result.detections[0]["label"] == "object"
    assert result.tags == ["scene"]
    assert result.embeddings.shape == (2,)
    assert result.metadata == {"backend_version": "test"}


def test_callable_extractor_supports_both_outputs():
    backend = CallableExtractor(
        lambda image: (np.ones(image.shape[:2]), np.zeros(image.shape[:2])),
        task="both",
    )
    result = backend.extract(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result.depth.shape == result.segmentation.shape


def test_direct_callable_backend_enforces_input_limits():
    backend = CallableExtractor(
        lambda image: image[..., 0],
        input_limits=InputLimits(max_pixels=4),
    )
    assert backend.extract(np.zeros((2, 2, 3), dtype=np.uint8)).depth.shape == (2, 2)
    with pytest.raises(InputError, match="pixel limit"):
        backend.extract(np.zeros((3, 2, 3), dtype=np.uint8))


def test_video_stats_count_skipped_frames():
    capture = FakeCapture(5)
    stream = VideoStream(capture, lambda image: np.zeros(image.shape[:2]), stride=2)
    list(stream)
    assert isinstance(stream.stats, VideoStats)
    assert stream.stats.frames_read == 5
    assert stream.stats.frames_processed == 3
    assert stream.stats.frames_dropped == 2
    assert stream.stats.output_fps >= 0


def test_video_error_policy_skip_and_callback():
    capture = FakeCapture(3)
    calls = []

    def failing(_image):
        raise RuntimeError("bad frame")

    frames = list(VideoStream(capture, failing, error_policy="skip"))
    assert frames == []
    assert capture.released
    assert (
        VideoStream(FakeCapture(1), failing, error_policy="skip").error_policy == "skip"
    )

    def recover(error, index):
        calls.append((str(error), index))
        return SemanticResult(depth=np.zeros((2, 3)))

    frames = list(VideoStream(FakeCapture(2), failing, on_error=recover))
    assert len(frames) == 2
    assert calls == [("bad frame", 0), ("bad frame", 1)]


def test_video_error_policy_validation():
    with pytest.raises(ValueError, match="error_policy"):
        VideoStream(FakeCapture(1), lambda image: image, error_policy="ignore")


def test_video_stream_applies_input_limits_before_callable_backend():
    limited = VideoStream(
        FakeCapture(1),
        lambda image: np.zeros(image.shape[:2]),
        input_limits=InputLimits(max_width=2),
        error_policy="skip",
    )
    assert list(limited) == []
    assert limited.stats.frames_dropped == 1


def test_video_stream_records_metrics():
    metrics = RuntimeMetrics(namespace="video_test")
    stream = VideoStream(
        FakeCapture(3),
        lambda image: np.zeros(image.shape[:2]),
        stride=2,
        metrics=metrics,
    )
    list(stream)
    snapshot = metrics.snapshot()["video"]
    assert snapshot["frames_processed"] == 2
    assert snapshot["frames_dropped"] == 1
    assert "video_test_video_frames_processed_total 2" in metrics.prometheus_text()


def test_image_path_and_pillow_inputs(tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.fromarray(np.full((2, 3, 3), 7, dtype=np.uint8)).save(path)
    backend = CallableExtractor(lambda image: image[..., 0])
    assert backend.extract(path).depth.shape == (2, 3)
    assert backend.extract(Image.open(path)).depth.shape == (2, 3)


def test_load_image_rejects_invalid_shapes_and_order():
    with pytest.raises(ValueError, match="dimensions"):
        load_image(np.zeros((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="channels"):
        load_image(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="color_order"):
        load_image(np.zeros((2, 2, 3)), color_order="XYZ")


def test_input_limits_validate_decoded_images_and_yase_inputs():
    limits = InputLimits(max_pixels=4, max_width=2, max_height=2, max_bytes=12)
    valid = np.zeros((2, 2, 3), dtype=np.uint8)
    assert limits.validate(valid) is valid
    with pytest.raises(InputError, match="pixel limit"):
        InputLimits(max_pixels=3).validate(valid)
    with pytest.raises(InputError, match="width limit"):
        Yase(
            extractor=lambda image: image[..., 0],
            input_limits=InputLimits(max_width=2),
        ).extract(np.zeros((2, 3, 3), dtype=np.uint8))
    result = Yase(
        extractor=lambda image: image[..., 0],
        input_limits=InputLimits(max_pixels=4),
    ).extract(valid)
    assert result.depth.shape == (2, 2)
    with pytest.raises(ValueError, match="positive"):
        InputLimits(max_channels=0)


def test_yase_records_direct_extraction_metrics_on_success_and_failure():
    metrics = RuntimeMetrics(namespace="image_test")
    api = Yase(
        extractor=lambda image: image[..., 0],
        input_limits=InputLimits(max_width=2),
        metrics=metrics,
    )
    api.extract(np.zeros((1, 2, 3), dtype=np.uint8))
    with pytest.raises(InputError):
        api.extract(np.zeros((1, 3, 3), dtype=np.uint8))
    extraction = metrics.snapshot()["extraction"]
    assert extraction["attempts"] == 2
    assert extraction["failures"] == 1
    assert "image_test_extraction_attempts_total 2" in metrics.prometheus_text()


def test_opentelemetry_tracer_instruments_yase_and_records_errors():
    class Span:
        def __init__(self):
            self.attributes = {}
            self.exceptions = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def record_exception(self, error):
            self.exceptions.append(error)

    class Tracer:
        def __init__(self):
            self.spans = []

        def start_as_current_span(self, name):
            span = Span()
            self.spans.append((name, span))
            return span

    tracer = Tracer()
    bridge = OpenTelemetryTracer(tracer=tracer)
    api = Yase(extractor=lambda image: image[..., 0], tracer=bridge)
    result = api.extract(np.ones((1, 1, 3), dtype=np.uint8))
    assert result.depth.shape == (1, 1)
    assert tracer.spans[0][0] == "yase.extract"
    assert tracer.spans[0][1].attributes == {
        "yase.task": "depth",
        "yase.model": "custom",
    }

    failing = Yase(
        extractor=lambda image: (_ for _ in ()).throw(RuntimeError("bad")),
        tracer=bridge,
    )
    with pytest.raises(RuntimeError, match="bad"):
        failing.extract(np.ones((1, 1, 3), dtype=np.uint8))
    assert isinstance(tracer.spans[-1][1].exceptions[0], RuntimeError)


def test_asgi_service_exposes_health_metrics_and_safe_image_extraction():
    from PIL import Image

    image_buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(7, 8, 9)).save(image_buffer, format="PNG")
    encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    metrics = RuntimeMetrics(namespace="http_test")
    api = Yase(extractor=lambda image: image[..., 0], metrics=metrics)
    app = create_asgi_app(api)
    assert isinstance(app, YaseASGI)

    async def request(target, method, path, payload=b""):
        events = [{"type": "http.request", "body": payload, "more_body": False}]
        sent = []

        async def receive():
            return events.pop(0)

        async def send(message):
            sent.append(message)

        await target({"type": "http", "method": method, "path": path}, receive, send)
        return sent

    health = asyncio.run(request(app, "GET", "/health"))
    assert health[0]["status"] == 200
    assert json.loads(health[1]["body"])["status"] == "ok"

    response = asyncio.run(
        request(
            app,
            "POST",
            "/extract",
            json.dumps({"image_base64": encoded, "timestamp": 12.5}).encode("utf-8"),
        )
    )
    payload = json.loads(response[1]["body"])
    assert response[0]["status"] == 200
    assert payload["result"]["depth"] == {"dtype": "uint8", "shape": [1, 1]}
    assert payload["result"]["timestamp"] == 12.5

    batch_response = asyncio.run(
        request(
            app,
            "POST",
            "/extract/batch",
            json.dumps(
                {
                    "images_base64": [encoded, encoded],
                    "timestamps": [1.0, 2.0],
                    "error_policy": "skip",
                }
            ).encode("utf-8"),
        )
    )
    batch_payload = json.loads(batch_response[1]["body"])
    assert batch_response[0]["status"] == 200
    assert [item["timestamp"] for item in batch_payload["results"]] == [1.0, 2.0]

    invalid = asyncio.run(
        request(app, "POST", "/extract", b'{"image_base64":"not-base64"}')
    )
    assert invalid[0]["status"] == 400
    assert json.loads(invalid[1]["body"])["error"]["type"] == "invalid_request"

    too_large = asyncio.run(
        request(create_asgi_app(api, max_body_bytes=2), "POST", "/extract", b"123")
    )
    assert too_large[0]["status"] == 413
    too_many = asyncio.run(
        request(
            create_asgi_app(api, max_batch_size=1),
            "POST",
            "/extract/batch",
            json.dumps({"images_base64": [encoded, encoded]}).encode("utf-8"),
        )
    )
    assert too_many[0]["status"] == 400


def test_normalise_tuple_and_result_timestamp():
    both = Yase(
        task="both",
        extractor=lambda image: (np.ones((2, 2)), np.zeros((2, 2))),
    ).extract(np.zeros((2, 2, 3)), timestamp=4.0)
    assert both.timestamp == 4.0
    assert both.segmentation.sum() == 0
    original = SemanticResult(tags=["x"], timestamp=1)
    result = Yase(extractor=lambda image: original).extract(
        np.zeros((1, 1, 3)), timestamp=2
    )
    assert result.tags == ["x"] and result.timestamp == 2


def test_default_backend_requires_explicit_configuration():
    with pytest.raises(ValueError, match="backend"):
        Yase().extract(np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="backend"):
        Yase(model="unknown").extract(np.zeros((2, 2, 3)))


def test_callable_backend_validation_and_segmentation():
    with pytest.raises(TypeError, match="callable"):
        CallableExtractor(None)
    backend = CallableExtractor(
        lambda image: np.ones(image.shape[:2]), task="segmentation"
    )
    assert backend.extract(np.zeros((2, 2, 3))).segmentation.shape == (2, 2)
    with pytest.raises(TypeError, match="pair"):
        CallableExtractor(lambda image: image, task="both").extract(np.zeros((2, 2, 3)))


def test_torchscript_backend_reports_optional_dependency():
    from yase.backends import TorchScriptExtractor

    with pytest.raises(ImportError, match="torch"):
        TorchScriptExtractor("missing.pt")


class FailingCapture(FakeCapture):
    def read(self):
        raise OSError("camera disconnected")


def test_realtime_latest_frame_drops_backlog_and_closes():
    capture = FakeCapture(30)

    def slow_backend(image):
        import time as _time

        _time.sleep(0.001)
        return np.zeros(image.shape[:2])

    stream = RealtimeVideoStream(capture, slow_backend, max_frames=1)
    frames = list(stream)
    assert len(frames) == 1
    assert stream.stats.frames_read >= 1
    assert stream.stats.frames_dropped >= 0
    assert not stream._worker
    assert capture.released


def test_realtime_backpressure_disables_drops():
    capture = FakeCapture(3)
    stream = RealtimeVideoStream(
        capture, lambda image: np.zeros(image.shape[:2]), drop_frames=False
    )
    frames = list(stream)
    assert len(frames) == 3
    assert stream.stats.frames_dropped == 0


def test_realtime_worker_exception_is_propagated_and_closed():
    capture = FailingCapture(1)
    stream = RealtimeVideoStream(capture, lambda image: image)
    with pytest.raises(RuntimeError, match="worker"):
        list(stream)
    stream.close()


def test_realtime_skip_errors_are_counted_as_dropped_frames():
    def failing(_image):
        raise RuntimeError("inference failed")

    stream = RealtimeVideoStream(
        FakeCapture(3), failing, drop_frames=False, error_policy="skip"
    )
    assert list(stream) == []
    assert stream.stats.frames_read == 3
    assert stream.stats.frames_processed == 0
    assert stream.stats.frames_dropped == 3


def test_realtime_input_limit_errors_are_counted_as_dropped_frames():
    stream = RealtimeVideoStream(
        FakeCapture(2),
        lambda image: image,
        drop_frames=False,
        error_policy="skip",
        input_limits=InputLimits(max_width=2),
    )
    assert list(stream) == []
    assert stream.stats.frames_dropped == 2


def test_rich_schema_and_pipeline_preserve_semantic_fields():
    box = BoundingBox.from_xywh(1, 2, 3, 4)
    result = SemanticPipeline(
        {
            "objects": lambda _image: {
                "detections": [Detection("person", 0.9, box)],
                "ocr": [TextRegion("hello", 0.8, box)],
            },
            "events": lambda _image: {
                "events": [SemanticEvent("arrival", 0.7, 1.0)],
            },
        }
    ).extract(np.zeros((4, 4, 3), dtype=np.uint8), timestamp=3.0)
    assert result.detections[0].box.area == 12
    assert result.ocr[0].text == "hello"
    assert result.events[0].label == "arrival"
    assert result.timestamp == 3.0
    batch = SemanticPipeline(
        {"objects": lambda image: {"tags": [int(image[0, 0, 0])]}}
    ).extract_many(
        [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)],
        timestamps=[1.0, 2.0],
    )
    assert [item.tags for item in batch] == [[0], [1]]


def test_pipeline_batch_skip_isolates_invalid_inputs_and_reports_stage():
    class BatchStage:
        def extract_batch(self, images):
            return [{"tags": [int(image[0, 0, 0])]} for image in images]

    pipeline = SemanticPipeline({"tags": BatchStage()})
    seen = []
    results = pipeline.extract_many(
        [
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1,), dtype=np.uint8),
            np.ones((1, 1, 3), dtype=np.uint8),
        ],
        error_policy="skip",
        on_error=lambda error, index, stage: (
            seen.append((type(error), index, stage)) or None
        ),
    )
    assert results[0].tags == [0]
    assert results[1] is None
    assert results[2].tags == [1]
    assert seen == [(ValueError, 1, "tags")]


def test_numpy_vector_index_search_and_result_ingestion():
    index = NumpyVectorIndex()
    index.add_result("a", SemanticResult(embeddings=np.array([1.0, 0.0])))
    index.add("b", [0.0, 1.0], metadata={"kind": "other"})
    hits = index.search([0.9, 0.1], limit=2)
    assert [hit.item_id for hit in hits] == ["a", "b"]
    assert hits[1].metadata["kind"] == "other"
    assert index.search([0.0, 1.0], where={"kind": "other"})[0].item_id == "b"


def test_numpy_vector_index_persistence(tmp_path):
    index = NumpyVectorIndex()
    index.add("a", [1.0, 0.0], metadata={"source": "test"})
    path = tmp_path / "vectors.npz"
    index.save(path)
    restored = NumpyVectorIndex.load(path)
    assert restored.search([1.0, 0.0])[0].metadata["source"] == "test"


def test_backend_registry_is_explicit_and_replaceable():
    registry = BackendRegistry()
    registry.register("fake", lambda value=1: value, capabilities=("test",))
    assert registry.create("fake", value=3) == 3
    assert registry.get("fake").capabilities == ("test",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("fake", lambda: 2)


def test_yase_can_inject_a_registry_backend():
    registry = BackendRegistry()
    registry.register("mean", lambda **_options: lambda image: image[..., 0])
    api = Yase(model="mean", registry=registry)
    result = api.extract(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result.depth.shape == (2, 2)
    with pytest.raises(TypeError, match="registry"):
        Yase(model="mean", registry=object())


def test_backend_registry_discovers_optional_entry_points(monkeypatch):
    from importlib import metadata

    class EntryPoint:
        name = "plugin"
        value = "tests.plugin:factory"

        def load(self):
            return lambda **_options: "loaded"

    class EntryPoints(list):
        def select(self, **_criteria):
            return self

    monkeypatch.setattr(metadata, "entry_points", lambda: EntryPoints([EntryPoint()]))
    registry = BackendRegistry()
    specs = registry.discover_entry_points()
    assert [spec.name for spec in specs] == ["plugin"]
    assert registry.create("plugin") == "loaded"
    assert specs[0].metadata["entry_point"] == "tests.plugin:factory"
    with pytest.raises(ValueError, match="group"):
        registry.discover_entry_points("")


def test_composite_reports_structured_backend_error():
    def failing(_image):
        raise ValueError("bad input")

    with pytest.raises(BackendError) as error:
        CompositeExtractor({"detector": failing}).extract(
            np.zeros((2, 2, 3), dtype=np.uint8)
        )
    assert error.value.backend == "detector"


def test_iou_tracker_keeps_ids_and_expires_missing_tracks():
    tracker = IoUTracker(iou_threshold=0.2, max_missed=1)
    first = tracker.update([Detection("person", 0.9, (0, 0, 10, 10))])
    second = tracker.update([Detection("person", 0.8, (1, 1, 11, 11))])
    assert first[0].track_id == second[0].track_id
    tracker.update([])
    assert tracker.active_ids
    tracker.update([])
    assert not tracker.active_ids


def test_trackers_round_trip_json_compatible_state():
    tracker = IoUTracker(start_id=10)
    first = tracker.update([Detection("person", 0.9, (0, 0, 10, 10))])
    state = json.loads(json.dumps(tracker.state_dict()))
    restored = IoUTracker(start_id=1)
    restored.load_state_dict(state)
    second = restored.update([Detection("person", 0.8, (1, 0, 11, 10))])
    assert isinstance(restored, Tracker)
    assert first[0].track_id == second[0].track_id == 10

    motion = ByteTrackLite(start_id=20)
    motion.update([Detection("person", 0.9, (0, 0, 10, 10))])
    motion.update([Detection("person", 0.9, (2, 0, 12, 10))])
    motion_state = json.loads(json.dumps(motion.state_dict()))
    restored_motion = ByteTrackLite(start_id=1)
    restored_motion.load_state_dict(motion_state)
    assert restored_motion.state_dict() == motion_state
    with pytest.raises(ValueError, match="state version"):
        restored.load_state_dict({"version": 2})
    with pytest.raises(ValueError, match="score"):
        restored.load_state_dict(
            {
                "version": 1,
                "next_id": 2,
                "tracks": [
                    {
                        "track_id": 1,
                        "box": [0, 0, 1, 1],
                        "label": "person",
                        "score": 2.0,
                    }
                ],
            }
        )


def test_iou_matrix_has_a_portable_fallback_and_native_contract():
    import yase.native as native

    values = native.iou_matrix([(0, 0, 2, 2)], [(1, 1, 3, 3)])
    assert values[0][0] == pytest.approx(1 / 7)
    assert isinstance(native.NATIVE_AVAILABLE, bool)
    compiled = native._native_iou_matrix
    native._native_iou_matrix = None
    try:
        fallback = native.iou_matrix([(0, 0, 2, 2)], [(1, 1, 3, 3)])
    finally:
        native._native_iou_matrix = compiled
    assert fallback[0][0] == pytest.approx(1 / 7)


def test_nms_indices_is_class_aware_and_has_a_portable_fallback():
    import yase.native as native

    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (1, 1, 11, 11)]
    scores = [0.8, 0.9, 0.7]
    assert nms_indices(boxes, scores, 0.5) == [1]
    assert nms_indices(boxes, scores, 0.5, [1, 1, 2]) == [1, 2]
    compiled = native._native_nms_indices
    native._native_nms_indices = None
    try:
        fallback = native.nms_indices(boxes, scores, 0.5, [1, 1, 2])
        fallback_without_classes = native.nms_indices(boxes, scores, 0.5)
    finally:
        native._native_nms_indices = compiled
    assert fallback == [1, 2]
    assert fallback_without_classes == [1]
    with pytest.raises(ValueError, match="same length"):
        nms_indices(boxes, scores[:2])


def test_nms_indices_handles_empty_and_deterministic_boundary_cases():
    assert nms_indices([], []) == []
    overlapping = [(0, 0, 2, 2), (0, 0, 2, 2)]
    assert nms_indices(overlapping, [0.5, 0.5], 0.5) == [0]
    assert nms_indices(overlapping, [0.5, 0.5], 0.5, [1, 2]) == [0, 1]
    assert nms_indices([(0, 0, 1, 1)], [0.5], 0.0) == [0]
    with pytest.raises(ValueError, match="iou_threshold"):
        nms_indices([], [], -0.1)
    with pytest.raises(ValueError, match="class_ids"):
        nms_indices([], [], class_ids=[1])
    with pytest.raises(ValueError, match="finite"):
        nms_indices([(0, 0, float("nan"), 1)], [0.5])
    with pytest.raises(ValueError, match="maximum coordinates"):
        nms_indices([(2, 0, 1, 1)], [0.5])


def test_typed_non_maximum_suppression_preserves_detection_payload():
    first = Detection(
        "person", 0.8, (0, 0, 10, 10), track_id=7, attributes={"source": "a"}
    )
    winner = Detection(
        "person", 0.9, (1, 1, 11, 11), track_id=8, attributes={"source": "b"}
    )
    other_class = Detection("car", 0.7, (1, 1, 11, 11), track_id=9)
    assert non_maximum_suppression([first, winner, other_class]) == [
        winner,
        other_class,
    ]
    assert non_maximum_suppression([first, winner, other_class], class_aware=False) == [
        winner
    ]


def test_bytetrack_lite_uses_the_same_native_matching_contract():
    tracker = ByteTrackLite(high_threshold=0.5, low_threshold=0.1)
    first = tracker.update([Detection("person", 0.9, (0, 0, 10, 10))])
    second = tracker.update([Detection("person", 0.4, (1, 1, 11, 11))])
    assert first[0].track_id == second[0].track_id


def test_presence_event_engine_emits_enter_and_exit():
    engine = EventEngine([PresenceRule("person", enter_frames=2, exit_frames=2)])
    detection = Detection("person", 0.9, (0, 0, 10, 10), track_id=3)
    assert engine.update(SemanticResult(detections=[detection]), 1.0) == []
    entered = engine.update(SemanticResult(detections=[detection]), 2.0)
    assert entered[0].metadata["type"] == "enter"
    assert engine.update(SemanticResult(detections=[]), 3.0) == []
    exited = engine.update(SemanticResult(detections=[]), 4.0)
    assert exited[0].metadata["type"] == "exit"


def test_qdrant_adapter_works_with_injected_client_and_models():
    from types import SimpleNamespace

    class FakeModels:
        class Distance:
            COSINE = "cosine"

        VectorParams = staticmethod(lambda **kwargs: kwargs)
        PointStruct = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
        Filter = staticmethod(lambda **kwargs: kwargs)
        FieldCondition = staticmethod(lambda **kwargs: kwargs)
        MatchValue = staticmethod(lambda **kwargs: kwargs)

    class FakeClient:
        def __init__(self):
            self.created = False
            self.points = []

        def collection_exists(self, _name):
            return self.created

        def create_collection(self, **_kwargs):
            self.created = True

        def upsert(self, points, **_kwargs):
            self.points.extend(points)

        def query_points(self, **_kwargs):
            point = SimpleNamespace(id="a", score=1.0, payload={"kind": "test"})
            return SimpleNamespace(points=[point])

    index = QdrantVectorIndex(
        "images", dimension=2, client=FakeClient(), models=FakeModels
    )
    index.add("a", [1.0, 0.0], {"kind": "test"})
    assert index.search([1.0, 0.0])[0].item_id == "a"
    assert index.search([1.0, 0.0], min_score=0.9)[0].item_id == "a"
    with pytest.raises(ValueError, match="min_score"):
        index.search([1.0, 0.0], min_score=float("nan"))


def test_qdrant_adapter_enforces_embedding_spaces():
    from types import SimpleNamespace

    class FakeModels:
        class Distance:
            COSINE = "cosine"

        VectorParams = staticmethod(lambda **kwargs: kwargs)
        PointStruct = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
        Filter = staticmethod(lambda **kwargs: kwargs)
        FieldCondition = staticmethod(lambda **kwargs: kwargs)
        MatchValue = staticmethod(lambda **kwargs: kwargs)

    class FakeClient:
        def __init__(self):
            self.points = []
            self.query = None

        def collection_exists(self, _name):
            return False

        def create_collection(self, **_kwargs):
            pass

        def upsert(self, points, **_kwargs):
            self.points.extend(points)

        def query_points(self, **kwargs):
            self.query = kwargs
            return SimpleNamespace(
                points=[SimpleNamespace(id="a", score=0.99, payload={"space": "clip"})]
            )

    client = FakeClient()
    index = QdrantVectorIndex(
        "embeddings", dimension=2, client=client, models=FakeModels, space="clip"
    )
    record = EmbeddingRecord(
        vector=np.array([1.0, 0.0]), space="clip", model_id="model-a"
    )
    index.add_record("a", record)
    assert client.points[0].payload == {"space": "clip", "model_id": "model-a"}
    assert index.search_record(record, min_score=0.9)[0].item_id == "a"
    assert client.query["score_threshold"] == 0.9
    with pytest.raises(ValueError, match="space clip"):
        index.add_record(
            "b",
            EmbeddingRecord(
                vector=np.array([0.0, 1.0]), space="dino", model_id="model-b"
            ),
        )
    with pytest.raises(ValueError, match="space clip"):
        index.search_record(
            EmbeddingRecord(
                vector=np.array([0.0, 1.0]), space="dino", model_id="model-b"
            )
        )


def test_qdrant_adapter_supports_named_vectors_and_payload_indexes():
    from types import SimpleNamespace

    class FakeModels:
        class Distance:
            COSINE = "cosine"

        VectorParams = staticmethod(lambda **kwargs: kwargs)
        PointStruct = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
        Filter = staticmethod(lambda **kwargs: kwargs)
        FieldCondition = staticmethod(lambda **kwargs: kwargs)
        MatchValue = staticmethod(lambda **kwargs: kwargs)

    class FakeClient:
        def __init__(self):
            self.created = None
            self.indexes = []
            self.points = []
            self.query = None

        def collection_exists(self, _name):
            return False

        def create_collection(self, **kwargs):
            self.created = kwargs

        def create_payload_index(self, **kwargs):
            self.indexes.append(kwargs)

        def upsert(self, points, **_kwargs):
            self.points.extend(points)

        def query_points(self, **kwargs):
            self.query = kwargs
            return SimpleNamespace(
                points=[SimpleNamespace(id="a", score=0.9, payload={})]
            )

    client = FakeClient()
    index = QdrantVectorIndex(
        "multimodal",
        dimension=2,
        client=client,
        models=FakeModels,
        vector_name="image",
        payload_indexes={"camera_id": "keyword"},
    )
    index.add("a", [1.0, 0.0], {"camera_id": "cam-1"})
    hits = index.search([1.0, 0.0], where={"camera_id": "cam-1"})

    assert client.created["vectors_config"]["image"]["size"] == 2
    assert client.indexes == [
        {
            "collection_name": "multimodal",
            "field_name": "camera_id",
            "field_schema": "keyword",
        }
    ]
    assert client.points[0].vector == {"image": [1.0, 0.0]}
    assert client.query["using"] == "image"
    assert hits[0].item_id == "a"


def test_video_tracker_and_event_engine_integration():
    capture = FakeCapture(3)
    detection = Detection("person", 0.9, (0, 0, 2, 2))

    def extractor(_image):
        return SemanticResult(detections=[detection])

    stream = VideoStream(
        capture,
        extractor,
        tracker=IoUTracker(iou_threshold=0.1),
        event_engine=EventEngine([PresenceRule("person")]),
    )
    frames = list(stream)
    assert frames[0].result.detections[0].track_id == 1
    assert frames[0].result.events[0].metadata["type"] == "enter"


def test_video_memory_runs_after_tracker():
    capture = FakeCapture(2)
    detection = Detection("person", 0.9, (0, 0, 2, 2))
    memory = SemanticTrackMemory()
    stream = VideoStream(
        capture,
        lambda _image: SemanticResult(detections=[detection]),
        tracker=IoUTracker(iou_threshold=0.1),
        memory=memory,
    )
    frames = list(stream)
    assert frames[1].result.detections[0].attributes["seen_count"] == 2


def test_video_stream_checkpoint_methods_restore_attached_state(tmp_path):
    detection = Detection("person", 0.9, (0, 0, 2, 2), track_id=None)
    tracker = IoUTracker()
    memory = SemanticTrackMemory()
    stream = VideoStream(
        FakeCapture(2),
        lambda _image: SemanticResult(detections=[detection]),
        tracker=tracker,
        memory=memory,
        source_id="camera-9",
        camera_id="cam-9",
    )
    list(stream)
    checkpoint = tmp_path / "video-state.json"
    stream.save_checkpoint(checkpoint, metadata={"job_id": "resume-me"})

    restored_tracker = IoUTracker()
    restored_memory = SemanticTrackMemory()
    restored_stream = VideoStream(
        FakeCapture(0),
        lambda _image: SemanticResult(detections=[]),
        tracker=restored_tracker,
        memory=restored_memory,
    )
    payload = restored_stream.load_checkpoint(checkpoint)
    assert payload["metadata"] == {
        "camera_id": "cam-9",
        "job_id": "resume-me",
        "source_id": "camera-9",
    }
    assert restored_tracker.active_ids == tracker.active_ids
    assert restored_memory.states == memory.states


def test_adaptive_cascade_refines_then_uses_fast_path():
    calls = {"fast": 0, "accurate": 0}
    box = BoundingBox(0, 0, 2, 2)

    def fast(_image):
        calls["fast"] += 1
        return {"detections": [Detection("person", 0.95, box)]}

    def accurate(_image):
        calls["accurate"] += 1
        return {"detections": [Detection("person", 0.99, box)]}

    cascade = AdaptiveSemanticCascade(fast, accurate)
    first = cascade.extract(np.zeros((8, 8, 3), dtype=np.uint8))
    second = cascade.extract(np.zeros((8, 8, 3), dtype=np.uint8))
    assert first.metadata["cascade"]["route"] == "fast+accurate"
    assert second.metadata["cascade"]["route"] == "fast"
    assert calls == {"fast": 2, "accurate": 1}


def test_semantic_track_memory_adds_stable_attributes_and_expires():
    memory = SemanticTrackMemory(max_missed=1)
    detection = Detection("person", 0.8, (0, 0, 2, 2), track_id=4)
    first = memory.update(SemanticResult(detections=[detection]), timestamp=1)
    second = memory.update(SemanticResult(detections=[detection]), timestamp=2)
    assert first.detections[0].attributes["seen_count"] == 1
    assert second.detections[0].attributes["seen_count"] == 2
    memory.update(SemanticResult(detections=[]), timestamp=3)
    assert memory.states
    memory.update(SemanticResult(detections=[]), timestamp=4)
    assert not memory.states


def test_semantic_track_memory_round_trips_json_checkpoint():
    memory = SemanticTrackMemory(decay=0.5)
    detection = Detection("person", 0.8, (0, 0, 2, 2), track_id=4)
    memory.update(SemanticResult(detections=[detection]), timestamp=1.0)
    memory.update(
        SemanticResult(detections=[Detection("person", 0.6, (0, 0, 2, 2), track_id=4)]),
        timestamp=2.0,
    )
    restored = SemanticTrackMemory()
    restored.load_state_dict(json.loads(json.dumps(memory.state_dict())))
    result = restored.update(
        SemanticResult(detections=[Detection("person", 0.9, (0, 0, 2, 2), track_id=4)]),
        timestamp=3.0,
    )
    assert result.detections[0].attributes["seen_count"] == 3
    with pytest.raises(ValueError, match="state version"):
        restored.load_state_dict({"version": 2})


def test_stream_checkpoint_atomically_restores_all_state_components(tmp_path):
    detection = Detection(
        "person", 0.9, (0, 0, 2, 2), attributes={"embedding": [1.0, 0.0]}
    )
    tracker = IoUTracker()
    tracked = tracker.update([detection], timestamp=1.0)
    memory = SemanticTrackMemory()
    memory.update(SemanticResult(detections=tracked), timestamp=1.0)
    identity_store = GlobalIdentityStore(start_id=7)
    identity_store.update([detection], camera_id="cam-a", timestamp=1.0)
    checkpoint = tmp_path / "stream.json"

    assert (
        save_stream_checkpoint(
            checkpoint,
            tracker=tracker,
            memory=memory,
            identity_store=identity_store,
            metadata={"job_id": "demo", "frame": 1},
        )
        == checkpoint
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert set(payload["components"]) == {"tracker", "memory", "identity_store"}

    restored_tracker = IoUTracker()
    restored_memory = SemanticTrackMemory()
    restored_identity = GlobalIdentityStore()
    restored = load_stream_checkpoint(
        checkpoint,
        tracker=restored_tracker,
        memory=restored_memory,
        identity_store=restored_identity,
    )
    assert restored["metadata"]["job_id"] == "demo"
    assert restored_tracker.active_ids == tracker.active_ids
    assert restored_memory.states == memory.states
    assert tuple(restored_identity.identities) == tuple(identity_store.identities)


def test_stream_checkpoint_validates_components_and_versions(tmp_path):
    with pytest.raises(TypeError, match="state_dict"):
        make_stream_checkpoint(tracker=object())
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_stream_checkpoint(path)


def test_multimodal_consensus_fuses_boxes_and_audits_evidence():
    box_a = Detection("car", 0.8, (0, 0, 10, 10))
    box_b = Detection("car", 0.9, (1, 1, 11, 11))
    result = MultimodalConsensus().fuse(
        {
            "detector": SemanticResult(detections=[box_a]),
            "vlm": SemanticResult(detections=[box_b]),
        }
    )
    assert len(result.detections) == 1
    assert result.detections[0].attributes["evidence_count"] == 2
    assert result.metadata["consensus"][0]["abstained"] is False


def test_benchmark_runner_reports_latency_and_quality():
    def extractor(_image):
        return SemanticResult(tags=["ok"])

    report = BenchmarkRunner().run(
        extractor,
        [np.zeros((2, 2, 3), dtype=np.uint8)] * 3,
        evaluator=lambda result, _item: {"has_tags": float(bool(result.tags))},
    )
    assert report.samples == 3
    assert report.failures == 0
    assert report.throughput > 0
    assert report.to_dict()["quality"]["has_tags"] == 1.0


def test_bytetrack_lite_uses_low_score_detections_for_existing_tracks():
    tracker = ByteTrackLite(high_threshold=0.7, low_threshold=0.2, iou_threshold=0.1)
    first = tracker.update([Detection("person", 0.9, (0, 0, 10, 10))])
    second = tracker.update([Detection("person", 0.3, (1, 0, 11, 10))])
    assert first[0].track_id == second[0].track_id
    assert tracker.update([Detection("person", 0.1, (2, 0, 12, 10))]) == []


def test_rfdetr_adapter_normalizes_injected_model():
    class Raw:
        xyxy = np.asarray([[0, 0, 4, 4]])
        confidence = np.asarray([0.8])
        class_id = np.asarray([2])
        class_name = ["car"]

    class Model:
        def predict(self, _image, threshold):
            assert threshold == 0.5
            return Raw()

    result = RFDETRExtractor(model=Model()).extract(np.zeros((4, 4, 3), dtype=np.uint8))
    assert result.detections[0].label == "car"
    assert result.metadata["backend"] == "rf-detr"


def test_promptable_segmentation_adapter_normalizes_predictor():
    class Predictor:
        def segment(self, _image, prompt):
            assert prompt == "the red car"
            return {"mask": np.ones((4, 4), dtype=np.uint8), "quality": 0.9}

    result = PromptableSegmentationExtractor(Predictor()).extract(
        np.zeros((4, 4, 3), dtype=np.uint8), prompt="the red car"
    )
    assert result.segmentation.shape == (4, 4)
    assert result.metadata["quality"] == 0.9


def test_transformers_sam3_adapter_normalizes_masks_boxes_and_scores():
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(name):
            return name

        @staticmethod
        @contextmanager
        def inference_mode():
            yield

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class Processor:
        def __call__(self, **_kwargs):
            return FakeInputs(original_sizes=np.asarray([[4, 5]]))

        def post_process_instance_segmentation(self, _outputs, **_kwargs):
            return [
                {
                    "masks": np.asarray([[[1, 0, 0, 0, 0]] * 4], dtype=np.uint8),
                    "boxes": np.asarray([[0, 0, 1, 4]], dtype=np.float32),
                    "scores": np.asarray([0.8], dtype=np.float32),
                }
            ]

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **_kwargs):
            return object()

    result = TransformersSAM3Extractor(
        prompt="person",
        processor=Processor(),
        model=Model(),
        torch_module=FakeTorch,
    ).extract(np.zeros((4, 5, 3), dtype=np.uint8))
    assert result.segmentation.shape == (1, 4, 5)
    assert result.detections[0].label == "person"


def test_model_catalog_and_detection_metrics_are_serializable():
    catalog = default_model_catalog()
    assert "sam3" in catalog.names()
    assert catalog.get("rf-detr").task == "detection"
    metrics = evaluate_detections(
        [Detection("car", 0.9, (0, 0, 10, 10))],
        [Detection("car", 1.0, (1, 1, 11, 11))],
    )
    assert isinstance(metrics, DetectionMetrics)
    assert metrics.f1 == 1.0
    assert metrics.to_dict()["mean_iou"] > 0.5


def test_structured_query_parses_and_validates_vlm_json():
    schema = {
        "type": "object",
        "required": ["label", "score"],
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string"},
            "score": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }
    query = StructuredQuery("Extract the object", schema, max_new_tokens=32)
    assert query.to_dict()["max_new_tokens"] == 32
    parsed = parse_structured_output(
        'Here is the result: ```json\n{"label":"car","score":0.91}\n```',
        schema,
    )
    assert parsed == {"label": "car", "score": 0.91}
    with pytest.raises(ValueError, match="required"):
        parse_structured_output('{"label":"car"}', schema)
    with pytest.raises(ValueError, match="unknown fields"):
        parse_structured_output('{"label":"car","score":0.9,"other":true}', schema)


def test_transformers_vlm_supports_injected_structured_generation():
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(name):
            return name

        @staticmethod
        @contextmanager
        def inference_mode():
            yield

    class InputValue:
        shape = (1, 1)

        def to(self, _device):
            return self

    class Processor:
        def __call__(self, **_kwargs):
            return {"input_ids": InputValue()}

        def batch_decode(self, _generated, skip_special_tokens):
            assert skip_special_tokens is True
            return ['{"label":"person","score":0.88}']

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 24
            return np.asarray([[1, 2]])

    query = StructuredQuery(
        "Return a person object",
        {
            "type": "object",
            "required": ["label", "score"],
            "properties": {
                "label": {"type": "string"},
                "score": {"type": "number"},
            },
        },
        max_new_tokens=24,
    )
    result = TransformersVLMExtractor(
        model_id="fake",
        processor=Processor(),
        model=Model(),
        torch_module=FakeTorch,
    ).ask_structured(np.zeros((2, 2, 3), dtype=np.uint8), query)
    assert result.scene == {"label": "person", "score": 0.88}
    assert result.metadata["structured"] is True


def test_transformers_vlm_batches_caption_and_structured_generation():
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(name):
            return name

        @staticmethod
        @contextmanager
        def inference_mode():
            yield

    class InputValue:
        shape = (2, 1)

        def to(self, _device):
            return self

    class Processor:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            assert len(kwargs["images"]) == 2
            assert kwargs["padding"] is True
            self.calls += 1
            return {"input_ids": InputValue()}

        def batch_decode(self, _generated, skip_special_tokens):
            assert skip_special_tokens is True
            return ['{"label":"car","score":0.9}', '{"label":"person","score":0.8}']

    class Model:
        def __init__(self):
            self.calls = 0

        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            self.calls += 1
            assert kwargs["max_new_tokens"] == 128
            return np.asarray([[1, 2], [3, 4]])

    processor = Processor()
    model = Model()
    extractor = TransformersVLMExtractor(
        model_id="fake",
        processor=processor,
        model=model,
        torch_module=FakeTorch,
    )
    images = [np.zeros((2, 2, 3), dtype=np.uint8)] * 2
    results = extractor.extract_batch(images)
    assert [result.caption for result in results] == [
        '{"label":"car","score":0.9}',
        '{"label":"person","score":0.8}',
    ]
    assert processor.calls == model.calls == 1
    query = StructuredQuery(
        "Return the object",
        {
            "type": "object",
            "required": ["label", "score"],
            "properties": {
                "label": {"type": "string"},
                "score": {"type": "number"},
            },
        },
    )
    structured = extractor.ask_structured_batch(images, query)
    assert structured[1].scene["label"] == "person"
    assert model.calls == 2


def test_transformers_sam3_video_adapter_preserves_object_ids():
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(name):
            return name

    class Processor:
        def init_video_session(self, **kwargs):
            assert len(kwargs["video"]) == 2
            return object()

        def add_text_prompt(self, inference_session, text):
            assert text == "person"
            return inference_session

        def postprocess_outputs(self, _session, output):
            return output

    class Model:
        def to(self, _device):
            return self

        def propagate_in_video_iterator(self, **_kwargs):
            for frame_idx in range(2):
                yield {
                    "frame_idx": frame_idx,
                    "masks": np.ones((1, 2, 2), dtype=np.uint8),
                    "boxes": np.asarray([[0, 0, 2, 2]], dtype=np.float32),
                    "scores": np.asarray([0.9]),
                    "object_ids": np.asarray([7]),
                }

    extractor = TransformersSAM3VideoExtractor(
        prompt="person",
        model=Model(),
        processor=Processor(),
        torch_module=FakeTorch,
    )
    results = extractor.extract_video([np.zeros((2, 2, 3), dtype=np.uint8)] * 2)
    assert len(results) == 2
    assert results[1].detections[0].track_id == 7


def test_cli_models_lists_filtered_model_cards(capsys):
    from yase.cli import main

    assert main(["models", "--task", "detection"]) == 0
    output = capsys.readouterr().out
    assert "rf-detr" in output
    assert "sam3" not in output


def test_default_registry_exposes_modern_optional_backends():
    registry = default_registry()
    names = registry.names()
    assert {"rf-detr", "sam3", "sam3-video", "grounding-dino"}.issubset(names)
    assert registry.get("vlm").capabilities == (
        "caption",
        "question-answering",
        "structured-output",
        "image",
    )
    assert registry.get("image-embedding").extra == "transformers"
    assert registry.get("tesseract").extra == "ocr"
    assert registry.get("paddleocr").extra == "paddle"
    extras = set(metadata("yase").get_all("Provides-Extra") or ())
    assert {spec.extra for spec in registry.specs() if spec.extra} <= extras


def test_zone_and_line_event_rules_emit_geometry_events():
    inside = Detection("person", 0.9, (1, 1, 3, 3), track_id=8)
    outside = Detection("person", 0.9, (8, 1, 10, 3), track_id=8)
    zone = ZoneRule("person", [(0, 0), (5, 0), (5, 5), (0, 5)])
    assert (
        zone.evaluate(SemanticResult(detections=[inside]), 1.0)[0].metadata["type"]
        == "zone_enter"
    )
    assert (
        zone.evaluate(SemanticResult(detections=[outside]), 2.0)[0].metadata["type"]
        == "zone_exit"
    )
    line = LineCrossingRule("person", (5, 0), (5, 10))
    line.evaluate(SemanticResult(detections=[inside]), 1.0)
    crossed = line.evaluate(SemanticResult(detections=[outside]), 2.0)
    assert crossed[0].metadata["type"] == "line_crossing"


def test_global_identity_store_matches_embeddings_across_cameras():
    store = GlobalIdentityStore(similarity_threshold=0.8)
    first = Detection("person", 0.9, (0, 0, 2, 2), attributes={"embedding": [1.0, 0.0]})
    second = Detection(
        "person", 0.9, (1, 1, 3, 3), attributes={"embedding": [0.99, 0.01]}
    )
    one = store.update([first], camera_id="cam-a", timestamp=1.0)
    two = store.update([second], camera_id="cam-b", timestamp=2.0)
    assert one[0].attributes["global_id"] == two[0].attributes["global_id"]
    assert store.identities[two[0].attributes["global_id"]].camera_ids == (
        "cam-a",
        "cam-b",
    )


def test_global_identity_store_round_trips_json_checkpoint():
    store = GlobalIdentityStore(start_id=10)
    detection = Detection(
        "person", 0.9, (0, 0, 2, 2), attributes={"embedding": [1.0, 0.0]}
    )
    store.update([detection], camera_id="cam-a", timestamp=1.0)
    restored = GlobalIdentityStore(start_id=1)
    restored.load_state_dict(json.loads(json.dumps(store.state_dict())))
    matched = restored.update(
        [
            Detection(
                "person", 0.8, (0, 0, 2, 2), attributes={"embedding": [0.99, 0.01]}
            )
        ],
        camera_id="cam-b",
        timestamp=2.0,
    )
    assert matched[0].attributes["global_id"] == 10
    assert restored.identities[10].camera_ids == ("cam-a", "cam-b")
    with pytest.raises(ValueError, match="state version"):
        restored.load_state_dict({"version": 2})


def test_numpy_index_enforces_embedding_spaces_and_persists_them(tmp_path):
    first = EmbeddingRecord(
        np.asarray([1.0, 0.0]), space="siglip", model_id="model-a", normalized=True
    )
    second = EmbeddingRecord(
        np.asarray([0.0, 1.0]), space="dinov2", model_id="model-b", normalized=True
    )
    index = NumpyVectorIndex()
    index.add_record("one", first)
    assert index.search_record(first)[0].item_id == "one"
    with pytest.raises(ValueError, match="space"):
        index.add_record("two", second)
    path = tmp_path / "vectors.npz"
    index.save(path)
    restored = NumpyVectorIndex.load(path)
    assert restored.space == "siglip"
    assert restored.search_record(first)[0].item_id == "one"
    with pytest.raises(ValueError, match="space"):
        restored.search_record(second)


def test_tracking_metrics_detect_identity_switches():
    truth = [
        [Detection("person", 1.0, (0, 0, 10, 10), track_id=4)],
        [Detection("person", 1.0, (1, 0, 11, 10), track_id=4)],
    ]
    predictions = [
        [Detection("person", 0.9, (0, 0, 10, 10), track_id=1)],
        [Detection("person", 0.9, (1, 0, 11, 10), track_id=2)],
    ]
    metrics = evaluate_tracking(predictions, truth)
    assert isinstance(metrics, TrackingMetrics)
    assert metrics.identity_switches == 1
    assert metrics.idf1 < 1.0


def test_hota_is_perfect_for_perfect_track_sequence():
    truth = [
        [Detection("person", 1.0, (0, 0, 10, 10), track_id=4)],
        [Detection("person", 1.0, (1, 0, 11, 10), track_id=4)],
    ]
    result = evaluate_hota(truth, truth)
    assert isinstance(result, HOTAResult)
    assert result.hota == 1.0
    assert result.association_accuracy == 1.0


def test_hota_curve_uses_default_mot_threshold_grid():
    truth = [[Detection("person", 1.0, (0, 0, 10, 10), track_id=4)]]
    result = evaluate_hota_curve((frame for frame in truth), truth)
    assert isinstance(result, HOTACurveResult)
    assert result.alphas == tuple(
        round(value, 2) for value in np.linspace(0.05, 0.95, 19)
    )
    assert result.mean_hota == pytest.approx(1.0)
    assert result.per_threshold[0].association_accuracy == pytest.approx(1.0)


def test_hota_curve_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="alphas"):
        evaluate_hota_curve([], [], alphas=[])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_hota_curve([[]], [[]], alphas=[1.1])


def test_average_precision_and_map_are_perfect_on_perfect_predictions():
    truth = [Detection("person", 1.0, (0, 0, 10, 10), track_id=4)]
    prediction = [Detection("person", 0.9, (0, 0, 10, 10), track_id=2)]
    average_precision = evaluate_average_precision(prediction, truth)
    mean_average_precision = evaluate_mean_average_precision([prediction], [truth])
    assert isinstance(average_precision, AveragePrecisionResult)
    assert average_precision.average_precision == pytest.approx(1.0)
    assert isinstance(mean_average_precision, MeanAveragePrecisionResult)
    assert mean_average_precision.mean_average_precision == pytest.approx(1.0)
    assert len(mean_average_precision.iou_thresholds) == 10


def test_average_precision_penalizes_high_score_false_positive():
    truth = [Detection("person", 1.0, (0, 0, 10, 10))]
    predictions = [
        Detection("person", 0.95, (20, 20, 30, 30)),
        Detection("person", 0.9, (0, 0, 10, 10)),
    ]
    result = evaluate_average_precision(predictions, truth)
    assert result.average_precision == pytest.approx(0.5)
    assert result.average_recall == 1.0


def test_mask_average_precision_uses_masks_and_validates_frames():
    truth_mask = np.ones((3, 3), dtype=np.uint8)
    truth = [Detection("car", 1.0, (0, 0, 3, 3), mask=truth_mask)]
    prediction = [Detection("car", 0.8, (0, 0, 3, 3), mask=truth_mask)]
    result = evaluate_mask_average_precision([prediction], [truth])
    assert result.average_precision == pytest.approx(1.0)
    with pytest.raises(ValueError, match="equal frame counts"):
        evaluate_average_precision([prediction], [])


def test_coco_loader_rasterizes_polygons_and_exports_predictions(tmp_path):
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": "frame.jpg", "width": 6, "height": 5},
                    {"id": 8, "file_name": "empty.jpg", "width": 6, "height": 5},
                ],
                "categories": [{"id": 1, "name": "car"}],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 7,
                        "category_id": 1,
                        "bbox": [1, 1, 3, 2],
                        "area": 6,
                        "segmentation": [[1, 1, 4, 1, 4, 3, 1, 3]],
                    },
                    {
                        "id": 11,
                        "image_id": 7,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "iscrowd": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = load_coco_dataset(annotation_path)
    assert isinstance(dataset, CocoDataset)
    assert isinstance(dataset.images[0], CocoImage)
    assert dataset.image_ids == (7, 8)
    assert len(dataset.frames()[0]) == 1
    assert dataset.frames()[0][0].mask.any()
    assert evaluate_mask_average_precision(
        dataset.frames(), dataset.frames()
    ).average_precision == pytest.approx(1.0)

    output_path = tmp_path / "predictions.json"
    count = write_coco_predictions(
        dataset.frames(),
        output_path,
        image_ids=dataset.image_ids,
        category_ids={"car": 1},
        include_masks=True,
    )
    assert count == 1
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert rows[0]["image_id"] == 7
    assert rows[0]["segmentation"]["size"] == [5, 6]
    loaded_predictions = load_coco_predictions(output_path, dataset, include_masks=True)
    assert evaluate_mean_mask_average_precision(
        loaded_predictions, dataset.frames()
    ).mean_average_precision == pytest.approx(1.0)


def test_coco_cli_evaluation_and_image_discovery(tmp_path, capsys):
    from yase.cli import main

    payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 4, "height": 4}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]],
            }
        ],
    }
    ground_truth = tmp_path / "gt.json"
    predictions = tmp_path / "predictions.json"
    ground_truth.write_text(json.dumps(payload), encoding="utf-8")
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 2, 2],
                    "score": 0.9,
                    "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert main(["evaluate-coco", str(predictions), str(ground_truth), "--masks"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["bbox"]["mean_average_precision"] == pytest.approx(1.0)
    assert result["mask"]["mean_average_precision"] == pytest.approx(1.0)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "b.png").touch()
    (image_dir / "a.jpg").touch()
    (image_dir / "skip.txt").touch()
    assert [item.name for item in discover_images([image_dir])] == ["a.jpg", "b.png"]


def test_mask_metrics_and_mot_roundtrip(tmp_path):
    mask = np.ones((3, 3), dtype=np.uint8)
    prediction = Detection("car", 0.9, (0, 0, 3, 3), mask=mask, track_id=2)
    metrics = evaluate_masks([prediction], [prediction])
    assert isinstance(metrics, MaskMetrics)
    assert metrics.f1 == 1.0
    path = tmp_path / "gt.txt"
    assert write_mot_sequence([[prediction]], path) == 1
    restored = load_mot_sequence(path)
    assert restored[0][0].track_id == 2
    assert restored[0][0].box.width == 3


def test_dwell_rule_emits_once_after_duration():
    rule = DwellRule("person", 2.0)
    detection = Detection("person", 0.9, (0, 0, 2, 2), track_id=5)
    result = SemanticResult(detections=[detection])
    assert rule.evaluate(result, 0.0) == []
    assert rule.evaluate(result, 1.0) == []
    event = rule.evaluate(result, 2.0)[0]
    assert event.metadata["type"] == "dwell"
    assert rule.evaluate(result, 3.0) == []


def test_benchmark_tracking_reports_quality_metrics():
    tracker = IoUTracker(iou_threshold=0.1)
    frame = [Detection("person", 0.9, (0, 0, 2, 2), track_id=None)]
    truth = [Detection("person", 1.0, (0, 0, 2, 2), track_id=1)]
    report = BenchmarkRunner().run_tracking(tracker, [frame], [[truth[0]]])
    assert report.quality["mota"] == 1.0
    assert report.quality["idf1"] == 1.0


def test_cli_evaluate_mot_reports_hota_and_tracking(tmp_path, capsys):
    from yase.cli import main

    truth = tmp_path / "truth.txt"
    prediction = tmp_path / "prediction.txt"
    truth.write_text("1,4,0,0,2,2,1,1,1,1\n", encoding="utf-8")
    prediction.write_text("1,4,0,0,2,2,0.9,1,1,1\n", encoding="utf-8")
    assert main(["evaluate-mot", str(prediction), str(truth)]) == 0
    output = capsys.readouterr().out
    assert '"hota"' in output
    assert '"idf1": 1.0' in output


def test_cli_artifact_inspection_and_verification(tmp_path, capsys):
    from yase.cli import main

    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    assert main(["artifact", str(path), "--sha256"]) == 0
    output = capsys.readouterr().out
    digest = json.loads(output)["sha256"]
    assert main(["artifact", str(path), "--verify", digest]) == 0
    assert json.loads(capsys.readouterr().out)["size_bytes"] == 5


def test_external_tracker_adapter_normalizes_mapping_output():
    class Tracker:
        def update(self, _detections):
            return [{"bbox": (0, 0, 2, 2), "track_id": 12, "label": "car"}]

    result = ExternalTrackerAdapter(Tracker()).update([])
    assert result[0].track_id == 12


def test_video_identity_store_receives_camera_id():
    capture = FakeCapture(1)
    detection = Detection("person", 0.9, (0, 0, 2, 2), attributes={"embedding": [1, 0]})
    stream = VideoStream(
        capture,
        lambda _image: SemanticResult(detections=[detection]),
        identity_store=GlobalIdentityStore(),
        camera_id="cam-1",
    )
    frame = list(stream)[0]
    assert frame.result.detections[0].attributes["camera_id"] == "cam-1"


def test_benchmark_compare_uses_same_inputs_for_each_backend():
    images = [np.zeros((2, 2, 3), dtype=np.uint8)] * 2
    reports = BenchmarkRunner().compare(
        {
            "a": lambda _image: {"tags": ["a"]},
            "b": lambda _image: {"tags": ["b"]},
        },
        images,
    )
    assert set(reports) == {"a", "b"}
    assert all(report.samples == 2 for report in reports.values())


def test_temperature_scaler_calibrates_and_can_be_used_by_consensus():
    scaler = TemperatureScaler().fit([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], steps=20)
    assert scaler.temperature > 0
    assert scaler.transform(0.9) > scaler.transform(0.2)
    detection = Detection("car", 0.8, (0, 0, 4, 4))
    result = MultimodalConsensus(calibrators={"model": scaler}).fuse(
        {"model": SemanticResult(detections=[detection])}
    )
    assert result.detections[0].score > 0.5


def test_tesseract_adapter_and_json_serialization():
    class FakeTesseract:
        def image_to_data(self, _image, **_kwargs):
            return {
                "text": ["hello", ""],
                "conf": ["90", "-1"],
                "left": [1, 0],
                "top": [2, 0],
                "width": [3, 0],
                "height": [4, 0],
            }

    result = TesseractExtractor(engine=FakeTesseract()).extract(
        np.zeros((8, 8, 3), dtype=np.uint8)
    )
    assert result.ocr[0].text == "hello"
    payload = result_to_dict(result)
    assert payload["ocr"][0]["box"]["x2"] == 4.0
    assert '"ocr"' in result_to_json(result)


def test_realtime_queue_size_is_fixed_to_latest_frame():
    with pytest.raises(ValueError, match="queue_size"):
        RealtimeVideoStream(FakeCapture(1), lambda image: image, queue_size=2)


class FakeOnnxIO:
    def __init__(self, name):
        self.name = name


class FakeOnnxSession:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def get_inputs(self):
        return [FakeOnnxIO("pixels")]

    def run(self, names, inputs):
        self.calls.append((names, inputs))
        return self.outputs


def test_onnx_injected_session_prepares_nchw_and_depth():
    session = FakeOnnxSession([np.ones((1, 1, 2, 3), dtype=np.float32)])
    backend = OnnxRuntimeExtractor("unused.onnx", session=session, size=(3, 2))
    result = backend.extract(np.zeros((4, 5, 3), dtype=np.uint8))
    assert result.depth.shape == (1, 2, 3)
    assert session.calls[0][0] is None
    feed = session.calls[0][1]["pixels"]
    assert feed.shape == (1, 3, 2, 3)
    assert feed.dtype == np.float32


def test_onnx_both_outputs_and_named_io():
    session = FakeOnnxSession([np.ones((1, 2, 2)), np.zeros((1, 2, 2))])
    backend = OnnxRuntimeExtractor(
        "unused.onnx",
        session=session,
        task="both",
        input_name="pixels",
        output_names=["depth", "mask"],
    )
    result = backend.extract(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result.depth.shape == (2, 2)
    assert result.segmentation.shape == (2, 2)
    assert session.calls[0][0] == ["depth", "mask"]


def test_onnx_validates_task_size_outputs_and_optional_dependency():
    with pytest.raises(ValueError, match="task"):
        OnnxRuntimeExtractor("x", session=FakeOnnxSession([]), task="bad")
    with pytest.raises(ValueError, match="size"):
        OnnxRuntimeExtractor("x", session=FakeOnnxSession([]), size=(0, 2))
    with pytest.raises(ValueError, match="two"):
        OnnxRuntimeExtractor(
            "x", session=FakeOnnxSession([np.zeros((1, 1))]), task="both"
        ).extract(np.zeros((2, 2, 3)))
    with pytest.raises(ImportError, match="onnx"):
        OnnxRuntimeExtractor("missing.onnx")


def test_onnx_supports_nhwc_and_provider_configuration():
    session = FakeOnnxSession([np.ones((1, 2, 2), dtype=np.float32)])
    backend = OnnxRuntimeExtractor(
        "unused.onnx",
        session=session,
        input_layout="NHWC",
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": "0"}],
    )
    backend.extract(np.zeros((2, 3, 3), dtype=np.uint8))
    assert session.calls[0][1]["pixels"].shape == (1, 2, 3, 3)
    with pytest.raises(ValueError, match="provider_options"):
        OnnxRuntimeExtractor(
            "unused.onnx",
            session=FakeOnnxSession([]),
            provider_options=[{"device_id": "0"}],
        )


def test_onnx_optional_io_binding_uses_bound_outputs():
    class Output:
        def __init__(self, name):
            self.name = name

    class Binding:
        def __init__(self):
            self.input = None
            self.outputs = []
            self.bound_outputs = []

        def bind_cpu_input(self, name, tensor):
            self.input = (name, tensor)

        def bind_output(self, name, device_type):
            self.bound_outputs.append((name, device_type))

        def copy_outputs_to_cpu(self):
            return self.outputs

    class Session:
        def __init__(self):
            self.binding = Binding()
            self.outputs = [np.ones((1, 2, 2), dtype=np.float32)]

        def get_inputs(self):
            return [FakeOnnxIO("pixels")]

        def get_outputs(self):
            return [Output("depth")]

        def io_binding(self):
            return self.binding

        def run_with_iobinding(self, binding):
            binding.outputs = self.outputs

        def get_providers(self):
            return ["CUDAExecutionProvider"]

    session = Session()
    result = OnnxRuntimeExtractor(
        "unused.onnx", session=session, use_io_binding=True
    ).extract(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result.depth.shape == (2, 2)
    assert session.binding.input[0] == "pixels"
    assert session.binding.bound_outputs == [("depth", "cpu")]
    assert result.metadata["io_binding"] is True
    with pytest.raises(RuntimeError, match="io_binding"):
        OnnxRuntimeExtractor(
            "unused.onnx",
            session=FakeOnnxSession([np.ones((1, 1, 1))]),
            use_io_binding=True,
        ).extract(np.zeros((1, 1, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="boolean"):
        OnnxRuntimeExtractor(
            "unused.onnx", session=FakeOnnxSession([]), use_io_binding=1
        )


def test_detection_normalisation_handles_columnar_rows_and_normalized_boxes():
    detections = normalise_detections(
        {
            "boxes": [[0.1, 0.2, 0.5, 0.8]],
            "scores": [0.9],
            "labels": [2],
        },
        image_shape=(100, 200),
        box_format="normalized_xyxy",
        label_map={2: "car"},
    )
    assert detections[0].label == "car"
    assert detections[0].box.as_xyxy() == pytest.approx((20, 20, 100, 80))
    rows = normalise_detections([[0, 0, 4, 4, 0.8, 1]], label_map={1: "person"})
    assert rows[0].label == "person"
    assert normalise_detections(rows, score_threshold=0.9) == []


def test_artifact_helpers_are_reproducible(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"yase-model")
    info = inspect_artifact(str(path), checksum=True)
    assert isinstance(info, ArtifactInfo)
    assert info.sha256 == sha256_file(str(path))
    assert verify_artifact(str(path), info.sha256).size_bytes == len(b"yase-model")
    with pytest.raises(ValueError, match="checksum"):
        verify_artifact(str(path), "0" * 64)


def test_composite_merges_fields_and_preserves_timestamp():
    composite = CompositeExtractor(
        {
            "depth": lambda image: SemanticResult(depth=np.ones((2, 2)), timestamp=3.0),
            "segmentation": lambda image: {
                "segmentation": np.zeros((2, 2)),
                "detections": [{"label": "car"}],
                "tags": ["street"],
                "embeddings": [1.0, 2.0],
                "metadata": "kept",
            },
        }
    )
    result = composite.extract(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result.depth.shape == (2, 2)
    assert result.segmentation.shape == (2, 2)
    assert result.detections == [{"label": "car"}]
    assert result.tags == ["street"]
    assert result.embeddings.shape == (2,)
    assert result.timestamp == 3.0
    assert result.metadata == {"metadata": "kept"}


def test_composite_extract_batch_uses_native_backend_batches_and_preserves_order():
    class BatchDepth:
        def __init__(self):
            self.calls = 0

        def extract_batch(self, images):
            self.calls += 1
            return [{"depth": image[..., 0]} for image in images]

    class Tags:
        def __call__(self, image):
            return {"tags": [int(image[0, 0, 0])]}

    depth = BatchDepth()
    composite = CompositeExtractor({"depth": depth, "tags": Tags()})
    images = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (2, 7, 4)]
    results = composite.extract_batch(images)
    assert depth.calls == 1
    assert [int(result.depth[0, 0]) for result in results] == [2, 7, 4]
    assert [result.tags for result in results] == [[2], [7], [4]]


def test_composite_extract_batch_contextualizes_size_errors():
    class Broken:
        def extract_batch(self, _images):
            return []

    with pytest.raises(BackendError, match="backend 'broken'"):
        CompositeExtractor({"broken": Broken()}).extract_batch(
            [np.zeros((1, 1, 3), dtype=np.uint8)]
        )


def test_composite_conflicts_support_explicit_policies():
    backends = [
        lambda image: SemanticResult(depth=np.ones((1, 1)), timestamp=1.0),
        lambda image: SemanticResult(depth=np.zeros((1, 1)), timestamp=2.0),
    ]
    with pytest.raises(ValueError, match="conflict"):
        CompositeExtractor(backends).extract(np.zeros((1, 1, 3)))
    first = CompositeExtractor(backends, conflict="first").extract(np.zeros((1, 1, 3)))
    last = CompositeExtractor(backends, conflict="last").extract(np.zeros((1, 1, 3)))
    assert first.depth[0, 0] == 1 and first.timestamp == 1.0
    assert last.depth[0, 0] == 0 and last.timestamp == 2.0


def test_composite_contextualises_backend_errors_and_validates_inputs():
    def fail(_image):
        raise OSError("offline")

    with pytest.raises(RuntimeError, match="backend 'camera'.*offline"):
        CompositeExtractor({"camera": fail}).extract(np.zeros((1, 1, 3)))
    with pytest.raises(ValueError, match="conflict"):
        CompositeExtractor([], conflict="bad")
    with pytest.raises(TypeError, match="mapping or sequence"):
        CompositeExtractor("not a backend")
    with pytest.raises(RuntimeError, match="raw backend output"):
        CompositeExtractor({"unknown": lambda image: np.zeros((1, 1))}).extract(
            np.zeros((1, 1, 3))
        )


class NativeBatchBackend:
    def __init__(self):
        self.calls = 0

    def extract_batch(self, images):
        self.calls += 1
        return [{"depth": image[..., 0]} for image in images]


def test_extract_many_uses_native_batch_and_preserves_order():
    backend = NativeBatchBackend()
    api = Yase(extractor=backend)
    images = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    results = api.extract_many(images, timestamps=[0.1, 0.2, 0.3])
    assert backend.calls == 1
    assert [result.depth[0, 0] for result in results] == [1, 2, 3]
    assert [result.timestamp for result in results] == [0.1, 0.2, 0.3]


def test_extract_many_batch_skip_handles_invalid_image_and_records_metrics():
    backend = NativeBatchBackend()
    metrics = RuntimeMetrics(namespace="batch_test")
    api = Yase(extractor=backend, metrics=metrics)
    images = [
        np.full((1, 1, 3), 1, dtype=np.uint8),
        np.zeros((1,), dtype=np.uint8),
        np.full((1, 1, 3), 3, dtype=np.uint8),
    ]

    results = api.extract_many(images, error_policy="skip")

    assert backend.calls == 1
    assert results[0] is not None and results[0].depth[0, 0] == 1
    assert results[1] is None
    assert results[2] is not None and results[2].depth[0, 0] == 3
    extraction = metrics.snapshot()["extraction"]
    assert extraction["attempts"] == 3
    assert extraction["failures"] == 1


def test_extract_many_batch_load_error_uses_error_callback_with_original_index():
    backend = NativeBatchBackend()
    replacement = SemanticResult(tags=["recovered"])
    seen = []
    api = Yase(extractor=backend)

    results = api.extract_many(
        [np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1,), dtype=np.uint8)],
        on_error=lambda error, index: seen.append((type(error), index)) or replacement,
    )

    assert backend.calls == 1
    assert results[1] is replacement
    assert seen == [(ValueError, 1)]


def test_extract_many_validates_batch_contract():
    api = Yase(extractor=NativeBatchBackend())
    with pytest.raises(ValueError, match="same length"):
        api.extract_many([np.zeros((1, 1, 3))], timestamps=[])
    with pytest.raises(ValueError, match="error_policy"):
        api.extract_many([], error_policy="ignore")


def test_extract_many_fallback_skip_and_error_callback():
    def sometimes_fails(image):
        if image[0, 0, 0] == 2:
            raise RuntimeError("bad image")
        return image[..., 0]

    api = Yase(extractor=sometimes_fails)
    images = [np.full((1, 1, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    skipped = api.extract_many(images, error_policy="skip")
    assert skipped[0].depth[0, 0] == 1
    assert skipped[1] is None
    assert skipped[2].depth[0, 0] == 3

    replacement = SemanticResult(tags=["invalid"])
    recovered = api.extract_many(images, on_error=lambda error, index: replacement)
    assert recovered[1] is replacement


def test_extract_many_can_parallelize_callable_backends_in_input_order():
    barrier = Barrier(2)

    class ParallelBackend:
        def __call__(self, image):
            barrier.wait(timeout=2)
            return image[..., 0]

    api = Yase(extractor=ParallelBackend())
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    results = api.extract_many([image, image], max_workers=2)
    assert [result.depth.shape for result in results] == [(2, 2), (2, 2)]
    with pytest.raises(ValueError, match="positive integer"):
        api.extract_many([image], max_workers=0)


def test_extract_many_rejects_wrong_native_batch_size():
    class BrokenBatch:
        def extract_batch(self, images):
            return []

    with pytest.raises(ValueError, match="one result"):
        Yase(extractor=BrokenBatch()).extract_many([np.zeros((1, 1, 3))])


class BatchOnnxSession(FakeOnnxSession):
    def run(self, names, inputs):
        self.calls.append((names, inputs))
        batch = inputs["pixels"].shape[0]
        return [np.arange(batch, dtype=np.float32)[:, None, None, None]]


def test_onnx_extract_batch_uses_one_call_and_preserves_order():
    session = BatchOnnxSession([])
    backend = OnnxRuntimeExtractor("unused.onnx", session=session)
    images = [np.full((2, 3, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    results = backend.extract_batch(images)
    assert len(session.calls) == 1
    assert session.calls[0][1]["pixels"].shape == (3, 3, 2, 3)
    assert [float(item.depth.flat[0]) for item in results] == [0.0, 1.0, 2.0]


def test_onnx_batch_rejects_mismatched_shapes_without_resize():
    backend = OnnxRuntimeExtractor("unused.onnx", session=FakeOnnxSession([]))
    images = [np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((3, 2, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="same shape"):
        backend.extract_batch(images)


def test_openvino_extractor_uses_injected_compiled_model():
    class Port:
        def get_any_name(self):
            return "pixels"

    class CompiledModel:
        inputs = [Port()]

        def __call__(self, inputs):
            tensor = next(iter(inputs.values()))
            return {"depth": np.zeros((tensor.shape[0], 2, 2), dtype=np.float32)}

    backend = OpenVINOExtractor(compiled_model=CompiledModel())
    results = backend.extract_batch(
        [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]
    )
    assert len(results) == 2
    assert results[0].depth.shape == (2, 2)
    assert results[0].metadata["backend"] == "openvino"


def test_openvino_extractor_uses_async_queue_and_restores_input_order():
    class Port:
        def get_any_name(self):
            return "pixels"

    class Request:
        def __init__(self, tensor):
            self.results = {
                "depth": tensor[:, 0, :, :].astype(np.float32),
            }

    class Queue:
        def __init__(self):
            self.callback = None
            self.started = []
            self.waited = False

        def set_callback(self, callback):
            self.callback = callback

        def start_async(self, inputs, userdata=None):
            self.started.append(userdata)
            tensor = next(iter(inputs.values()))
            self.callback(Request(tensor), userdata)

        def wait_all(self):
            self.waited = True

    class CompiledModel:
        inputs = [Port()]

    queue = Queue()
    backend = OpenVINOExtractor(
        compiled_model=CompiledModel(), async_queue=queue, async_jobs=3
    )
    results = backend.extract_batch_async(
        [
            np.full((2, 2, 3), 1, dtype=np.uint8),
            np.full((2, 2, 3), 2, dtype=np.uint8),
        ]
    )

    assert queue.started == [0, 1]
    assert queue.waited
    assert [float(result.depth[0, 0]) for result in results] == pytest.approx(
        [1 / 255, 2 / 255]
    )
    assert all(result.metadata["async_queue"] for result in results)


def test_tensorrt_extractor_accepts_custom_runner():
    class Runner:
        def infer(self, tensor):
            return {"depth": np.ones((tensor.shape[0], 2, 2), dtype=np.float32)}

    backend = TensorRTExtractor(runner=Runner())
    result = backend.extract(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result.depth.shape == (2, 2)
    assert result.metadata["backend"] == "tensorrt"


def test_tensorrt_context_pool_parallel_extraction_preserves_order_and_closes():
    class Runner:
        def __init__(self):
            self.closed = False

        def infer(self, tensor):
            return {"depth": tensor[:, 0, :, :]}

        def close(self):
            self.closed = True

    created = []

    def factory():
        runner = Runner()
        created.append(runner)
        return runner

    pool = TensorRTContextPool(factory, size=2)
    backend = TensorRTExtractor(runner_pool=pool)
    results = backend.extract_batch_parallel(
        [
            np.full((2, 2, 3), 1, dtype=np.uint8),
            np.full((2, 2, 3), 2, dtype=np.uint8),
            np.full((2, 2, 3), 3, dtype=np.uint8),
        ]
    )
    assert [float(result.depth[0, 0]) for result in results] == pytest.approx(
        [1 / 255, 2 / 255, 3 / 255]
    )
    backend.close()
    assert all(runner.closed for runner in created)
    with pytest.raises(RuntimeError, match="closed"):
        pool.infer(np.zeros((1, 3, 2, 2), dtype=np.float32))


def test_declarative_config_builds_facade_pipeline_and_json_file(tmp_path):
    registry = BackendRegistry()

    def factory(offset=0):
        return lambda image: {"tags": [int(image[0, 0, 0]) + offset]}

    registry.register("toy", factory)
    facade = build_from_config(
        {
            "backend": "toy",
            "options": {"offset": 4},
            "limits": {"max_pixels": 4},
        },
        registry=registry,
    )
    assert facade.extract(np.zeros((2, 2, 3), dtype=np.uint8)).tags == [4]

    pipeline = build_from_config(
        {
            "stages": [
                {"name": "first", "backend": "toy"},
                {"name": "disabled", "backend": "toy", "enabled": False},
            ],
            "record_timings": False,
        },
        registry=registry,
    )
    assert pipeline.extract(np.zeros((1, 1, 3), dtype=np.uint8)).tags == [0]

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"backend": "toy", "options": {"offset": 2}}),
        encoding="utf-8",
    )
    loaded = load_config(path, registry=registry)
    assert loaded.extract(np.zeros((1, 1, 3), dtype=np.uint8)).tags == [2]
    with pytest.raises(ValueError, match="unknown config"):
        build_from_config({"backend": "toy", "unexpected": True}, registry=registry)


def test_torchscript_batch_uses_one_model_call(monkeypatch):
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def permute(self, *order):
            return Tensor(self.value.transpose(order))

        def to(self, _device):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Model:
        def __init__(self):
            self.calls = 0

        def eval(self):
            return self

        def __call__(self, tensor):
            self.calls += 1
            return Tensor(np.zeros((tensor.value.shape[0], 1, 2, 2)))

    model = Model()
    from types import SimpleNamespace

    fake_cuda = SimpleNamespace(is_available=lambda: False)
    fake_jit = SimpleNamespace(load=lambda *args, **kwargs: model)
    fake_functional = SimpleNamespace(interpolate=lambda tensor, **kwargs: tensor)
    fake_nn = SimpleNamespace(functional=fake_functional)

    class FakeTorch:
        cuda = fake_cuda
        jit = fake_jit
        nn = fake_nn
        device = staticmethod(lambda name: name)
        from_numpy = staticmethod(Tensor)

        @staticmethod
        @contextmanager
        def inference_mode():
            yield

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    from yase.backends import TorchScriptExtractor

    backend = TorchScriptExtractor("local.pt", size=(2, 2))
    results = backend.extract_batch(
        [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        ]
    )
    assert len(results) == 2 and model.calls == 1
    assert results[0].depth.shape == (2, 2)


def test_observation_bundle_keeps_frame_provenance_and_uncertainty():
    digest = "a" * 64
    result = SemanticResult(depth=np.ones((2, 3)), timestamp=4.5)
    provenance = ModelProvenance(
        model_id="depth-model",
        revision="v1",
        artifact_sha256=digest,
        runtime="onnxruntime",
        device="cpu",
    )
    bundle = ObservationBundle.from_result(
        result,
        frame_id=7,
        source_id="camera-1",
        uncertainty={"depth": Uncertainty(0.82, calibrated=True, method="ece")},
        provenance=(provenance,),
        metadata={"tenant": "demo"},
        width=3,
        height=2,
    )
    payload = bundle.to_dict()
    assert payload["schema_version"] == "1.0"
    assert payload["frame"]["source_id"] == "camera-1"
    assert payload["provenance"][0]["artifact_sha256"] == digest
    assert payload["uncertainty"]["depth"]["calibrated"] is True
    assert payload["result"]["depth"] == {"dtype": "float64", "shape": [2, 3]}


def test_observation_contract_validation_and_embedding_metadata():
    with pytest.raises(ValueError, match="finite"):
        FrameRef(frame_id=0, timestamp=float("nan"))
    with pytest.raises(ValueError, match="hex digest"):
        ModelProvenance(model_id="model", artifact_sha256="bad")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Uncertainty(confidence=1.1)

    embedding = EmbeddingRecord(
        vector=np.array([3.0, 4.0]) / 5.0,
        space="clip-v1",
        model_id="siglip",
        normalized=True,
    )
    assert embedding.to_dict()["shape"] == [2]
    assert embedding.to_dict(include_values=True)["vector"] == [0.6, 0.8]


def test_stage_specs_validate_ordered_dependencies_and_pipeline_compatibility():
    detector = StageSpec(
        "detector", provides=("detections",), capabilities=("open-vocabulary",)
    )
    caption = StageSpec("caption", requires=("detections",), provides=("caption",))
    assert "caption" in validate_stage_specs((detector, caption))
    with pytest.raises(ValueError, match="unavailable"):
        validate_stage_specs((caption, detector))
    with pytest.raises(ValueError, match="duplicate"):
        validate_stage_specs((detector, detector))

    context = StageContext(metadata={"request_id": "abc"})
    assert context.metadata["request_id"] == "abc"
    assert not context.cancelled
    context.cancel_event = Event()
    context.cancel_event.set()
    assert context.cancelled
    pipeline = SemanticPipeline(
        [
            PipelineStage(
                "detector",
                lambda image: {"detections": [1]},
                spec=detector,
            ),
            PipelineStage(
                "caption",
                lambda image: {"caption": "scene"},
                spec=caption,
            ),
        ]
    )
    assert "caption" in pipeline.validate()

    class ContractedStage:
        spec = detector

        def run(self, image, context):
            assert image.shape == (2, 2, 3)
            context.metadata["called"] = True
            return {"detections": [{"score": 1.0}]}

    contracted = SemanticPipeline(
        [PipelineStage("detector", ContractedStage(), spec=detector)]
    )
    assert contracted.extract(np.zeros((2, 2, 3), dtype=np.uint8)).detections


def test_yase_extract_bundle_infers_dimensions():
    api = Yase(extractor=lambda image: {"depth": image[..., 0]})
    bundle = api.extract_bundle(
        np.zeros((4, 5, 3), dtype=np.uint8), frame_id=3, source_id="stream"
    )
    assert bundle.frame == FrameRef(
        frame_id=3,
        source_id="stream",
        width=5,
        height=4,
    )
    assert bundle.result.depth.shape == (4, 5)


def test_semantic_result_serialization_is_versioned_and_reconstructable():
    original = SemanticResult(
        depth=np.ones((2, 2), dtype=np.float32),
        tags=["scene"],
        timestamp=3.5,
        metadata={"source": "unit"},
    )
    payload = result_to_dict(original, include_arrays=True)
    assert payload["schema_version"] == "1.0"
    restored = result_from_dict(payload)
    assert np.array_equal(restored.depth, original.depth)
    assert restored.tags == original.tags
    assert restored.metadata == original.metadata

    legacy = dict(payload)
    legacy.pop("schema_version")
    assert result_from_dict(legacy).timestamp == 3.5
    summarized = result_to_dict(original)
    with pytest.raises(ValueError, match="shape/dtype"):
        result_from_dict(summarized)
    with pytest.raises(ValueError, match="unsupported result schema"):
        result_from_dict({"schema_version": "2.0"})


def test_observation_json_and_jsonl_are_stable_and_array_safe():
    bundle = ObservationBundle.from_result(
        SemanticResult(depth=np.ones((1, 2), dtype=np.float32)),
        frame_id=1,
        metadata={"score": np.float32(0.5)},
    )
    payload = json.loads(observation_to_json(bundle))
    assert payload["result"]["depth"] == {"dtype": "float32", "shape": [1, 2]}
    assert payload["metadata"]["score"] == 0.5
    output = StringIO()
    assert write_observation_jsonl([bundle, bundle], output) == 2
    assert len(output.getvalue().splitlines()) == 2


def test_observation_sinks_support_archive_callback_fanout_and_bounded_queue():
    bundle = ObservationBundle.from_result(SemanticResult(caption="scene"))
    memory = MemorySink(max_items=1)
    assert memory.emit(bundle)
    assert not memory.emit(bundle)
    callback_values = []
    callback = CallbackSink(callback_values.append)
    stream = StringIO()
    jsonl = JsonlObservationSink(stream, flush_each=True)
    fanout = FanoutSink((callback, jsonl))
    assert fanout.emit(bundle)
    jsonl.close()
    assert callback_values == [bundle]
    assert len(stream.getvalue().splitlines()) == 1

    queue_sink = QueueSink(maxsize=1, on_full="drop")
    assert queue_sink.emit(bundle)
    assert not queue_sink.emit(bundle)
    assert queue_sink.dropped == 1
    assert queue_sink.get(timeout=0.1) == bundle
    queue_sink.close()
    with pytest.raises(RuntimeError, match="closed"):
        queue_sink.emit(bundle)


def test_rich_vision_contracts_validate_and_survive_composition():
    pose = Pose(
        keypoints=(
            Keypoint("nose", 10.0, 12.0, score=0.95, visible=True),
            Keypoint("left_eye", 9.0, 11.0, score=0.8),
        ),
        score=0.9,
        skeleton="coco17",
    )
    obb = OrientedBoundingBox(10.0, 12.0, 8.0, 6.0, 0.25)
    depth = DepthMap(np.ones((2, 3), dtype=np.float32), unit="m", scale=1.0)
    relation = Relation(1, "next_to", 2, score=0.7, evidence=("detector",))
    assert obb.as_cxcywh_angle()[-1] == 0.25
    assert depth.values.shape == (2, 3)
    result = CompositeExtractor(
        [
            lambda image: {
                "keypoints": [pose],
                "relations": [relation],
                "document": {"pages": 1},
            }
        ]
    ).extract(np.zeros((2, 3, 3), dtype=np.uint8))
    assert result.keypoints == [pose]
    assert result.relations == [relation]
    assert result.document == {"pages": 1}
    depth_result = Yase(
        extractor=lambda image: {
            "depth_map": DepthMap(np.ones((2, 3), dtype=np.float32), unit="m")
        }
    ).extract(np.zeros((2, 3, 3), dtype=np.uint8))
    assert depth_result.depth_map.unit == "m"
    with pytest.raises(ValueError, match="2-D"):
        DepthMap(np.zeros((1, 1, 1)))


def test_runtime_diagnostics_are_lazy_and_machine_readable():
    runtime = collect_runtime_info(
        optional_packages=("numpy", "package_that_is_missing")
    )
    assert isinstance(runtime, RuntimeInfo)
    assert runtime.optional_packages["numpy"] is True
    assert runtime.optional_packages["package_that_is_missing"] is False

    healthy = health_check(lambda image: image)
    assert isinstance(healthy, HealthReport)
    assert healthy.ready
    degraded = health_check(object(), required_packages=("package_that_is_missing",))
    assert degraded.status == "degraded"
    assert degraded.ready is False
    assert degraded.to_dict()["checks"]["extractor_interface"] is False


def test_cli_diagnostics_reports_readiness(capsys):
    from yase.cli import main

    assert main(["diagnostics"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert "runtime" in payload


def test_cli_processing_commands_parse_input_limits():
    from yase.cli import _input_limits, _parser

    args = _parser().parse_args(
        [
            "extract",
            "frame.jpg",
            "--model",
            "onnx",
            "--model-path",
            "model.onnx",
            "--max-pixels",
            "10000",
            "--max-width",
            "640",
            "--max-bytes",
            "1000000",
            "--max-workers",
            "3",
        ]
    )
    limits = _input_limits(args)
    assert limits is not None
    assert limits.max_pixels == 10000
    assert limits.max_width == 640
    assert limits.max_bytes == 1000000
    assert args.max_workers == 3
    modern = _parser().parse_args(
        [
            "extract",
            "frame.jpg",
            "--model",
            "vlm",
            "--model-path",
            "Qwen/Qwen3-VL-2B-Instruct",
            "--prompt",
            "describe the scene",
        ]
    )
    assert modern.model == "vlm"
    assert modern.prompt == "describe the scene"
    onnx = _parser().parse_args(
        [
            "extract",
            "frame.jpg",
            "--model",
            "onnx",
            "--model-path",
            "model.onnx",
            "--io-binding",
        ]
    )
    assert onnx.io_binding is True
    video = _parser().parse_args(
        [
            "video",
            "input.mp4",
            "--model",
            "onnx",
            "--model-path",
            "model.onnx",
            "--batch-size",
            "8",
        ]
    )
    assert video.batch_size == 8
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "video",
                "input.mp4",
                "--model",
                "onnx",
                "--model-path",
                "model.onnx",
                "--max-pixels",
                "0",
            ]
        )


def test_scheduler_reorders_dag_and_reuses_bounded_stage_cache():
    calls = []

    class Detector:
        def run(self, image, context):
            calls.append(("detector", context.metadata["stage"]))
            return {"detections": [{"score": float(image.mean())}]}

    class Caption:
        def run(self, image, context):
            calls.append(("caption", context.metadata["inputs"]["detections"]))
            return {"caption": "one object"}

    detector_spec = StageSpec("detector", provides=("detections",))
    caption_spec = StageSpec("caption", requires=("detections",), provides=("caption",))
    scheduler = ObservationScheduler(
        [
            PipelineStage("caption", Caption(), spec=caption_spec),
            PipelineStage("detector", Detector(), spec=detector_spec),
        ],
        config=SchedulerConfig(cache_size=2),
    )
    assert [stage.name for stage in scheduler.plan] == ["detector", "caption"]
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    first = scheduler.run(image)
    second = scheduler.run(image)
    assert first.fields["caption"] == "one object"
    assert [item.status for item in second.stages] == ["cached", "cached"]
    assert len(calls) == 2
    assert second.cache_hits == 2
    assert scheduler.cache_info() == {"size": 2, "capacity": 2}
    batch = scheduler.run_many([image, np.ones((2, 2, 3), dtype=np.uint8)])
    assert len(batch) == 2
    assert batch[0].as_result().detections[0]["score"] == 0.0
    with pytest.raises(ValueError, match="same length"):
        scheduler.run_many([image], frames=[])


def test_scheduler_supports_async_execution_and_callbacks():
    events = []
    spec = StageSpec("tags", provides=("tags",))
    scheduler = ObservationScheduler(
        [PipelineStage("tags", lambda image: {"tags": ["outdoor"]}, spec=spec)],
        config=SchedulerConfig(cache_size=0),
        on_stage=events.append,
    )
    report = asyncio.run(scheduler.arun(np.zeros((1, 1, 3), dtype=np.uint8)))
    assert report.fields["tags"] == ["outdoor"]
    assert isinstance(events[0], StageExecution)
    assert events[0].status == "completed"


def test_scheduler_runs_independent_stages_in_parallel_with_stable_report_order():
    barrier = Barrier(2)

    class ParallelBackend:
        def __init__(self, name, field):
            self.name = name
            self.field = field

        def run(self, image, context):
            barrier.wait(timeout=2)
            return {self.field: context.metadata["stage"]}

    scheduler = ObservationScheduler(
        [
            PipelineStage(
                "caption",
                ParallelBackend("caption", "caption"),
                spec=StageSpec("caption", provides=("caption",)),
            ),
            PipelineStage(
                "tags",
                ParallelBackend("tags", "tags"),
                spec=StageSpec("tags", provides=("tags",)),
            ),
        ],
        config=SchedulerConfig(cache_size=0, max_workers=2),
    )
    report = scheduler.run(np.zeros((1, 1, 3), dtype=np.uint8))
    assert report.fields["caption"] == "caption"
    assert report.fields["tags"] == "tags"
    assert [stage.name for stage in report.stages] == ["caption", "tags"]
    with pytest.raises(ValueError, match="positive integer"):
        SchedulerConfig(max_workers=0)


def test_scheduler_metrics_are_thread_safe_and_prometheus_compatible():
    metrics = RuntimeMetrics(namespace="demo")
    scheduler = ObservationScheduler(
        [
            PipelineStage(
                "tags",
                lambda image: {"tags": ["outdoor"]},
                spec=StageSpec("tags", provides=("tags",)),
            )
        ],
        config=SchedulerConfig(cache_size=0),
        metrics=metrics,
    )
    scheduler.run(np.zeros((1, 1, 3), dtype=np.uint8))
    snapshot = metrics.snapshot()
    exposition = metrics.prometheus_text()
    assert snapshot["runs"] == 1
    assert snapshot["stage_counts"] == {"tags:completed": 1}
    assert (
        'demo_scheduler_stage_executions_total{stage="tags",status="completed"} 1'
        in exposition
    )
    assert "demo_scheduler_elapsed_seconds_count 1" in exposition
    with pytest.raises(ValueError, match="namespace"):
        RuntimeMetrics(namespace="not-valid")
    with pytest.raises(ValueError, match="namespace"):
        RuntimeMetrics(namespace="123metrics")


def test_scheduler_handles_optional_failures_and_cancellation():
    failing = StageSpec("optional", provides=("caption",), optional=True)
    scheduler = ObservationScheduler(
        [
            PipelineStage(
                "optional",
                lambda image: (_ for _ in ()).throw(RuntimeError("offline")),
                spec=failing,
            )
        ],
        config=SchedulerConfig(on_error="skip", cache_size=0),
    )
    report = scheduler.run(np.zeros((1, 1, 3), dtype=np.uint8))
    assert report.stages[0].status == "failed"
    cancel = Event()
    cancel.set()
    with pytest.raises(StageCancelled, match="cancelled"):
        scheduler.run(np.zeros((1, 1, 3), dtype=np.uint8), cancel_event=cancel)


def test_scheduler_rejects_unresolvable_graph_and_bad_stage_name():
    missing = StageSpec("caption", requires=("detections",), provides=("caption",))
    with pytest.raises(SchedulerError, match="dependencies"):
        ObservationScheduler([PipelineStage("caption", lambda image: {}, spec=missing)])
    wrong = StageSpec("right", provides=("tags",))
    with pytest.raises(SchedulerError, match="does not match"):
        ObservationScheduler([PipelineStage("wrong", lambda image: {}, spec=wrong)])


def test_semantic_pipeline_can_promote_contracts_to_sync_async_and_bundle_apis():
    detector = StageSpec("detector", provides=("detections",))
    caption = StageSpec("caption", requires=("detections",), provides=("caption",))
    pipeline = SemanticPipeline(
        [
            PipelineStage("caption", lambda image: {"caption": "scene"}, spec=caption),
            PipelineStage(
                "detector", lambda image: {"detections": ["object"]}, spec=detector
            ),
        ]
    )
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    result = pipeline.extract_scheduled(image, timestamp=2.0)
    assert result.caption == "scene"
    assert result.metadata["scheduler"]["stages"][0]["name"] == "detector"
    async_result = asyncio.run(pipeline.extract_scheduled_async(image))
    assert async_result.caption == "scene"
    bundle = pipeline.extract_bundle_scheduled(image, frame_id=9, source_id="cam")
    assert bundle.frame.frame_id == 9
    assert bundle.result.caption == "scene"
