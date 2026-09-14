import numpy as np
import pytest

from yase import (
    CallableExtractor,
    CompositeExtractor,
    OnnxRuntimeExtractor,
    RealtimeVideoStream,
    SemanticResult,
    VideoStats,
    Yase,
    load_image,
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
    assert capture.released


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


def test_extract_many_rejects_wrong_native_batch_size():
    class BrokenBatch:
        def extract_batch(self, images):
            return []

    with pytest.raises(ValueError, match="one result"):
        Yase(extractor=BrokenBatch()).extract_many([np.zeros((1, 1, 3))])
