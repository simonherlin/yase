import numpy as np
import pytest

from yase import CallableExtractor, SemanticResult, VideoStats, Yase, load_image
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
