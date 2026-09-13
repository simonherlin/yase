"""Yase: semantic extraction for images and real-time video."""

from .backends import CallableExtractor, OnnxRuntimeExtractor, TorchScriptExtractor
from .core import Extractor, ImageInput, SemanticResult, Yase, load_image
from .video import (
    FrameResult,
    RealtimeVideoStream,
    VideoStats,
    VideoStream,
    process_video,
)

__version__ = "0.2.0"

__all__ = [
    "CallableExtractor",
    "Extractor",
    "OnnxRuntimeExtractor",
    "FrameResult",
    "ImageInput",
    "SemanticResult",
    "TorchScriptExtractor",
    "RealtimeVideoStream",
    "VideoStats",
    "VideoStream",
    "Yase",
    "load_image",
    "process_video",
]
