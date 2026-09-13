"""Yase: semantic extraction for images and real-time video."""

from .backends import CallableExtractor, TorchScriptExtractor
from .core import Extractor, ImageInput, SemanticResult, Yase, load_image
from .video import FrameResult, VideoStats, VideoStream, process_video

__version__ = "0.2.0"

__all__ = [
    "CallableExtractor",
    "Extractor",
    "FrameResult",
    "ImageInput",
    "SemanticResult",
    "TorchScriptExtractor",
    "VideoStats",
    "VideoStream",
    "Yase",
    "load_image",
    "process_video",
]
