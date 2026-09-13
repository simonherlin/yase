"""Backward-compatible import location for the Yase API."""

from .core import Extractor, ImageInput, SemanticResult, Yase, load_image

__all__ = ["Extractor", "ImageInput", "SemanticResult", "Yase", "load_image"]
