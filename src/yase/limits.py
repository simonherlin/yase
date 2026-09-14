"""Input safety limits for untrusted image and video workloads."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .errors import InputError


@dataclass(frozen=True)
class InputLimits:
    """Optional resource limits applied after image decoding."""

    max_pixels: Optional[int] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_channels: Optional[int] = None
    max_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "max_pixels",
            "max_width",
            "max_height",
            "max_channels",
            "max_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be positive when provided")

    def validate(self, image: np.ndarray) -> np.ndarray:
        """Validate and return the input array, raising ``InputError``."""
        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise InputError("image must have 2 or 3 dimensions")
        height, width = array.shape[:2]
        channels = 1 if array.ndim == 2 else array.shape[2]
        checks = (
            (self.max_pixels, height * width, "pixel limit"),
            (self.max_width, width, "width limit"),
            (self.max_height, height, "height limit"),
            (self.max_channels, channels, "channel limit"),
            (self.max_bytes, array.nbytes, "byte limit"),
        )
        for limit, actual, label in checks:
            if limit is not None and actual > limit:
                raise InputError(f"image exceeds {label}: {actual} > {limit}")
        return array


__all__ = ["InputLimits"]
