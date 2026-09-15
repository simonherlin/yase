"""Input safety limits for untrusted image and video workloads."""

from dataclasses import dataclass

import numpy as np

from .errors import InputError


@dataclass(frozen=True)
class InputLimits:
    """Optional resource limits applied after image decoding."""

    max_pixels: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    max_channels: int | None = None
    max_bytes: int | None = None

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
        self.validate_shape(
            width=width,
            height=height,
            channels=channels,
            byte_count=array.nbytes,
        )
        return array

    def validate_shape(
        self,
        *,
        width: int,
        height: int,
        channels: int | None = None,
        byte_count: int | None = None,
    ) -> None:
        """Validate image metadata before allocating a decoded array.

        ``load_image`` uses this method with dimensions read from an image
        header, which prevents a decompression bomb from reaching the pixel
        conversion step. ``byte_count`` is optional because encoded file size
        and decoded array size are different resources.
        """
        if width <= 0 or height <= 0:
            raise InputError("image dimensions must be positive")
        if channels is not None and channels <= 0:
            raise InputError("image channels must be positive")
        checks = (
            (self.max_pixels, height * width, "pixel limit"),
            (self.max_width, width, "width limit"),
            (self.max_height, height, "height limit"),
            (self.max_channels, channels, "channel limit")
            if channels is not None
            else None,
            (self.max_bytes, byte_count, "byte limit")
            if byte_count is not None
            else None,
        )
        for check in checks:
            if check is None:
                continue
            limit, actual, label = check
            if limit is not None and actual is not None and actual > limit:
                raise InputError(f"image exceeds {label}: {actual} > {limit}")


__all__ = ["InputLimits"]
