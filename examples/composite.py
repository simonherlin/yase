"""Compose independent depth and segmentation backends."""

import numpy as np

from yase import CallableExtractor, CompositeExtractor


def depth(image):
    return image.mean(axis=2, dtype=np.float32)


def segmentation(image):
    return (image[..., 0] > 128).astype(np.uint8)


def main():
    pipeline = CompositeExtractor(
        {
            "depth": CallableExtractor(depth),
            "segmentation": CallableExtractor(segmentation, task="segmentation"),
        }
    )
    result = pipeline.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    print(result.depth.shape, result.segmentation.shape)


if __name__ == "__main__":
    main()
