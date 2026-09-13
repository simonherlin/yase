"""Run a deterministic image extraction backend.

Replace mean_depth with a model-specific function in production.
"""

import argparse

import numpy as np

from yase import CallableExtractor, Yase


def mean_depth(image: np.ndarray) -> np.ndarray:
    return image.mean(axis=2, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    extractor = Yase(extractor=CallableExtractor(mean_depth))
    result = extractor.extract(args.image)
    print("depth:", result.depth.shape, result.depth.dtype)


if __name__ == "__main__":
    main()
