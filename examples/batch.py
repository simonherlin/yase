"""Extract an ordered image batch with a lightweight custom backend."""

import argparse

import numpy as np

from yase import CallableExtractor, Yase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    backend = CallableExtractor(
        lambda image: image.mean(axis=2, dtype=np.float32), task="depth"
    )
    results = Yase(extractor=backend).extract_many(args.images)
    for path, result in zip(args.images, results):
        print(path, result.depth.shape if result is not None else "failed")


if __name__ == "__main__":
    main()
