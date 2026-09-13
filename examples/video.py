"""Process a webcam or video file with a custom extractor."""

import argparse

import numpy as np

from yase import CallableExtractor, VideoStream, Yase


def mean_depth(image: np.ndarray) -> np.ndarray:
    return image.mean(axis=2, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default=0)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    extractor = Yase(extractor=CallableExtractor(mean_depth))
    for item in VideoStream(source, extractor, stride=args.stride, max_fps=15):
        print(item.frame_index, f"{item.processing_time * 1000:.1f} ms")


if __name__ == "__main__":
    main()
