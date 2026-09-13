"""Synthetic benchmark for sequential and latest-frame video modes."""

import argparse
import time

import numpy as np

from yase import RealtimeVideoStream, VideoStream


class SyntheticCapture:
    def __init__(self, count):
        self.frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(count)]
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, _property):
        return 30.0

    def isOpened(self):
        return True

    def release(self):
        self.released = True


def run(stream):
    started = time.perf_counter()
    count = sum(1 for _ in stream)
    elapsed = time.perf_counter() - started
    stats = stream.stats
    return count, elapsed, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--delay-ms", type=float, default=2.0)
    args = parser.parse_args()

    def backend(image):
        time.sleep(args.delay_ms / 1000.0)
        return image[..., 0]

    for name, stream in (
        ("sequential", VideoStream(SyntheticCapture(args.frames), backend)),
        ("latest-frame", RealtimeVideoStream(SyntheticCapture(args.frames), backend)),
    ):
        count, elapsed, stats = run(stream)
        print(
            name,
            "processed=",
            count,
            "elapsed_s=",
            round(elapsed, 3),
            "output_fps=",
            round(stats.output_fps, 1),
            "dropped=",
            stats.frames_dropped,
            "mean_latency_ms=",
            round(stats.mean_latency * 1000, 2),
        )


if __name__ == "__main__":
    main()
