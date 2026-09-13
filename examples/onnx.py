"""Run a local ONNX Runtime model (install the onnx extra first)."""

import argparse

from yase import OnnxRuntimeExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("--task", choices=("depth", "segmentation"), default="depth")
    args = parser.parse_args()
    backend = OnnxRuntimeExtractor(args.model, task=args.task)
    result = backend.extract(args.image)
    output = result.depth if args.task == "depth" else result.segmentation
    print(output.shape, output.dtype)


if __name__ == "__main__":
    main()
