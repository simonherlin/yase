"""Command-line entry point for local extraction workflows."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import (
    BenchmarkRunner,
    InputLimits,
    __version__,
    default_model_catalog,
    default_registry,
    discover_images,
    evaluate_hota,
    evaluate_hota_curve,
    evaluate_mean_average_precision,
    evaluate_mean_mask_average_precision,
    evaluate_tracking,
    health_check,
    inspect_artifact,
    load_coco_dataset,
    load_coco_predictions,
    load_mot_sequence,
    verify_artifact,
)
from .core import Yase
from .serialization import result_to_dict, write_jsonl
from .video import RealtimeVideoStream, VideoStream

_CLI_MODELS = (
    "torchscript",
    "onnx",
    "openvino",
    "tensorrt",
    "rf-detr",
    "sam3",
    "grounding-dino",
    "image-embedding",
    "vlm",
    "tesseract",
    "paddleocr",
)
_LOCAL_ARTIFACT_MODELS = {"torchscript", "onnx", "openvino", "tensorrt"}
_MODEL_ID_MODELS = {"sam3", "grounding-dino", "image-embedding", "vlm"}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _add_input_limits(parser: argparse.ArgumentParser) -> None:
    """Add consistent resource-limit flags to image-processing commands."""
    parser.add_argument("--max-pixels", type=_positive_int)
    parser.add_argument("--max-width", type=_positive_int)
    parser.add_argument("--max-height", type=_positive_int)
    parser.add_argument("--max-channels", type=_positive_int)
    parser.add_argument("--max-bytes", type=_positive_int)


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=_CLI_MODELS, required=True)
    parser.add_argument(
        "--model-path",
        help="local artifact for tensor runtimes, or local HF model ID/path",
    )
    parser.add_argument("--prompt", help="text prompt for SAM3 or a VLM")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="concept labels for Grounding DINO (space-separated)",
    )
    parser.add_argument(
        "--io-binding",
        action="store_true",
        help="use ONNX Runtime I/O binding for accelerator inference",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="PROVIDER",
        help="ONNX Runtime provider (repeatable, e.g. CUDAExecutionProvider)",
    )
    parser.add_argument(
        "--strict-providers",
        action="store_true",
        help="fail if requested ONNX providers are not active instead of falling back",
    )
    parser.add_argument(
        "--device",
        help="runtime device for TorchScript/OpenVINO/TensorRT (e.g. CPU, GPU, cuda)",
    )


def _input_limits(args: argparse.Namespace) -> InputLimits | None:
    values = {
        "max_pixels": args.max_pixels,
        "max_width": args.max_width,
        "max_height": args.max_height,
        "max_channels": args.max_channels,
        "max_bytes": args.max_bytes,
    }
    if not any(value is not None for value in values.values()):
        return None
    return InputLimits(**values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yase", description="Semantic extraction for images and video."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show version and built-in backends")
    info.set_defaults(handler=_info)

    diagnostics = subparsers.add_parser(
        "diagnostics", help="show runtime capabilities and readiness"
    )
    diagnostics.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="PACKAGE",
        help="optional package that must be installed (repeatable)",
    )
    diagnostics.add_argument(
        "--providers",
        action="store_true",
        help="probe installed ONNX/OpenVINO/Torch/TensorRT providers",
    )
    diagnostics.set_defaults(handler=_diagnostics)

    artifact = subparsers.add_parser(
        "artifact", help="inspect or verify a local model artifact"
    )
    artifact.add_argument("path", type=Path)
    artifact.add_argument("--sha256", action="store_true", help="compute SHA-256")
    artifact.add_argument("--verify", metavar="DIGEST", help="verify SHA-256")
    artifact.set_defaults(handler=_artifact)

    models = subparsers.add_parser("models", help="list model cards and licenses")
    models.add_argument("--task")
    models.set_defaults(handler=_models)

    evaluate = subparsers.add_parser(
        "evaluate-mot", help="evaluate MOTChallenge-style prediction files"
    )
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("ground_truth", type=Path)
    evaluate.add_argument("--label", default="object")
    evaluate.add_argument("--iou", type=float, default=0.5)
    evaluate.set_defaults(handler=_evaluate_mot)

    evaluate_coco = subparsers.add_parser(
        "evaluate-coco", help="evaluate COCO result JSON against annotations"
    )
    evaluate_coco.add_argument("predictions", type=Path)
    evaluate_coco.add_argument("ground_truth", type=Path)
    evaluate_coco.add_argument("--masks", action="store_true")
    evaluate_coco.add_argument("--no-class-aware", action="store_true")
    evaluate_coco.set_defaults(handler=_evaluate_coco)

    extract = subparsers.add_parser("extract", help="extract semantics from images")
    extract.add_argument("images", nargs="+", type=Path)
    _add_model_options(extract)
    extract.add_argument(
        "--task", choices=("depth", "segmentation", "both"), default="depth"
    )
    extract.add_argument("--output", type=Path)
    extract.add_argument("--include-arrays", action="store_true")
    extract.add_argument("--max-workers", type=_positive_int, default=1)
    _add_input_limits(extract)
    extract.set_defaults(handler=_extract)

    video = subparsers.add_parser("video", help="extract semantics from a video")
    video.add_argument("source")
    _add_model_options(video)
    video.add_argument(
        "--task", choices=("depth", "segmentation", "both"), default="depth"
    )
    video.add_argument("--stride", type=int, default=1)
    video.add_argument("--max-fps", type=float)
    video.add_argument("--max-frames", type=int)
    video.add_argument("--batch-size", type=_positive_int, default=1)
    video.add_argument("--realtime", action="store_true")
    video.add_argument("--output", type=Path)
    video.add_argument("--include-arrays", action="store_true")
    _add_input_limits(video)
    video.set_defaults(handler=_video)

    benchmark = subparsers.add_parser(
        "benchmark", help="measure latency of a local model on images"
    )
    benchmark.add_argument("images", nargs="+", type=Path)
    _add_model_options(benchmark)
    benchmark.add_argument(
        "--task", choices=("depth", "segmentation", "both"), default="depth"
    )
    benchmark.add_argument("--warmup", type=int, default=0)
    benchmark.add_argument("--output", type=Path)
    _add_input_limits(benchmark)
    benchmark.set_defaults(handler=_benchmark)
    return parser


def _info(_args: argparse.Namespace) -> int:
    print(json.dumps({"version": __version__, "backends": default_registry().names()}))
    return 0


def _diagnostics(args: argparse.Namespace) -> int:
    report = health_check(
        required_packages=tuple(args.require), probe_providers=args.providers
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if report.ready else 1


def _artifact(args: argparse.Namespace) -> int:
    if args.verify:
        info = verify_artifact(str(args.path), args.verify)
    else:
        info = inspect_artifact(str(args.path), checksum=args.sha256)
    print(json.dumps(info.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def _models(args: argparse.Namespace) -> int:
    cards = default_model_catalog().search(task=args.task)
    print(json.dumps([card.to_dict() for card in cards], ensure_ascii=False))
    return 0


def _make_extractor(args: argparse.Namespace) -> Yase:
    """Build a CLI extractor while keeping optional dependencies lazy."""
    options = {"input_limits": _input_limits(args)}
    if args.provider or args.strict_providers:
        if args.model != "onnx":
            raise ValueError("--provider/--strict-providers require --model onnx")
        if args.strict_providers and not args.provider:
            raise ValueError("--strict-providers requires at least one --provider")
        options["providers"] = args.provider
        options["strict_providers"] = args.strict_providers
    if args.device:
        if args.model not in {"torchscript", "openvino", "tensorrt"}:
            raise ValueError(
                "--device is supported with --model torchscript, openvino, or tensorrt"
            )
        options["device"] = args.device
    if args.model in _LOCAL_ARTIFACT_MODELS:
        if not args.model_path:
            raise ValueError(f"--model-path is required for {args.model}")
        options["model_path"] = args.model_path
        if args.model == "onnx" and args.io_binding:
            options["use_io_binding"] = True
        elif args.io_binding:
            raise ValueError("--io-binding is only supported with --model onnx")
    elif args.model in _MODEL_ID_MODELS:
        if not args.model_path:
            raise ValueError(f"--model-path is required for {args.model}")
        options["model_id"] = args.model_path
        if args.model == "grounding-dino":
            if not args.labels:
                raise ValueError("--labels is required for grounding-dino")
            options["labels"] = args.labels
        elif args.model == "sam3" and args.prompt:
            options["prompt"] = args.prompt
        elif args.model == "vlm" and args.prompt:
            options["default_prompt"] = args.prompt
    elif args.model == "rf-detr":
        if args.prompt:
            raise ValueError("--prompt is not supported by rf-detr")
        if args.io_binding:
            raise ValueError("--io-binding is only supported with --model onnx")
    elif args.model == "tesseract" and args.prompt:
        raise ValueError("--prompt is not supported by tesseract")
    elif args.model == "paddleocr" and args.prompt:
        raise ValueError("--prompt is not supported by paddleocr")
    elif args.io_binding:
        raise ValueError("--io-binding is only supported with --model onnx")
    return Yase(
        task=args.task,
        model=args.model,
        registry=default_registry(),
        **options,
    )


def _evaluate_mot(args: argparse.Namespace) -> int:
    predictions = load_mot_sequence(
        args.predictions, label=args.label, score_column=6, min_score=0.0
    )
    ground_truth = load_mot_sequence(args.ground_truth, label=args.label)
    payload = {
        "hota": evaluate_hota(
            predictions, ground_truth, iou_threshold=args.iou
        ).to_dict(),
        "hota_curve": evaluate_hota_curve(predictions, ground_truth).to_dict(),
        "tracking": evaluate_tracking(
            predictions, ground_truth, iou_threshold=args.iou
        ).to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _evaluate_coco(args: argparse.Namespace) -> int:
    dataset = load_coco_dataset(args.ground_truth, include_masks=args.masks)
    predictions = load_coco_predictions(
        args.predictions, dataset, include_masks=args.masks
    )
    class_aware = not args.no_class_aware
    payload = {
        "bbox": evaluate_mean_average_precision(
            predictions, dataset.frames(), class_aware=class_aware
        ).to_dict()
    }
    if args.masks:
        payload["mask"] = evaluate_mean_mask_average_precision(
            predictions, dataset.frames(), class_aware=class_aware
        ).to_dict()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _extract(args: argparse.Namespace) -> int:
    extractor = _make_extractor(args)
    results = extractor.extract_many(
        discover_images(args.images), max_workers=args.max_workers
    )
    valid = [result for result in results if result is not None]
    if args.output is not None:
        write_jsonl(valid, args.output, include_arrays=args.include_arrays)
    else:
        for result in valid:
            print(
                json.dumps(result_to_dict(result, include_arrays=args.include_arrays))
            )
    return 0


def _video(args: argparse.Namespace) -> int:
    source = int(args.source) if args.source.isdigit() else args.source
    extractor = _make_extractor(args)
    stream_type = RealtimeVideoStream if args.realtime else VideoStream
    stream_options = {"max_frames": args.max_frames}
    if not args.realtime:
        stream_options.update(
            stride=args.stride, max_fps=args.max_fps, batch_size=args.batch_size
        )
    stream = stream_type(source, extractor, **stream_options)
    results = []
    for item in stream:
        if args.output is not None:
            results.append(item.result)
        else:
            payload = result_to_dict(item.result, include_arrays=args.include_arrays)
            payload.update(
                frame_index=item.frame_index,
                timestamp=item.timestamp,
                processing_time=item.processing_time,
            )
            print(json.dumps(payload, ensure_ascii=False))
    if args.output is not None:
        write_jsonl(results, args.output, include_arrays=args.include_arrays)
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    extractor = _make_extractor(args)
    report = BenchmarkRunner(warmup=args.warmup).run(
        extractor,
        discover_images(args.images),
        name=f"{args.model}:{args.model_path}",
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
