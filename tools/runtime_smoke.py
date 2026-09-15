"""Run small real-provider smoke tests without downloading model weights."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from yase.backends import OnnxRuntimeExtractor
from yase.runtimes import OpenVINOExtractor


def _openvino_smoke(devices: Sequence[str] = ("CPU",)) -> dict[str, object]:
    import openvino as ov
    from openvino import opset8 as ops

    parameter = ops.parameter([-1, 3, 2, 2], ov.Type.f32, name="pixels")
    axes = ops.constant(np.array([1, 2, 3], dtype=np.int64))
    model = ov.Model(
        [ops.reduce_mean(parameter, axes, keep_dims=False)], [parameter], "mean"
    )
    images = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 255, dtype=np.uint8),
    ]
    core = ov.Core()
    available = tuple(str(device) for device in core.available_devices)
    missing = tuple(device for device in devices if device not in available)
    if missing:
        raise RuntimeError(
            "requested OpenVINO devices are unavailable: "
            f"{', '.join(missing)}; available: {', '.join(available)}"
        )
    runs: dict[str, object] = {}
    for device in devices:
        compiled = core.compile_model(model, device)
        extractor = OpenVINOExtractor(
            compiled_model=compiled,
            device=device,
            task="depth",
            size=(2, 2),
            async_jobs=2,
        )
        sync = extractor.extract_batch(images)
        asynchronous = extractor.extract_batch_async(images)
        values = [float(item.depth) for item in sync]
        async_values = [float(item.depth) for item in asynchronous]
        if values != [0.0, 1.0] or async_values != values:
            raise AssertionError(
                f"unexpected OpenVINO values for {device}: {values}, {async_values}"
            )
        runs[device] = {"sync": values, "async": async_values}
    return {
        "version": str(ov.get_version()),
        "devices": list(available),
        "runs": runs,
    }


def _onnx_smoke(providers: Sequence[str] | None = None) -> dict[str, object]:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info(
        "pixels", TensorProto.FLOAT, ["batch", 3, 2, 2]
    )
    output_info = helper.make_tensor_value_info("mean", TensorProto.FLOAT, ["batch"])
    node = helper.make_node(
        "ReduceMean", ["pixels"], ["mean"], axes=[1, 2, 3], keepdims=0
    )
    graph = helper.make_graph([node], "mean", [input_info], [output_info])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 12)], producer_name="yase"
    )
    model.ir_version = 10
    images = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 255, dtype=np.uint8),
    ]
    with tempfile.TemporaryDirectory(prefix="yase-runtime-smoke-") as directory:
        path = Path(directory) / "mean.onnx"
        onnx.save(model, path)
        options = {}
        if providers:
            options = {"providers": list(providers), "strict_providers": True}
        normal = OnnxRuntimeExtractor(str(path), task="depth", size=(2, 2), **options)
        bound = OnnxRuntimeExtractor(
            str(path), task="depth", size=(2, 2), use_io_binding=True, **options
        )
        values = [float(item.depth) for item in normal.extract_batch(images)]
        bound_values = [float(item.depth) for item in bound.extract_batch(images)]
    if values != [0.0, 1.0] or bound_values != values:
        raise AssertionError(f"unexpected ONNX values: {values}, {bound_values}")
    return {
        "version": str(ort.__version__),
        "providers": list(ort.get_available_providers()),
        "requested_providers": list(providers or ()),
        "sync": values,
        "io_binding": bound_values,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-openvino-devices",
        action="store_true",
        help="compile and run the smoke graph on every available OpenVINO device",
    )
    parser.add_argument(
        "--onnx-provider",
        action="append",
        default=[],
        metavar="PROVIDER",
        help="request an ONNX provider and require it to be active (repeatable)",
    )
    args = parser.parse_args()
    import openvino as ov

    devices = (
        tuple(str(device) for device in ov.Core().available_devices)
        if args.all_openvino_devices
        else ("CPU",)
    )
    report = {
        "openvino": _openvino_smoke(devices),
        "onnxruntime": _onnx_smoke(args.onnx_provider or None),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
