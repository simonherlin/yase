"""Run small real-provider smoke tests without downloading model weights."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from yase.backends import OnnxRuntimeExtractor
from yase.runtimes import OpenVINOExtractor


def _openvino_smoke() -> dict[str, object]:
    import openvino as ov
    from openvino import opset8 as ops

    parameter = ops.parameter([-1, 3, 2, 2], ov.Type.f32, name="pixels")
    axes = ops.constant(np.array([1, 2, 3], dtype=np.int64))
    model = ov.Model(
        [ops.reduce_mean(parameter, axes, keep_dims=False)], [parameter], "mean"
    )
    compiled = ov.Core().compile_model(model, "CPU")
    extractor = OpenVINOExtractor(
        compiled_model=compiled, task="depth", size=(2, 2), async_jobs=2
    )
    images = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 255, dtype=np.uint8),
    ]
    sync = extractor.extract_batch(images)
    asynchronous = extractor.extract_batch_async(images)
    values = [float(item.depth) for item in sync]
    async_values = [float(item.depth) for item in asynchronous]
    if values != [0.0, 1.0] or async_values != values:
        raise AssertionError(f"unexpected OpenVINO values: {values}, {async_values}")
    return {
        "version": str(ov.get_version()),
        "devices": [str(device) for device in ov.Core().available_devices],
        "sync": values,
        "async": async_values,
    }


def _onnx_smoke() -> dict[str, object]:
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
        normal = OnnxRuntimeExtractor(str(path), task="depth", size=(2, 2))
        bound = OnnxRuntimeExtractor(
            str(path), task="depth", size=(2, 2), use_io_binding=True
        )
        values = [float(item.depth) for item in normal.extract_batch(images)]
        bound_values = [float(item.depth) for item in bound.extract_batch(images)]
    if values != [0.0, 1.0] or bound_values != values:
        raise AssertionError(f"unexpected ONNX values: {values}, {bound_values}")
    return {
        "version": str(ort.__version__),
        "providers": list(ort.get_available_providers()),
        "sync": values,
        "io_binding": bound_values,
    }


def main() -> None:
    report = {"openvino": _openvino_smoke(), "onnxruntime": _onnx_smoke()}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
