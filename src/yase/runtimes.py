"""Optional accelerator runtimes for local, already-exported models.

The module deliberately imports neither OpenVINO nor TensorRT at import time.
This keeps the base package usable on CPU-only machines while exposing a
stable ``extract``/``extract_batch`` contract to applications.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .core import SemanticResult, load_image


def _prepare_image(image: Any, size: Optional[tuple[int, int]]) -> np.ndarray:
    array = load_image(image).astype(np.float32, copy=False)
    if size is not None:
        from PIL import Image

        width, height = size
        array = np.asarray(
            Image.fromarray(array.astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        ).astype(np.float32, copy=False)
    return array / np.float32(255.0)


def _prepare_batch(
    images: Sequence[Any], size: Optional[tuple[int, int]]
) -> np.ndarray:
    arrays = [_prepare_image(image, size) for image in images]
    if not arrays:
        raise ValueError("extract_batch requires at least one image")
    if size is None and len({array.shape for array in arrays}) != 1:
        raise ValueError("images must have the same shape when size is None")
    return np.ascontiguousarray(np.stack(arrays, axis=0).transpose(0, 3, 1, 2))


def _validate_options(task: str, size: Optional[tuple[int, int]]) -> None:
    if task not in ("depth", "segmentation", "both"):
        raise ValueError("task must be depth, segmentation, or both")
    if size is not None and (len(size) != 2 or any(value <= 0 for value in size)):
        raise ValueError("size must be a positive (width, height) pair")


def _result_from_values(
    values: Sequence[Any],
    index: int,
    batch_size: int,
    task: str,
    metadata: dict[str, Any],
) -> SemanticResult:
    fields = []
    for value in values:
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == batch_size:
            array = array[index]
        elif batch_size != 1:
            raise ValueError("runtime output does not expose the input batch dimension")
        fields.append(array)
    if task == "depth":
        return SemanticResult(depth=fields[0], metadata=metadata)
    if task == "segmentation":
        return SemanticResult(segmentation=fields[0], metadata=metadata)
    if len(fields) < 2:
        raise ValueError("both task requires at least two runtime outputs")
    return SemanticResult(depth=fields[0], segmentation=fields[1], metadata=metadata)


class OpenVINOExtractor:
    """Run a local OpenVINO IR/ONNX model on CPU, GPU, NPU, or AUTO.

    ``compiled_model`` is injectable for testing and for applications that
    manage model compilation/caching themselves. Otherwise the adapter lazily
    imports OpenVINO and calls ``Core.compile_model``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "AUTO",
        task: str = "depth",
        size: Optional[tuple[int, int]] = None,
        input_name: Optional[str] = None,
        output_names: Optional[Sequence[str]] = None,
        compiled_model: Any = None,
        core: Any = None,
    ) -> None:
        _validate_options(task, size)
        if compiled_model is None and model_path is None:
            raise ValueError("model_path or compiled_model is required")
        self.model_path = str(model_path) if model_path is not None else None
        self.device = device
        self.task = task
        self.size = size
        self.output_names = tuple(output_names) if output_names is not None else None
        if compiled_model is None:
            try:
                import openvino as ov
            except ImportError as exc:
                raise ImportError(
                    "install the openvino extra to use OpenVINOExtractor"
                ) from exc
            runtime_core = core or ov.Core()
            compiled_model = runtime_core.compile_model(self.model_path, device)
        self.compiled_model = compiled_model
        self._input = self._resolve_input(input_name)
        self.input_name = self._port_name(self._input)

    @staticmethod
    def _port_name(port: Any) -> str:
        getter = getattr(port, "get_any_name", None)
        if callable(getter):
            return str(getter())
        name = getattr(port, "any_name", None)
        return str(name) if name is not None else str(port)

    def _resolve_input(self, input_name: Optional[str]) -> Any:
        inputs = list(getattr(self.compiled_model, "inputs", []))
        if not inputs:
            raise ValueError("OpenVINO compiled model has no inputs")
        if input_name is None:
            return inputs[0]
        for port in inputs:
            if self._port_name(port) == input_name:
                return port
        return input_name

    def _run(self, tensor: np.ndarray) -> Any:
        try:
            return self.compiled_model({self._input: tensor})
        except (TypeError, KeyError):
            return self.compiled_model({self.input_name: tensor})

    def _ordered_outputs(self, outputs: Any) -> list[Any]:
        if isinstance(outputs, Mapping) or hasattr(outputs, "values"):
            if self.output_names is not None:
                values = []
                for name in self.output_names:
                    try:
                        values.append(outputs[name])
                    except (KeyError, TypeError):
                        values.append(
                            outputs[
                                next(
                                    key
                                    for key in outputs
                                    if self._port_name(key) == name
                                )
                            ]
                        )
                return values
            return list(outputs.values())
        if isinstance(outputs, (tuple, list)):
            return list(outputs)
        return [outputs]

    def extract(self, image: Any) -> SemanticResult:
        return self.extract_batch([image])[0]

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        tensor = _prepare_batch(images, self.size)
        outputs = self._ordered_outputs(self._run(tensor))
        if not outputs:
            raise ValueError("OpenVINO model returned no outputs")
        metadata = {
            "backend": "openvino",
            "model_path": self.model_path,
            "device": self.device,
            "task": self.task,
        }
        return [
            _result_from_values(outputs, index, len(images), self.task, metadata)
            for index in range(len(images))
        ]


class _TorchTensorRTRunner:
    """Small native TensorRT runner using the official tensor-address API."""

    def __init__(self, model_path: str) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "install TensorRT or provide a custom runner to TensorRTExtractor"
            ) from exc
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "install torch with CUDA or provide a custom runner to "
                "TensorRTExtractor"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRTExtractor requires an available CUDA device")
        self._torch = torch
        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = Path(model_path).read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine: {model_path}")
        self.engine = engine
        self.context = engine.create_execution_context()

    def __call__(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        torch = self._torch
        input_names = []
        output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        if len(input_names) != 1:
            raise ValueError("TensorRTExtractor currently requires one input tensor")
        input_name = input_names[0]
        input_dtype = self._trt.nptype(self.engine.get_tensor_dtype(input_name))
        torch_dtype = torch.from_numpy(np.empty((), dtype=input_dtype)).dtype
        input_tensor = torch.as_tensor(tensor, device="cuda", dtype=torch_dtype)
        self.context.set_input_shape(input_name, tuple(input_tensor.shape))
        buffers: dict[str, Any] = {input_name: input_tensor}
        for name in output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(value < 0 for value in shape):
                raise ValueError(f"unresolved dynamic TensorRT output shape: {name}")
            dtype = self._trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            buffers[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
        for name, buffer in buffers.items():
            self.context.set_tensor_address(name, int(buffer.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execution failed")
        torch.cuda.synchronize()
        return {
            name: value.detach().cpu().numpy()
            for name, value in buffers.items()
            if name != input_name
        }


class TensorRTExtractor:
    """Run a TensorRT plan through a custom runner or native CUDA runner.

    A custom ``runner`` can be a callable or expose ``infer(array)`` and may
    return an array, a sequence of arrays, or a mapping of named outputs. This
    makes Polygraphy, CuPy, Triton and application-specific memory managers
    first-class without forcing them into Yase's base dependencies.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        runner: Any = None,
        task: str = "depth",
        size: Optional[tuple[int, int]] = None,
        device: str = "cuda",
    ) -> None:
        _validate_options(task, size)
        if runner is None:
            if model_path is None:
                raise ValueError("model_path or runner is required")
            runner = _TorchTensorRTRunner(model_path)
        if not callable(runner) and not hasattr(runner, "infer"):
            raise TypeError("runner must be callable or expose infer")
        self.model_path = str(model_path) if model_path is not None else None
        self.runner = runner
        self.task = task
        self.size = size
        self.device = device

    def extract(self, image: Any) -> SemanticResult:
        return self.extract_batch([image])[0]

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        tensor = _prepare_batch(images, self.size)
        outputs = (
            self.runner.infer(tensor)
            if hasattr(self.runner, "infer")
            else self.runner(tensor)
        )
        if isinstance(outputs, Mapping):
            values = list(outputs.values())
        elif isinstance(outputs, (tuple, list)):
            values = list(outputs)
        else:
            values = [outputs]
        if not values:
            raise ValueError("TensorRT runner returned no outputs")
        metadata = {
            "backend": "tensorrt",
            "model_path": self.model_path,
            "device": self.device,
            "task": self.task,
        }
        return [
            _result_from_values(values, index, len(images), self.task, metadata)
            for index in range(len(images))
        ]


__all__ = ["OpenVINOExtractor", "TensorRTExtractor"]
