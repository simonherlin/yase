"""Optional accelerator runtimes for local, already-exported models.

The module deliberately imports neither OpenVINO nor TensorRT at import time.
This keeps the base package usable on CPU-only machines while exposing a
stable ``extract``/``extract_batch`` contract to applications.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np

from ._cache import RuntimeCache, freeze_cache_key
from ._timing import add_phase_timings
from .core import SemanticResult, load_image
from .limits import InputLimits


def _prepare_image(
    image: Any,
    size: tuple[int, int] | None,
    limits: InputLimits | None = None,
) -> np.ndarray:
    array = load_image(image, limits=limits).astype(np.float32, copy=False)
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
    images: Sequence[Any],
    size: tuple[int, int] | None,
    limits: InputLimits | None = None,
) -> np.ndarray:
    arrays = [_prepare_image(image, size, limits) for image in images]
    if not arrays:
        raise ValueError("extract_batch requires at least one image")
    if size is None and len({array.shape for array in arrays}) != 1:
        raise ValueError("images must have the same shape when size is None")
    return np.ascontiguousarray(np.stack(arrays, axis=0).transpose(0, 3, 1, 2))


def _validate_options(task: str, size: tuple[int, int] | None) -> None:
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
        model_path: str | None = None,
        device: str = "AUTO",
        task: str = "depth",
        size: tuple[int, int] | None = None,
        input_name: str | None = None,
        output_names: Sequence[str] | None = None,
        compiled_model: Any = None,
        core: Any = None,
        input_limits: InputLimits | None = None,
        async_queue: Any = None,
        async_jobs: int = 0,
        record_timings: bool = False,
        cache: RuntimeCache[Any] | None = None,
    ) -> None:
        _validate_options(task, size)
        if compiled_model is None and model_path is None:
            raise ValueError("model_path or compiled_model is required")
        if (
            isinstance(async_jobs, bool)
            or not isinstance(async_jobs, int)
            or async_jobs < 0
        ):
            raise ValueError("async_jobs must be a non-negative integer")
        if not isinstance(record_timings, bool):
            raise TypeError("record_timings must be a boolean")
        self.model_path = str(model_path) if model_path is not None else None
        self.device = device
        self.task = task
        self.size = size
        self.input_limits = input_limits
        self.async_queue = async_queue
        self.async_jobs = async_jobs
        self.record_timings = record_timings
        self.cache = cache
        self.output_names = tuple(output_names) if output_names is not None else None
        if compiled_model is None:
            try:
                import openvino as ov
            except ImportError as exc:
                raise ImportError(
                    "install the openvino extra to use OpenVINOExtractor"
                ) from exc
            runtime_core = core or ov.Core()

            def compile_model() -> Any:
                return runtime_core.compile_model(self.model_path, device)

            cache_key = freeze_cache_key(("openvino", self.model_path, device))
            compiled_model = (
                cache.get_or_create(cache_key, compile_model)
                if cache is not None
                else compile_model()
            )
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

    def _resolve_input(self, input_name: str | None) -> Any:
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
        started = time.perf_counter() if self.record_timings else 0.0
        tensor = _prepare_batch(images, self.size, self.input_limits)
        prepared = time.perf_counter() if self.record_timings else 0.0
        outputs = self._ordered_outputs(self._run(tensor))
        inferred = time.perf_counter() if self.record_timings else 0.0
        if not outputs:
            raise ValueError("OpenVINO model returned no outputs")
        metadata = add_phase_timings(
            {
                "backend": "openvino",
                "model_path": self.model_path,
                "device": self.device,
                "task": self.task,
            },
            self.record_timings,
            {
                "preprocess": prepared - started,
                "inference": inferred - prepared,
            },
        )
        postprocess_started = time.perf_counter() if self.record_timings else 0.0
        results = [
            _result_from_values(outputs, index, len(images), self.task, metadata)
            for index in range(len(images))
        ]
        if self.record_timings:
            metadata["timings_seconds"]["postprocess"] = (
                time.perf_counter() - postprocess_started
            )
        return results

    def _async_outputs(self, request: Any) -> list[Any]:
        """Read outputs from one completed OpenVINO infer request."""
        for attribute in ("results", "outputs"):
            value = getattr(request, attribute, None)
            if callable(value):
                value = value()
            if value is not None:
                return self._ordered_outputs(value)

        getter = getattr(request, "get_output_tensor", None)
        if callable(getter):
            ports = list(getattr(self.compiled_model, "outputs", []))
            count = len(ports) or len(self.output_names or ()) or 1
            values = []
            for index in range(count):
                tensor = getter(index)
                values.append(getattr(tensor, "data", tensor))
            return values
        raise TypeError(
            "OpenVINO async request must expose results, outputs, or get_output_tensor"
        )

    def _make_async_queue(self) -> Any:
        if self.async_queue is not None:
            return self.async_queue
        try:
            import openvino as ov
        except ImportError as exc:
            raise ImportError(
                "install the openvino extra to use OpenVINO async extraction"
            ) from exc
        jobs = self.async_jobs or 1
        return ov.AsyncInferQueue(self.compiled_model, jobs)

    def extract_batch_async(self, images: Sequence[Any]) -> list[SemanticResult]:
        """Infer one image per OpenVINO ``AsyncInferQueue`` request.

        The method waits for all submitted requests before returning, but lets
        OpenVINO overlap host submission and device execution. Results are
        reconstructed by userdata index, so provider completion order cannot
        reorder the public batch contract.
        """
        started = time.perf_counter() if self.record_timings else 0.0
        tensor = _prepare_batch(images, self.size, self.input_limits)
        prepared = time.perf_counter() if self.record_timings else 0.0
        queue = self._make_async_queue()
        set_callback = getattr(queue, "set_callback", None)
        start_async = getattr(queue, "start_async", None)
        wait_all = getattr(queue, "wait_all", None)
        if (
            not callable(set_callback)
            or not callable(start_async)
            or not callable(wait_all)
        ):
            raise TypeError(
                "OpenVINO async queue must expose set_callback, start_async, and "
                "wait_all"
            )

        outputs: list[list[Any] | None] = [None] * len(images)

        def callback(request: Any, userdata: Any = None) -> None:
            if not isinstance(userdata, int) or not 0 <= userdata < len(images):
                raise ValueError("OpenVINO async callback returned an invalid index")
            outputs[userdata] = self._async_outputs(request)

        set_callback(callback)
        for index in range(len(images)):
            start_async(
                {self._input: tensor[index : index + 1]},
                userdata=index,
            )
        wait_all()
        inferred = time.perf_counter() if self.record_timings else 0.0
        if any(value is None for value in outputs):
            raise RuntimeError("OpenVINO async queue completed with missing outputs")

        metadata = add_phase_timings(
            {
                "backend": "openvino",
                "model_path": self.model_path,
                "device": self.device,
                "task": self.task,
                "async_queue": True,
                "async_jobs": self.async_jobs or 1,
            },
            self.record_timings,
            {
                "preprocess": prepared - started,
                "inference": inferred - prepared,
            },
        )
        postprocess_started = time.perf_counter() if self.record_timings else 0.0
        results = [
            _result_from_values(value, 0, 1, self.task, metadata)
            for value in outputs
            if value is not None
        ]
        if self.record_timings:
            metadata["timings_seconds"]["postprocess"] = (
                time.perf_counter() - postprocess_started
            )
        return results

    def extract_async(self, image: Any) -> SemanticResult:
        """Extract one image through the OpenVINO asynchronous queue."""
        return self.extract_batch_async([image])[0]


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
        model_path: str | None = None,
        runner: Any = None,
        runner_pool: Any = None,
        runner_factory: Callable[[], Any] | None = None,
        pool_size: int = 1,
        task: str = "depth",
        size: tuple[int, int] | None = None,
        device: str = "cuda",
        input_limits: InputLimits | None = None,
        record_timings: bool = False,
        cache: RuntimeCache[Any] | None = None,
    ) -> None:
        _validate_options(task, size)
        if (
            isinstance(pool_size, bool)
            or not isinstance(pool_size, int)
            or pool_size < 1
        ):
            raise ValueError("pool_size must be a positive integer")
        if not isinstance(record_timings, bool):
            raise TypeError("record_timings must be a boolean")
        configured = sum(
            value is not None for value in (runner, runner_pool, runner_factory)
        )
        if configured > 1:
            raise ValueError("pass only one of runner, runner_pool, or runner_factory")
        if runner_factory is not None:
            runner = TensorRTContextPool(runner_factory, size=pool_size)
        elif runner_pool is not None:
            runner = runner_pool
        if runner is None:
            if model_path is None:
                raise ValueError("model_path or runner is required")
            if cache is not None:
                runner = cache.get_or_create(
                    freeze_cache_key(("tensorrt", str(model_path), device)),
                    lambda: _TorchTensorRTRunner(model_path),
                )
            else:
                runner = _TorchTensorRTRunner(model_path)
        if not callable(runner) and not hasattr(runner, "infer"):
            raise TypeError("runner must be callable or expose infer")
        self.model_path = str(model_path) if model_path is not None else None
        self.runner = runner
        self.task = task
        self.size = size
        self.device = device
        self.input_limits = input_limits
        self.record_timings = record_timings
        self.cache = cache

    def _infer(self, tensor: np.ndarray) -> Any:
        if hasattr(self.runner, "infer"):
            return self.runner.infer(tensor)
        return self.runner(tensor)

    def extract(self, image: Any) -> SemanticResult:
        return self.extract_batch([image])[0]

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        started = time.perf_counter() if self.record_timings else 0.0
        tensor = _prepare_batch(images, self.size, self.input_limits)
        prepared = time.perf_counter() if self.record_timings else 0.0
        outputs = self._infer(tensor)
        inferred = time.perf_counter() if self.record_timings else 0.0
        if isinstance(outputs, Mapping):
            values = list(outputs.values())
        elif isinstance(outputs, (tuple, list)):
            values = list(outputs)
        else:
            values = [outputs]
        if not values:
            raise ValueError("TensorRT runner returned no outputs")
        metadata = add_phase_timings(
            {
                "backend": "tensorrt",
                "model_path": self.model_path,
                "device": self.device,
                "task": self.task,
            },
            self.record_timings,
            {
                "preprocess": prepared - started,
                "inference": inferred - prepared,
            },
        )
        postprocess_started = time.perf_counter() if self.record_timings else 0.0
        results = [
            _result_from_values(values, index, len(images), self.task, metadata)
            for index in range(len(images))
        ]
        if self.record_timings:
            metadata["timings_seconds"]["postprocess"] = (
                time.perf_counter() - postprocess_started
            )
        return results

    def extract_batch_parallel(
        self,
        images: Sequence[Any],
        max_workers: int | None = None,
    ) -> list[SemanticResult]:
        """Run one image per independent TensorRT context and keep input order.

        This method requires a ``runner_pool`` or ``runner_factory``. It is
        intended for latency-sensitive streams and heterogeneous image sizes;
        ordinary ``extract_batch`` remains the preferred path for engines whose
        optimized batch dimension is more efficient than concurrent contexts.
        """
        if not isinstance(self.runner, TensorRTContextPool) and not hasattr(
            self.runner, "infer"
        ):
            raise TypeError(
                "extract_batch_parallel requires a runner_pool or runner_factory"
            )
        workers = max_workers or getattr(self.runner, "size", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("max_workers must be a positive integer")

        def extract_one(image: Any) -> SemanticResult:
            started = time.perf_counter() if self.record_timings else 0.0
            tensor = _prepare_batch([image], self.size, self.input_limits)
            prepared = time.perf_counter() if self.record_timings else 0.0
            outputs = self._infer(tensor)
            inferred = time.perf_counter() if self.record_timings else 0.0
            if isinstance(outputs, Mapping):
                values = list(outputs.values())
            elif isinstance(outputs, (tuple, list)):
                values = list(outputs)
            else:
                values = [outputs]
            if not values:
                raise ValueError("TensorRT runner returned no outputs")
            metadata = add_phase_timings(
                {
                    "backend": "tensorrt",
                    "model_path": self.model_path,
                    "device": self.device,
                    "task": self.task,
                    "parallel_contexts": True,
                },
                self.record_timings,
                {
                    "preprocess": prepared - started,
                    "inference": inferred - prepared,
                },
            )
            postprocess_started = time.perf_counter() if self.record_timings else 0.0
            result = _result_from_values(values, 0, 1, self.task, metadata)
            if self.record_timings:
                metadata["timings_seconds"]["postprocess"] = (
                    time.perf_counter() - postprocess_started
                )
            return result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(extract_one, images))

    def close(self) -> None:
        """Release a managed runner pool when it exposes ``close``."""
        close = getattr(self.runner, "close", None)
        if callable(close):
            close()


class TensorRTContextPool:
    """Thread-safe pool of independent TensorRT-compatible runners."""

    def __init__(self, factory: Callable[[], Any], size: int = 1) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer")
        self.size = size
        self._available: Queue[Any] = Queue(maxsize=size)
        self._runners: list[Any] = []
        for _ in range(size):
            runner = factory()
            if not callable(runner) and not hasattr(runner, "infer"):
                raise TypeError("factory must return a callable or infer runner")
            self._runners.append(runner)
            self._available.put(runner)
        self._closed = False

    def infer(self, tensor: np.ndarray) -> Any:
        if self._closed:
            raise RuntimeError("TensorRTContextPool is closed")
        runner = self._available.get()
        try:
            return runner.infer(tensor) if hasattr(runner, "infer") else runner(tensor)
        finally:
            self._available.put(runner)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for runner in self._runners:
            close = getattr(runner, "close", None)
            if callable(close):
                close()


__all__ = ["OpenVINOExtractor", "TensorRTContextPool", "TensorRTExtractor"]
