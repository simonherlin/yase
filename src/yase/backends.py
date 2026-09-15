"""Optional model adapters.

The core package has no model weights or framework dependency. These adapters
are initialized explicitly by applications and accept local model artifacts.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import SemanticResult, _normalise_output, load_image
from .errors import BackendError
from .limits import InputLimits


class CallableExtractor:
    """Turn a callable into a documented extractor backend."""

    def __init__(
        self,
        function: Any,
        task: str = "depth",
        input_limits: InputLimits | None = None,
    ) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        if task not in ("depth", "segmentation", "both", "semantic"):
            raise ValueError("task must be depth, segmentation, both, or semantic")
        self.function = function
        self.task = task
        self.input_limits = input_limits

    def extract(self, image: Any) -> SemanticResult:
        output = self.function(load_image(image, limits=self.input_limits))
        if isinstance(output, SemanticResult):
            return output
        if self.task == "depth":
            return SemanticResult(depth=np.asarray(output))
        if self.task == "segmentation":
            return SemanticResult(segmentation=np.asarray(output))
        if self.task == "semantic":
            if isinstance(output, (Mapping, SemanticResult)):
                return _normalise_output(output, self.task, timestamp=None)
            raise TypeError("semantic task requires a SemanticResult or mapping")
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError("both task requires a (depth, segmentation) pair")
        return SemanticResult(
            depth=np.asarray(output[0]), segmentation=np.asarray(output[1])
        )


class TorchScriptExtractor:
    """Run a local TorchScript model without downloading weights.

    The model receives a float tensor shaped NCHW in the range [0, 1]. Its
    first output is converted to a NumPy array and exposed as depth by default.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        task: str = "depth",
        size: tuple | None = None,
        input_limits: InputLimits | None = None,
    ) -> None:
        if task not in ("depth", "segmentation", "both"):
            raise ValueError("task must be depth, segmentation, or both")
        if size is not None and (len(size) != 2 or size[0] <= 0 or size[1] <= 0):
            raise ValueError("size must be a positive (width, height) pair")
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "install the torch extra to use TorchScriptExtractor"
            ) from exc
        self._torch = torch
        self.model_path = str(model_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        self.task = task
        self.size = size
        self.input_limits = input_limits

    def _prepare_batch(self, images: Sequence[Any]) -> Any:
        torch = self._torch
        arrays = [
            load_image(image, limits=self.input_limits).astype(np.float32) / 255.0
            for image in images
        ]
        if not arrays:
            raise ValueError("extract_batch requires at least one image")
        if self.size is None and len({array.shape for array in arrays}) != 1:
            raise ValueError("images must have the same shape when size is None")
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device)
        if self.size is not None:
            width, height = self.size
            tensor = torch.nn.functional.interpolate(
                tensor, size=(height, width), mode="bilinear", align_corners=False
            )
        return tensor

    def _results_from_output(self, output: Any) -> list[SemanticResult]:
        if isinstance(output, (tuple, list)):
            if self.task == "both" and len(output) < 2:
                raise ValueError("both task requires at least two model outputs")
            outputs = [value.detach().cpu().numpy() for value in output]
        else:
            outputs = [output.detach().cpu().numpy()]
        results = []
        metadata = {
            "backend": "torchscript",
            "model_path": self.model_path,
            "device": str(self.device),
            "task": self.task,
        }
        for index in range(outputs[0].shape[0]):
            values = [value[index] for value in outputs]
            if self.task == "segmentation":
                results.append(
                    SemanticResult(segmentation=values[0].squeeze(), metadata=metadata)
                )
            elif self.task == "both":
                results.append(
                    SemanticResult(
                        depth=values[0].squeeze(),
                        segmentation=values[1].squeeze(),
                        metadata=metadata,
                    )
                )
            else:
                results.append(
                    SemanticResult(depth=values[0].squeeze(), metadata=metadata)
                )
        return results

    def extract(self, image: Any) -> SemanticResult:
        return self.extract_batch([image])[0]

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        """Run one model call for an ordered batch of images."""
        with self._torch.inference_mode():
            output = self.model(self._prepare_batch(images))
        return self._results_from_output(output)


class OnnxRuntimeExtractor:
    """Run a local ONNX model with an injectable ONNX Runtime session.

    Models receive one float32 NCHW tensor in [0, 1]. No model or runtime is
    downloaded automatically. Pass session in tests or install the onnx extra
    for the default session factory.
    """

    def __init__(
        self,
        model_path: str,
        session: Any | None = None,
        task: str = "depth",
        input_name: str | None = None,
        output_names: Any | None = None,
        size: tuple | None = None,
        providers: Sequence[Any] | None = None,
        provider_options: Sequence[Mapping[str, Any]] | None = None,
        session_options: Any | None = None,
        input_layout: str = "NCHW",
        graph_optimization_level: Any | None = None,
        enable_profiling: bool = False,
        use_io_binding: bool = False,
        input_limits: InputLimits | None = None,
    ) -> None:
        if task not in ("depth", "segmentation", "both"):
            raise ValueError("task must be depth, segmentation, or both")
        if size is not None and (len(size) != 2 or size[0] <= 0 or size[1] <= 0):
            raise ValueError("size must be a positive (width, height) pair")
        if input_layout not in ("NCHW", "NHWC"):
            raise ValueError("input_layout must be NCHW or NHWC")
        if not isinstance(use_io_binding, bool):
            raise TypeError("use_io_binding must be a boolean")
        if provider_options is not None and providers is None:
            raise ValueError("provider_options requires providers")
        if provider_options is not None and any(
            isinstance(provider, tuple) for provider in providers or ()
        ):
            raise ValueError(
                "use provider_options with provider names, not provider tuples"
            )
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "install the onnx extra to use OnnxRuntimeExtractor"
                ) from exc
            if session_options is None and (
                graph_optimization_level is not None or enable_profiling
            ):
                session_options = ort.SessionOptions()
            if session_options is not None:
                if graph_optimization_level is not None:
                    session_options.graph_optimization_level = graph_optimization_level
                if enable_profiling:
                    session_options.enable_profiling = True
            kwargs = {}
            if providers is not None:
                kwargs["providers"] = list(providers)
            if provider_options is not None:
                kwargs["provider_options"] = [dict(item) for item in provider_options]
            if session_options is not None:
                kwargs["sess_options"] = session_options
            session = ort.InferenceSession(model_path, **kwargs)
        self.session = session
        self.model_path = str(model_path)
        self.providers = providers
        self.provider_options = provider_options
        self.input_layout = input_layout
        self.task = task
        self.size = size
        self.input_limits = input_limits
        self.use_io_binding = use_io_binding
        inputs = session.get_inputs()
        if not inputs:
            raise ValueError("ONNX session has no inputs")
        self.input_name = input_name or inputs[0].name
        self.output_names = list(output_names) if output_names is not None else None

    @staticmethod
    def available_providers() -> tuple[str, ...]:
        """Return providers offered by the installed ONNX Runtime build."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "install the onnx extra to inspect ONNX Runtime providers"
            ) from exc
        return tuple(ort.get_available_providers())

    def _prepare_image(self, image: Any) -> np.ndarray:
        array = load_image(image, limits=self.input_limits).astype(np.float32)
        if self.size is not None:
            from PIL import Image

            width, height = self.size
            array = np.asarray(
                Image.fromarray(array.astype(np.uint8)).resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            )
        return array.astype(np.float32, copy=False) / np.float32(255.0)

    def _prepare(self, image: Any) -> np.ndarray:
        array = self._prepare_image(image)
        if self.input_layout == "NHWC":
            return np.ascontiguousarray(array[None, ...])
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])

    def _prepare_batch(self, images: Sequence[Any]) -> np.ndarray:
        arrays = [self._prepare_image(image) for image in images]
        if not arrays:
            raise ValueError("extract_batch requires at least one image")
        if self.size is None and len({array.shape for array in arrays}) != 1:
            raise ValueError("images must have the same shape when size is None")
        stacked = np.stack(arrays, axis=0)
        if self.input_layout == "NHWC":
            return np.ascontiguousarray(stacked)
        return np.ascontiguousarray(stacked.transpose(0, 3, 1, 2))

    @staticmethod
    def _without_batch(output: Any) -> np.ndarray:
        value = np.asarray(output)
        if value.ndim > 0 and value.shape[0] == 1:
            value = value[0]
        return value

    def _results_from_outputs(self, outputs: Any) -> list[SemanticResult]:
        if not outputs:
            raise ValueError("ONNX session returned no outputs")
        if self.task == "both" and len(outputs) < 2:
            raise ValueError("both task requires at least two ONNX outputs")
        batch_size = np.asarray(outputs[0]).shape[0]
        if batch_size < 1:
            raise ValueError("ONNX output batch is empty")
        values = [np.asarray(output) for output in outputs]
        results = []
        metadata = {
            "backend": "onnxruntime",
            "model_path": self.model_path,
            "task": self.task,
            "io_binding": self.use_io_binding,
            "providers": list(
                self.session.get_providers()
                if hasattr(self.session, "get_providers")
                else (self.providers or [])
            ),
        }
        for index in range(batch_size):
            fields = [self._without_batch(value[index : index + 1]) for value in values]
            if self.task == "depth":
                results.append(SemanticResult(depth=fields[0], metadata=metadata))
            elif self.task == "segmentation":
                results.append(
                    SemanticResult(segmentation=fields[0], metadata=metadata)
                )
            else:
                results.append(
                    SemanticResult(
                        depth=fields[0], segmentation=fields[1], metadata=metadata
                    )
                )
        return results

    def _run(self, tensor: np.ndarray) -> Any:
        if not self.use_io_binding:
            return self.session.run(self.output_names, {self.input_name: tensor})
        io_binding_factory = getattr(self.session, "io_binding", None)
        run_with_binding = getattr(self.session, "run_with_iobinding", None)
        if not callable(io_binding_factory) or not callable(run_with_binding):
            raise RuntimeError(
                "ONNX session does not support io_binding/run_with_iobinding"
            )
        binding = io_binding_factory()
        bind_cpu_input = getattr(binding, "bind_cpu_input", None)
        bind_output = getattr(binding, "bind_output", None)
        copy_outputs = getattr(binding, "copy_outputs_to_cpu", None)
        if not all(
            callable(item) for item in (bind_cpu_input, bind_output, copy_outputs)
        ):
            raise RuntimeError("ONNX I/O binding object is incomplete")
        bind_cpu_input(self.input_name, tensor)
        output_names = self.output_names
        if output_names is None:
            output_names = [output.name for output in self.session.get_outputs()]
        for output_name in output_names:
            bind_output(output_name, "cpu")
        run_with_binding(binding)
        return copy_outputs()

    def extract(self, image: Any) -> SemanticResult:
        return self.extract_batch([image])[0]

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        """Run one ONNX session call for an ordered batch of images."""
        tensor = self._prepare_batch(images)
        outputs = self._run(tensor)
        return self._results_from_outputs(outputs)


class CompositeExtractor:
    """Combine independent backends into one SemanticResult.

    backends maps an optional result field (for example depth or
    segmentation) to an extractor. Backends may return SemanticResult or
    mappings. Raw arrays require a field key. Conflicts use an explicit
    error, first, or last policy; backend failures are wrapped with their
    source name.
    """

    _FIELDS = (
        "depth",
        "segmentation",
        "detections",
        "tags",
        "embeddings",
        "ocr",
        "caption",
        "scene",
        "events",
        "keypoints",
        "relations",
        "document",
        "depth_map",
    )

    def __init__(self, backends: Any, conflict: str = "error") -> None:
        if conflict not in ("error", "first", "last"):
            raise ValueError("conflict must be error, first, or last")
        if isinstance(backends, Mapping):
            items = list(backends.items())
        elif isinstance(backends, Sequence) and not isinstance(backends, (str, bytes)):
            items = [(str(index), backend) for index, backend in enumerate(backends)]
        else:
            raise TypeError("backends must be a mapping or sequence")
        if not items:
            raise ValueError("at least one backend is required")
        self.backends = items
        self.conflict = conflict

    @classmethod
    def _coerce(cls, output: Any, default_field: str) -> SemanticResult:
        if isinstance(output, SemanticResult):
            return output
        if isinstance(output, Mapping):
            return SemanticResult(
                depth=output.get("depth"),
                segmentation=output.get("segmentation", output.get("mask")),
                detections=output.get("detections"),
                tags=output.get("tags"),
                embeddings=output.get("embeddings"),
                ocr=output.get("ocr", output.get("text")),
                caption=output.get("caption"),
                scene=output.get("scene"),
                events=output.get("events"),
                keypoints=output.get("keypoints", output.get("poses")),
                relations=output.get("relations"),
                document=output.get("document"),
                depth_map=output.get("depth_map"),
                timestamp=output.get("timestamp"),
                metadata={
                    key: value
                    for key, value in output.items()
                    if key not in cls._FIELDS + ("mask", "text", "timestamp", "poses")
                },
            )
        if default_field not in cls._FIELDS:
            raise TypeError(
                "raw backend output requires a field key: "
                "depth, segmentation, detections, tags, or embeddings"
            )
        return SemanticResult(**{default_field: output})

    @staticmethod
    def _invoke(backend: Any, image: Any) -> Any:
        if hasattr(backend, "extract"):
            return backend.extract(image)
        if hasattr(backend, "predict"):
            return backend.predict(image)
        if callable(backend):
            return backend(image)
        raise TypeError("backend must be callable or expose extract/predict")

    def _merge(self, merged: dict, field: str, value: Any, source: str) -> None:
        if value is None:
            return
        if field in merged:
            if self.conflict == "error":
                raise ValueError(
                    f"conflict for field '{field}' from backend '{source}'"
                )
            if self.conflict == "first":
                return
        merged[field] = value

    def _merge_results(
        self, values: Sequence[tuple[str, SemanticResult]]
    ) -> SemanticResult:
        merged = {}
        metadata = {}
        timestamp = None
        for source, result in values:
            for field in self._FIELDS:
                self._merge(merged, field, getattr(result, field), source)
            if result.timestamp is not None:
                if timestamp is not None and result.timestamp != timestamp:
                    if self.conflict == "error":
                        raise ValueError(
                            f"conflict for timestamp from backend '{source}'"
                        )
                    if self.conflict == "last":
                        timestamp = result.timestamp
                elif timestamp is None:
                    timestamp = result.timestamp
            for key, value in result.metadata.items():
                if key in metadata and self.conflict == "error":
                    raise ValueError(
                        f"conflict for metadata '{key}' from backend '{source}'"
                    )
                if key not in metadata or self.conflict == "last":
                    metadata[key] = value
        return SemanticResult(
            depth=merged.get("depth"),
            segmentation=merged.get("segmentation"),
            detections=merged.get("detections"),
            tags=merged.get("tags"),
            embeddings=merged.get("embeddings"),
            timestamp=timestamp,
            metadata=metadata,
            ocr=merged.get("ocr"),
            caption=merged.get("caption"),
            scene=merged.get("scene"),
            events=merged.get("events"),
            keypoints=merged.get("keypoints"),
            relations=merged.get("relations"),
            document=merged.get("document"),
            depth_map=merged.get("depth_map"),
        )

    def extract(self, image: Any) -> SemanticResult:
        """Run each backend and merge outputs, preserving timestamps."""
        values = []
        for source, backend in self.backends:
            try:
                result = self._coerce(self._invoke(backend, image), source)
            except Exception as exc:
                raise BackendError(str(source), str(exc), original=exc) from exc
            values.append((str(source), result))
        return self._merge_results(values)

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        """Run each backend in batch when supported and merge by input order."""
        items = list(images)
        if not items:
            raise ValueError("extract_batch requires at least one image")
        rows: list[list[tuple[str, SemanticResult]]] = [[] for _ in items]
        for source, backend in self.backends:
            try:
                if hasattr(backend, "extract_batch"):
                    outputs = list(backend.extract_batch(items))
                    if len(outputs) != len(items):
                        raise ValueError(
                            "extract_batch must return one result per input image"
                        )
                else:
                    outputs = [self._invoke(backend, image) for image in items]
                for index, output in enumerate(outputs):
                    rows[index].append((str(source), self._coerce(output, str(source))))
            except Exception as exc:
                raise BackendError(str(source), str(exc), original=exc) from exc
        return [self._merge_results(row) for row in rows]


__all__ = [
    "CallableExtractor",
    "CompositeExtractor",
    "OnnxRuntimeExtractor",
    "TorchScriptExtractor",
]
