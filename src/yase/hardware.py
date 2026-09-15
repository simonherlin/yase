"""Hardware-aware backend selection and resilient runtime fallback.

The selector is deliberately small and dependency-light. Detection uses the
existing lazy diagnostics probes, while actual backend construction remains
the source of truth: providers can be advertised by a runtime and still fail
when they touch an incompatible GPU or model operator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .core import SemanticResult
from .diagnostics import collect_provider_info


@dataclass(frozen=True)
class BackendCandidate:
    """One backend attempt produced by :class:`HardwareProfile`."""

    name: str
    options: Mapping[str, Any]
    reason: str


@dataclass(frozen=True)
class HardwareProfile:
    """Runtime capabilities used for adaptive backend selection.

    The fields describe advertised or detected capabilities, not a guarantee
    that every model will execute. ``AdaptiveExtractor`` validates the final
    choice by constructing and, when necessary, running the backend.
    """

    onnx_providers: tuple[str, ...] = ()
    openvino_devices: tuple[str, ...] = ()
    torch_cuda_available: bool = False
    torch_cuda_usable: bool = False
    tensorrt_available: bool = False

    @classmethod
    def detect(cls) -> HardwareProfile:
        """Probe installed runtimes without downloading models or weights."""
        info = collect_provider_info()
        onnx = info.get("onnxruntime", {})
        openvino = info.get("openvino", {})
        torch = info.get("torch", {})
        cuda = torch.get("cuda", {}) if isinstance(torch, Mapping) else {}
        return cls(
            onnx_providers=tuple(onnx.get("available_providers", ())),
            openvino_devices=tuple(
                str(item.get("device"))
                for item in openvino.get("devices", ())
                if isinstance(item, Mapping) and item.get("device")
            ),
            torch_cuda_available=bool(cuda.get("available", False)),
            torch_cuda_usable=bool(cuda.get("usable", cuda.get("available", False))),
            tensorrt_available=bool(info.get("tensorrt", {}).get("available", False)),
        )

    @property
    def has_openvino(self) -> bool:
        return bool(self.openvino_devices)

    @property
    def has_onnx(self) -> bool:
        return bool(self.onnx_providers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "onnx_providers": list(self.onnx_providers),
            "openvino_devices": list(self.openvino_devices),
            "torch_cuda_available": self.torch_cuda_available,
            "torch_cuda_usable": self.torch_cuda_usable,
            "tensorrt_available": self.tensorrt_available,
        }

    def candidates(
        self,
        model_path: str,
        *,
        task: str = "depth",
        device: str = "auto",
        preference: str = "auto",
        options: Mapping[str, Any] | None = None,
    ) -> tuple[BackendCandidate, ...]:
        """Return ordered backend candidates for a local model artifact."""
        if not model_path:
            raise ValueError("model_path is required for adaptive selection")
        if task not in ("depth", "segmentation", "both"):
            raise ValueError("task must be depth, segmentation, or both")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        if not isinstance(preference, str) or not preference:
            raise ValueError("preference must be a non-empty string")

        base = dict(options or {})
        suffix = Path(model_path).suffix.lower()
        requested_device = device.strip()
        automatic_device = requested_device.lower() in {"auto", "automatic"}
        explicit_preference = preference.lower() not in {"auto", "automatic"}
        if explicit_preference:
            order = [preference.lower()]
        elif suffix in {".plan", ".engine"}:
            order = ["tensorrt"]
        elif suffix in {".xml", ".bin", ".blob"}:
            order = ["openvino"]
        elif suffix in {".pt", ".pth", ".jit", ".torchscript", ".ts"}:
            order = ["torchscript"]
        elif suffix == ".onnx":
            order = []
            if self.has_openvino:
                order.append("openvino")
            if "TensorrtExecutionProvider" in self.onnx_providers:
                order.append("onnx-tensorrt")
            if "CUDAExecutionProvider" in self.onnx_providers:
                order.append("onnx-cuda")
            order.append("onnx-cpu")
        else:
            order = ["openvino", "onnx-cpu", "torchscript"]

        if not automatic_device and suffix == ".onnx":
            normalized_device = requested_device.lower()
            if normalized_device in {"cpu", "host"}:
                order = [
                    backend
                    for backend in order
                    if backend not in {"onnx-tensorrt", "onnx-cuda"}
                ]
            elif normalized_device in {"cuda", "gpu"} or normalized_device.startswith(
                "cuda:"
            ):
                order = [backend for backend in order if backend != "onnx-cpu"]

        candidates: list[BackendCandidate] = []
        seen: set[str] = set()
        for backend in order:
            if backend in seen:
                continue
            seen.add(backend)
            candidate = self._candidate(
                backend,
                model_path=model_path,
                task=task,
                device=requested_device,
                automatic_device=automatic_device,
                explicit_preference=explicit_preference,
                base=base,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            raise ValueError(
                f"could not derive an adaptive backend for model artifact {model_path!r}"
            )
        return tuple(candidates)

    def _candidate(
        self,
        backend: str,
        *,
        model_path: str,
        task: str,
        device: str,
        automatic_device: bool,
        explicit_preference: bool,
        base: Mapping[str, Any],
    ) -> BackendCandidate | None:
        options = dict(base)
        options.update(model_path=model_path, task=task)
        if backend == "openvino":
            if not self.has_openvino and not explicit_preference:
                return None
            selected_device = device
            if automatic_device:
                selected_device = "AUTO" if self.has_openvino else "CPU"
            elif selected_device.lower() in {"cuda", "gpu"}:
                selected_device = "GPU"
            options["device"] = selected_device
            return BackendCandidate(
                "openvino", options, f"OpenVINO device selection: {selected_device}"
            )
        if backend == "tensorrt":
            if not self.tensorrt_available and not explicit_preference:
                return None
            options["device"] = device if not automatic_device else "cuda"
            return BackendCandidate("tensorrt", options, "TensorRT CUDA plan")
        if backend == "torchscript":
            selected_device = device
            if automatic_device:
                selected_device = "cuda" if self.torch_cuda_usable else "cpu"
            options["device"] = selected_device
            return BackendCandidate(
                "torchscript",
                options,
                f"TorchScript device selection: {selected_device}",
            )
        if backend in {"onnx-tensorrt", "onnx-cuda", "onnx-cpu"}:
            providers = {
                "onnx-tensorrt": ["TensorrtExecutionProvider"],
                "onnx-cuda": ["CUDAExecutionProvider"],
                "onnx-cpu": ["CPUExecutionProvider"],
            }[backend]
            if backend != "onnx-cpu" and not explicit_preference:
                if not all(provider in self.onnx_providers for provider in providers):
                    return None
            if "providers" not in options:
                options["providers"] = providers
            options.setdefault("strict_providers", True)
            return BackendCandidate(
                "onnx", options, f"ONNX Runtime providers: {', '.join(providers)}"
            )
        if backend == "onnx":
            if "providers" not in options:
                normalized_device = device.lower()
                if normalized_device in {"cuda", "gpu"} or normalized_device.startswith(
                    "cuda:"
                ):
                    options["providers"] = ["CUDAExecutionProvider"]
                else:
                    options["providers"] = ["CPUExecutionProvider"]
            return BackendCandidate("onnx", options, "explicit ONNX Runtime backend")
        return BackendCandidate(
            backend, options, "explicit adaptive backend preference"
        )


class AdaptiveExtractor:
    """Instantiate and run the best compatible local backend.

    Initialization failures and provider-level inference failures are retried
    with the next candidate. Every successful result records the selected
    backend and the hardware profile in metadata, making fallback observable.
    """

    def __init__(
        self,
        model_path: str,
        *,
        task: str = "depth",
        device: str = "auto",
        preference: str = "auto",
        fallback: bool = True,
        registry: Any | None = None,
        hardware: HardwareProfile | None = None,
        **options: Any,
    ) -> None:
        if not isinstance(fallback, bool):
            raise TypeError("fallback must be a boolean")
        self.model_path = str(model_path)
        self.task = task
        self.device = device
        self.preference = preference
        self.fallback = fallback
        self.hardware = hardware or HardwareProfile.detect()
        self.registry = registry
        self.candidates = self.hardware.candidates(
            self.model_path,
            task=task,
            device=device,
            preference=preference,
            options=options,
        )
        self._backend: Any | None = None
        self._candidate_index = -1
        self._attempt_errors: list[str] = []

    @property
    def backend_name(self) -> str | None:
        if self._backend is None:
            return None
        return self.candidates[self._candidate_index].name

    @property
    def selected_candidate(self) -> BackendCandidate | None:
        if self._backend is None:
            return None
        return self.candidates[self._candidate_index]

    @property
    def backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        start = self._candidate_index + 1
        for index in range(start, len(self.candidates)):
            candidate = self.candidates[index]
            try:
                backend = self._create(candidate)
            except Exception as exc:
                self._attempt_errors.append(
                    f"{candidate.name}: {type(exc).__name__}: {exc}"
                )
                if not self.fallback:
                    break
                continue
            self._candidate_index = index
            self._backend = backend
            return backend
        details = "; ".join(self._attempt_errors) or "no backend candidate"
        raise RuntimeError(
            f"no compatible backend could load {self.model_path!r}: {details}"
        )

    def _create(self, candidate: BackendCandidate) -> Any:
        registry = self.registry
        if registry is None:
            from .registry import default_registry

            registry = default_registry()
        return registry.create(candidate.name, **dict(candidate.options))

    def _annotate(self, result: SemanticResult) -> SemanticResult:
        candidate = self.selected_candidate
        metadata = dict(result.metadata)
        metadata["adaptive_backend"] = self.backend_name
        metadata["adaptive_reason"] = candidate.reason if candidate else None
        metadata["hardware_profile"] = self.hardware.to_dict()
        if self._attempt_errors:
            metadata["adaptive_fallback_errors"] = list(self._attempt_errors)
        return replace(result, metadata=metadata)

    def _run(self, method: str, *args: Any) -> Any:
        try:
            backend = self.backend
            operation = getattr(backend, method, None)
            if not callable(operation):
                if method == "extract_batch":
                    return [self._annotate(backend.extract(item)) for item in args[0]]
                raise TypeError(f"selected backend does not expose {method}")
            output = operation(*args)
            if method == "extract_batch":
                return [self._annotate(item) for item in output]
            return self._annotate(output)
        except Exception as exc:
            if not self.fallback or self._candidate_index + 1 >= len(self.candidates):
                raise
            current = self.selected_candidate
            self._attempt_errors.append(
                f"{current.name if current else 'unknown'} inference: "
                f"{type(exc).__name__}: {exc}"
            )
            self._backend = None
            return self._run(method, *args)

    def extract(self, image: Any) -> SemanticResult:
        return self._run("extract", image)

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        return self._run("extract_batch", images)

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


__all__ = ["AdaptiveExtractor", "BackendCandidate", "HardwareProfile"]
