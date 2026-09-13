"""Optional model adapters.

The core package has no model weights or framework dependency. These adapters
are initialized explicitly by applications and accept local model artifacts.
"""

from typing import Any, Optional

import numpy as np

from .core import SemanticResult, load_image


class CallableExtractor:
    """Turn a callable into a documented extractor backend."""

    def __init__(self, function: Any, task: str = "depth") -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        if task not in ("depth", "segmentation", "both"):
            raise ValueError("task must be depth, segmentation, or both")
        self.function = function
        self.task = task

    def extract(self, image: Any) -> SemanticResult:
        output = self.function(load_image(image))
        if isinstance(output, SemanticResult):
            return output
        if self.task == "depth":
            return SemanticResult(depth=np.asarray(output))
        if self.task == "segmentation":
            return SemanticResult(segmentation=np.asarray(output))
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
        device: Optional[str] = None,
        task: str = "depth",
        size: Optional[tuple] = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "install the torch extra to use TorchScriptExtractor"
            ) from exc
        self._torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        self.task = task
        self.size = size

    def extract(self, image: Any) -> SemanticResult:
        torch = self._torch
        array = load_image(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.size is not None:
            tensor = torch.nn.functional.interpolate(
                tensor, size=self.size, mode="bilinear", align_corners=False
            )
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        result = output.detach().cpu().numpy().squeeze()
        if self.task == "segmentation":
            return SemanticResult(segmentation=result)
        return SemanticResult(depth=result)


class OnnxRuntimeExtractor:
    """Run a local ONNX model with an injectable ONNX Runtime session.

    Models receive one float32 NCHW tensor in [0, 1]. No model or runtime is
    downloaded automatically. Pass session in tests or install the onnx extra
    for the default session factory.
    """

    def __init__(
        self,
        model_path: str,
        session: Optional[Any] = None,
        task: str = "depth",
        input_name: Optional[str] = None,
        output_names: Optional[Any] = None,
        size: Optional[tuple] = None,
        providers: Optional[list] = None,
    ) -> None:
        if task not in ("depth", "segmentation", "both"):
            raise ValueError("task must be depth, segmentation, or both")
        if size is not None and (len(size) != 2 or size[0] <= 0 or size[1] <= 0):
            raise ValueError("size must be a positive (width, height) pair")
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "install the onnx extra to use OnnxRuntimeExtractor"
                ) from exc
            kwargs = {"providers": providers} if providers is not None else {}
            session = ort.InferenceSession(model_path, **kwargs)
        self.session = session
        self.task = task
        self.size = size
        inputs = session.get_inputs()
        if not inputs:
            raise ValueError("ONNX session has no inputs")
        self.input_name = input_name or inputs[0].name
        self.output_names = list(output_names) if output_names is not None else None

    def _prepare(self, image: Any) -> np.ndarray:
        array = load_image(image).astype(np.float32)
        if self.size is not None:
            from PIL import Image

            width, height = self.size
            array = np.asarray(
                Image.fromarray(array.astype(np.uint8)).resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            )
        array = array.astype(np.float32, copy=False) / np.float32(255.0)
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])

    @staticmethod
    def _without_batch(output: Any) -> np.ndarray:
        value = np.asarray(output)
        if value.ndim > 0 and value.shape[0] == 1:
            value = value[0]
        return value

    def extract(self, image: Any) -> SemanticResult:
        tensor = self._prepare(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        if not outputs:
            raise ValueError("ONNX session returned no outputs")
        if self.task == "both" and len(outputs) < 2:
            raise ValueError("both task requires at least two ONNX outputs")
        values = [self._without_batch(output) for output in outputs]
        if self.task == "depth":
            return SemanticResult(depth=values[0])
        if self.task == "segmentation":
            return SemanticResult(segmentation=values[0])
        return SemanticResult(depth=values[0], segmentation=values[1])


__all__ = [
    "CallableExtractor",
    "OnnxRuntimeExtractor",
    "TorchScriptExtractor",
]
