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


__all__ = ["CallableExtractor", "TorchScriptExtractor"]
