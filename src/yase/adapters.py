"""Optional high-level adapters for common semantic extraction tasks.

All heavy imports happen during adapter construction. Model identifiers are
never downloaded implicitly: Transformers adapters default to local-only model
loading and require ``local_files_only=False`` explicitly for hub access.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import SemanticResult, load_image
from .schema import BoundingBox, Detection, TextRegion
from .structured import StructuredQuery, parse_structured_output


class TesseractExtractor:
    """Extract text regions through an injected or local Tesseract engine."""

    def __init__(
        self,
        language: str = "eng",
        config: str = "",
        engine: Any | None = None,
    ) -> None:
        if not language:
            raise ValueError("language must not be empty")
        if engine is None:
            try:
                import pytesseract
            except ImportError as exc:
                raise ImportError(
                    "install the ocr extra to use TesseractExtractor"
                ) from exc
            engine = pytesseract
        if not hasattr(engine, "image_to_data"):
            raise TypeError("engine must expose image_to_data")
        self.engine = engine
        self.language = language
        self.config = config

    def extract(self, image: Any) -> SemanticResult:
        from PIL import Image

        rgb = load_image(image)
        output_type = getattr(getattr(self.engine, "Output", None), "DICT", "dict")
        data = self.engine.image_to_data(
            Image.fromarray(rgb),
            lang=self.language,
            config=self.config,
            output_type=output_type,
        )
        if not isinstance(data, Mapping):
            raise TypeError("Tesseract image_to_data must return a mapping")
        regions = []
        texts = data.get("text", [])
        count = len(texts)
        for index in range(count):
            text = str(texts[index]).strip()
            try:
                score = float(data.get("conf", [-1] * count)[index]) / 100.0
            except (TypeError, ValueError, IndexError):
                score = 0.0
            if not text or score < 0:
                continue
            try:
                box = BoundingBox.from_xywh(
                    float(data["left"][index]),
                    float(data["top"][index]),
                    float(data["width"][index]),
                    float(data["height"][index]),
                )
            except (KeyError, IndexError, TypeError, ValueError):
                box = None
            regions.append(
                TextRegion(
                    text=text,
                    score=min(1.0, max(0.0, score)),
                    box=box,
                    language=self.language,
                )
            )
        return SemanticResult(
            ocr=regions,
            metadata={"backend": "tesseract", "language": self.language},
        )


class PaddleOCRExtractor:
    """Adapter for PaddleOCR 2.x/3.x engines with a normalized output shape."""

    def __init__(self, language: str = "en", engine: Any | None = None, **options: Any):
        if engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "install the paddle extra and a PaddlePaddle engine "
                    "(paddlepaddle or paddlepaddle-gpu) to use "
                    "PaddleOCRExtractor"
                ) from exc
            engine = PaddleOCR(lang=language, **options)
        if not hasattr(engine, "predict") and not hasattr(engine, "ocr"):
            raise TypeError("engine must expose predict or ocr")
        self.engine = engine
        self.language = language

    @staticmethod
    def _mapping_regions(item: Mapping[str, Any]) -> list[TextRegion]:
        texts = item.get("rec_texts", item.get("texts", []))
        scores = item.get("rec_scores", item.get("scores", []))
        boxes = item.get("rec_boxes", item.get("boxes", []))
        regions = []
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 1.0
            box = None
            if index < len(boxes):
                values = np.asarray(boxes[index]).reshape(-1)
                if values.size == 4:
                    box = BoundingBox.from_sequence(values.tolist())
                elif values.size >= 8:
                    points = np.asarray(boxes[index], dtype=np.float32).reshape(-1, 2)
                    box = BoundingBox(
                        float(points[:, 0].min()),
                        float(points[:, 1].min()),
                        float(points[:, 0].max()),
                        float(points[:, 1].max()),
                    )
            regions.append(
                TextRegion(str(text), min(1.0, max(0.0, score)), box=box, language=None)
            )
        return regions

    def extract(self, image: Any) -> SemanticResult:
        rgb = load_image(image)
        if hasattr(self.engine, "predict"):
            raw = self.engine.predict(rgb)
        else:
            raw = self.engine.ocr(rgb, cls=True)
        regions: list[TextRegion] = []
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        for item in items:
            if isinstance(item, Mapping):
                regions.extend(self._mapping_regions(item))
                continue
            if isinstance(item, (list, tuple)):
                for value in item:
                    if not isinstance(value, (list, tuple)) or len(value) < 2:
                        continue
                    polygon, text_score = value[0], value[1]
                    if not isinstance(text_score, (list, tuple)) or len(text_score) < 2:
                        continue
                    text, score = text_score[0], float(text_score[1])
                    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    box = BoundingBox(
                        float(points[:, 0].min()),
                        float(points[:, 1].min()),
                        float(points[:, 0].max()),
                        float(points[:, 1].max()),
                    )
                    regions.append(TextRegion(str(text), score, box=box))
        return SemanticResult(
            ocr=regions,
            metadata={"backend": "paddleocr", "language": self.language},
        )


class TransformersImageEmbeddingExtractor:
    """Create image embeddings with a local Hugging Face vision encoder."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        normalize: bool = True,
        local_files_only: bool = True,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if not model_id and model is None:
            raise ValueError("model_id is required")
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "install the transformers extra to use this adapter"
            ) from exc
        if processor is None or model is None:
            try:
                from transformers import AutoImageProcessor, AutoModel
            except ImportError as exc:
                raise ImportError(
                    "install the transformers extra to use this adapter"
                ) from exc
            options = {"local_files_only": local_files_only}
            processor = processor or AutoImageProcessor.from_pretrained(
                model_id, **options
            )
            model = model or AutoModel.from_pretrained(model_id, **options)
        self._torch = torch
        self.model_id = model_id
        self.normalize = normalize
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = processor
        self.model = model.to(self.device).eval()

    def _move(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        if not images:
            raise ValueError("extract_batch requires at least one image")
        arrays = [load_image(image) for image in images]
        inputs = self.processor(images=arrays, return_tensors="pt")
        with self._torch.inference_mode():
            output = self.model(**self._move(inputs))
        embeddings = getattr(output, "pooler_output", None)
        if embeddings is None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                raise ValueError("vision model output has no embedding tensor")
            embeddings = hidden.mean(dim=1)
        embeddings = embeddings.detach().float()
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
        values = embeddings.cpu().numpy()
        metadata = {
            "backend": "transformers-image-embedding",
            "model_id": self.model_id,
            "normalized": self.normalize,
        }
        return [SemanticResult(embeddings=value, metadata=metadata) for value in values]

    def extract(self, image: Any) -> SemanticResult:
        return self.extract_batch([image])[0]


class TransformersObjectDetectionExtractor:
    """Normalize Hugging Face DETR-style object detection outputs."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        threshold: float = 0.5,
        local_files_only: bool = True,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if not model_id and model is None:
            raise ValueError("model_id is required")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "install the transformers extra to use this adapter"
            ) from exc
        if processor is None or model is None:
            try:
                from transformers import AutoImageProcessor, AutoModelForObjectDetection
            except ImportError as exc:
                raise ImportError(
                    "install the transformers extra to use this adapter"
                ) from exc
            options = {"local_files_only": local_files_only}
            processor = processor or AutoImageProcessor.from_pretrained(
                model_id, **options
            )
            model = model or AutoModelForObjectDetection.from_pretrained(
                model_id, **options
            )
        self._torch = torch
        self.model_id = model_id
        self.threshold = threshold
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = processor
        self.model = model.to(self.device).eval()

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        if not images:
            raise ValueError("extract_batch requires at least one image")
        arrays = [load_image(image) for image in images]
        inputs = self.processor(images=arrays, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        sizes = [(array.shape[0], array.shape[1]) for array in arrays]
        decoded = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=sizes
        )
        labels = getattr(self.model.config, "id2label", {})
        results = []
        metadata = {
            "backend": "transformers-object-detection",
            "model_id": self.model_id,
            "threshold": self.threshold,
        }
        for item in decoded:
            detections = []
            boxes = item.get("boxes", [])
            scores = item.get("scores", [])
            classes = item.get("labels", [])
            for box, score, class_id in zip(boxes, scores, classes):
                score_value = float(score.item() if hasattr(score, "item") else score)
                class_value = int(
                    class_id.item() if hasattr(class_id, "item") else class_id
                )
                box_values = (
                    box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
                )
                detections.append(
                    Detection(
                        str(labels.get(class_value, class_value)),
                        min(1.0, max(0.0, score_value)),
                        BoundingBox.from_sequence(box_values),
                    )
                )
            results.append(SemanticResult(detections=detections, metadata=metadata))
        return results

    def extract(self, image: Any) -> SemanticResult:
        return self.extract_batch([image])[0]


class TransformersGroundingDinoExtractor(TransformersObjectDetectionExtractor):
    """Open-vocabulary detection adapter for zero-shot Transformers models."""

    def __init__(self, model_id: str, labels: Sequence[str], **kwargs: Any) -> None:
        if not labels:
            raise ValueError("labels must contain at least one concept")
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "install the transformers extra to use this adapter"
            ) from exc
        local_files_only = kwargs.pop("local_files_only", True)
        processor = kwargs.pop("processor", None) or AutoProcessor.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        model = kwargs.pop(
            "model", None
        ) or AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        super().__init__(
            model_id,
            processor=processor,
            model=model,
            **kwargs,
        )
        self.labels = list(labels)

    def extract(self, image: Any) -> SemanticResult:
        from PIL import Image

        rgb = load_image(image)
        inputs = self.processor(
            text=[self.labels], images=[Image.fromarray(rgb)], return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        if not hasattr(self.processor, "post_process_grounded_object_detection"):
            raise RuntimeError(
                "processor does not support grounded detection postprocessing"
            )
        decoded = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=[(rgb.shape[0], rgb.shape[1])],
            text_labels=[self.labels],
        )[0]
        detections = []
        for box, score, label in zip(
            decoded.get("boxes", []),
            decoded.get("scores", []),
            decoded.get("text_labels", decoded.get("labels", [])),
        ):
            score_value = float(score.item() if hasattr(score, "item") else score)
            box_values = (
                box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            )
            detections.append(
                Detection(
                    str(label),
                    min(1.0, max(0.0, score_value)),
                    BoundingBox.from_sequence(box_values),
                )
            )
        return SemanticResult(
            detections=detections,
            metadata={
                "backend": "transformers-grounding-dino",
                "model_id": self.model_id,
                "labels": self.labels,
                "threshold": self.threshold,
            },
        )


class TransformersVLMExtractor:
    """Generate a caption/answer with a local image-text-to-text model."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        default_prompt: str = "Describe the image concisely.",
        max_new_tokens: int = 128,
        local_files_only: bool = True,
        processor: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if not model_id and model is None:
            raise ValueError("model_id is required")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if torch_module is None:
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "install the transformers extra to use this adapter"
                ) from exc
            torch_module = torch
        if processor is None or model is None:
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "install the transformers extra to use this adapter"
                ) from exc
            options = {"local_files_only": local_files_only}
            processor = processor or AutoProcessor.from_pretrained(model_id, **options)
            model = model or AutoModelForImageTextToText.from_pretrained(
                model_id, **options
            )
        self._torch = torch_module
        self.model_id = model_id
        self.device = torch_module.device(
            device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        self.default_prompt = default_prompt
        self.max_new_tokens = max_new_tokens
        self.processor = processor
        self.model = model.to(self.device).eval()

    def _prompt(self, prompt: str) -> str:
        if not hasattr(self.processor, "apply_chat_template"):
            return prompt
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_text(
        self, image: Any, prompt: str, max_new_tokens: int | None = None
    ) -> str:
        return self._generate_text_batch(
            [image], [prompt], max_new_tokens=max_new_tokens
        )[0]

    def _generate_text_batch(
        self,
        images: Sequence[Any],
        prompts: Sequence[str],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        from PIL import Image

        if not images:
            raise ValueError("images must contain at least one item")
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")
        if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError("prompts must contain non-empty strings")
        arrays = [load_image(image) for image in images]
        texts = [self._prompt(prompt) for prompt in prompts]
        inputs = self.processor(
            text=texts,
            images=[Image.fromarray(array) for array in arrays],
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens
            )
        generated_text = generated
        if "input_ids" in inputs and generated.shape[1] > inputs["input_ids"].shape[1]:
            generated_text = generated[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(generated_text, skip_special_tokens=True)
        if len(decoded) != len(images):
            raise ValueError("VLM processor must return one answer per image")
        return [str(value).strip() for value in decoded]

    def ask(self, image: Any, prompt: str | None = None) -> SemanticResult:
        actual_prompt = prompt or self.default_prompt
        decoded = self._generate_text(image, actual_prompt)
        return SemanticResult(
            caption=decoded,
            scene={"prompt": actual_prompt},
            metadata={"backend": "transformers-vlm", "model_id": self.model_id},
        )

    def ask_batch(
        self, images: Sequence[Any], prompts: Sequence[str] | None = None
    ) -> list[SemanticResult]:
        """Generate ordered answers for a batch of images in one model call."""
        actual_prompts = (
            [self.default_prompt] * len(images) if prompts is None else list(prompts)
        )
        decoded = self._generate_text_batch(images, actual_prompts)
        return [
            SemanticResult(
                caption=answer,
                scene={"prompt": prompt},
                metadata={"backend": "transformers-vlm", "model_id": self.model_id},
            )
            for answer, prompt in zip(decoded, actual_prompts)
        ]

    def extract_batch(self, images: Sequence[Any]) -> list[SemanticResult]:
        """Use the native batched generation path for video and image batches."""
        return self.ask_batch(images)

    def ask_structured(self, image: Any, query: StructuredQuery) -> SemanticResult:
        """Generate and validate a JSON answer for a structured query."""
        if not isinstance(query, StructuredQuery):
            raise TypeError("query must be a StructuredQuery")
        raw = self._generate_text(image, query.prompt, query.max_new_tokens)
        parsed = parse_structured_output(raw, query.schema)
        return SemanticResult(
            caption=raw,
            scene=parsed,
            metadata={
                "backend": "transformers-vlm",
                "model_id": self.model_id,
                "structured": True,
                "query": query.to_dict(),
            },
        )

    def ask_structured_batch(
        self, images: Sequence[Any], query: StructuredQuery
    ) -> list[SemanticResult]:
        """Generate and validate one structured answer per image."""
        if not isinstance(query, StructuredQuery):
            raise TypeError("query must be a StructuredQuery")
        raw_answers = self._generate_text_batch(
            images, [query.prompt] * len(images), query.max_new_tokens
        )
        return [
            SemanticResult(
                caption=raw,
                scene=parse_structured_output(raw, query.schema),
                metadata={
                    "backend": "transformers-vlm",
                    "model_id": self.model_id,
                    "structured": True,
                    "query": query.to_dict(),
                },
            )
            for raw in raw_answers
        ]

    def extract(self, image: Any) -> SemanticResult:
        return self.ask(image)


class RFDETRExtractor:
    """Adapter for the optional Apache-licensed RF-DETR package.

    The adapter accepts an injected model for tests and deployments that
    manage model construction themselves. When no model is injected it uses
    the package's ``RFDETRBase`` class and never downloads anything through
    Yase itself; model loading remains the application's responsibility.
    """

    def __init__(
        self,
        model: Any | None = None,
        threshold: float = 0.5,
        model_variant: str = "base",
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if model is None:
            try:
                import rfdetr
            except ImportError as exc:
                raise ImportError(
                    "install RF-DETR separately to use RFDETRExtractor"
                ) from exc
            class_name = "RFDETR" + model_variant.capitalize()
            constructor = getattr(rfdetr, class_name, None)
            if constructor is None:
                constructor = getattr(rfdetr, "RFDETRBase", None)
            if constructor is None:
                raise ImportError("installed rfdetr package exposes no RF-DETR model")
            model = constructor()
        if not hasattr(model, "predict"):
            raise TypeError("RF-DETR model must expose predict")
        self.model = model
        self.threshold = threshold
        self.model_variant = model_variant

    def extract(self, image: Any) -> SemanticResult:
        rgb = load_image(image)
        raw = self.model.predict(rgb, threshold=self.threshold)
        boxes = getattr(raw, "xyxy", getattr(raw, "boxes", []))
        scores = getattr(raw, "confidence", getattr(raw, "scores", []))
        class_ids = getattr(raw, "class_id", getattr(raw, "labels", []))
        names = getattr(raw, "class_name", None)
        if names is None and hasattr(raw, "data"):
            names = raw.data.get("class_name")
        names = list(names) if names is not None else []
        detections = []
        for index, (box, score) in enumerate(zip(boxes, scores)):
            values = (
                box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            )
            score_value = float(score.item() if hasattr(score, "item") else score)
            class_id = class_ids[index] if index < len(class_ids) else index
            if hasattr(class_id, "item"):
                class_id = class_id.item()
            label = names[index] if index < len(names) else str(class_id)
            detections.append(
                Detection(
                    str(label),
                    min(1.0, max(0.0, score_value)),
                    BoundingBox.from_sequence(values),
                )
            )
        return SemanticResult(
            detections=detections,
            metadata={
                "backend": "rf-detr",
                "model_variant": self.model_variant,
                "threshold": self.threshold,
            },
        )


class PromptableSegmentationExtractor:
    """Normalize a SAM2/SAM3-style promptable predictor.

    The predictor may expose ``segment(image, prompt=...)`` or
    ``predict(image, prompt=...)``. It can be a native SAM predictor, a remote
    service client, or a custom implementation; no SAM license or runtime is
    imposed on Yase's core package.
    """

    def __init__(self, predictor: Any, prompt: Any | None = None) -> None:
        if not hasattr(predictor, "segment") and not hasattr(predictor, "predict"):
            raise TypeError("predictor must expose segment or predict")
        self.predictor = predictor
        self.prompt = prompt

    def extract(self, image: Any, prompt: Any | None = None) -> SemanticResult:
        prompt = self.prompt if prompt is None else prompt
        if prompt is None:
            raise ValueError("a prompt is required for promptable segmentation")
        rgb = load_image(image)
        if hasattr(self.predictor, "segment"):
            raw = self.predictor.segment(rgb, prompt=prompt)
        else:
            raw = self.predictor.predict(rgb, prompt=prompt)
        if isinstance(raw, SemanticResult):
            return raw
        if isinstance(raw, Mapping):
            excluded = {"segmentation", "masks", "mask", "detections"}
            return SemanticResult(
                segmentation=raw.get("segmentation", raw.get("masks", raw.get("mask"))),
                detections=raw.get("detections"),
                metadata={
                    "backend": "promptable-segmentation",
                    "prompt": prompt,
                    **{key: value for key, value in raw.items() if key not in excluded},
                },
            )
        return SemanticResult(
            segmentation=np.asarray(raw),
            metadata={"backend": "promptable-segmentation", "prompt": prompt},
        )


class TransformersSAM3Extractor:
    """Run official SAM3 image concept segmentation through Transformers.

    The implementation follows the public ``Sam3Processor`` /
    ``Sam3Model`` contract. ``processor`` and ``model`` can be injected for
    testing or for applications that manage model loading and licensing
    themselves. The default path is local-only and never downloads weights
    unless ``local_files_only=False`` is explicitly selected.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        prompt: str | None = None,
        device: str | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        local_files_only: bool = True,
        processor: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if not model_id and model is None:
            raise ValueError("model_id is required")
        if not 0 <= threshold <= 1 or not 0 <= mask_threshold <= 1:
            raise ValueError("thresholds must be in [0, 1]")
        if processor is None or model is None:
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "install the transformers extra to use TransformersSAM3Extractor"
                ) from exc
            options = {"local_files_only": local_files_only}
            processor = processor or AutoProcessor.from_pretrained(model_id, **options)
            model = model or AutoModel.from_pretrained(model_id, **options)
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise ImportError(
                    "install the transformers extra to use TransformersSAM3Extractor"
                ) from exc
        self._torch = torch_module
        self.model_id = model_id
        self.prompt = prompt
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.device = torch_module.device(
            device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        self.processor = processor
        self.model = model.to(self.device).eval() if hasattr(model, "to") else model

    def extract(self, image: Any, prompt: str | None = None) -> SemanticResult:
        from PIL import Image

        text = self.prompt if prompt is None else prompt
        if not text:
            raise ValueError("a text prompt is required for SAM3 extraction")
        rgb = load_image(image)
        inputs = self.processor(
            images=Image.fromarray(rgb), text=text, return_tensors="pt"
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, Mapping):
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        if not hasattr(self.processor, "post_process_instance_segmentation"):
            raise RuntimeError("processor does not support SAM3 postprocessing")
        original_sizes = (
            inputs.get("original_sizes")
            if isinstance(inputs, Mapping)
            else getattr(inputs, "original_sizes", None)
        )
        if hasattr(original_sizes, "tolist"):
            original_sizes = original_sizes.tolist()
        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=original_sizes or [[rgb.shape[0], rgb.shape[1]]],
        )[0]
        masks = processed.get("masks", [])
        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        masks_array = self._array(masks)
        if masks_array.ndim == 4 and masks_array.shape[0] == 1:
            masks_array = masks_array[0]
        detections = []
        for index in range(len(masks_array) if masks_array.ndim == 3 else 0):
            mask = masks_array[index].astype(bool, copy=False)
            if index < len(boxes):
                box_values = self._array(boxes[index]).reshape(-1).tolist()
            else:
                ys, xs = np.where(mask)
                box_values = (
                    [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    ]
                    if len(xs)
                    else [0.0, 0.0, 0.0, 0.0]
                )
            raw_score = scores[index] if index < len(scores) else 1.0
            score = float(raw_score.item() if hasattr(raw_score, "item") else raw_score)
            detections.append(
                Detection(
                    label=text,
                    score=min(1.0, max(0.0, score)),
                    box=BoundingBox.from_sequence(box_values),
                    mask=mask,
                )
            )
        return SemanticResult(
            segmentation=masks_array,
            detections=detections,
            metadata={
                "backend": "transformers-sam3",
                "model_id": self.model_id,
                "prompt": text,
                "threshold": self.threshold,
                "mask_threshold": self.mask_threshold,
            },
        )

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)


class TransformersSAM3VideoExtractor:
    """Run SAM3 promptable concept segmentation over a preloaded video.

    ``extract_video`` returns one ``SemanticResult`` per emitted frame and
    preserves SAM3 object IDs as ``Detection.track_id``. Preloaded mode is
    intentionally separate from :class:`VideoStream`: SAM3 can use future
    frames for confirmation and therefore offers better offline quality than
    a strictly causal realtime loop.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        prompt: str | None = None,
        device: str | None = None,
        local_files_only: bool = True,
        processing_device: str = "cpu",
        model: Any | None = None,
        processor: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if not model_id and model is None:
            raise ValueError("model_id is required")
        if model is None or processor is None:
            try:
                from transformers import Sam3VideoModel, Sam3VideoProcessor
            except ImportError as exc:
                raise ImportError(
                    "install a recent transformers extra for SAM3 video support"
                ) from exc
            options = {"local_files_only": local_files_only}
            processor = processor or Sam3VideoProcessor.from_pretrained(
                model_id, **options
            )
            model = model or Sam3VideoModel.from_pretrained(model_id, **options)
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise ImportError(
                    "install the transformers extra to use SAM3 video"
                ) from exc
        self._torch = torch_module
        self.model_id = model_id
        self.prompt = prompt
        self.device = torch_module.device(
            device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        self.processing_device = processing_device
        self.model = model.to(self.device) if hasattr(model, "to") else model
        self.processor = processor

    def extract_video(
        self,
        frames: Any,
        prompt: str | None = None,
        max_frames: int | None = None,
    ) -> list[SemanticResult]:
        from PIL import Image

        text = self.prompt if prompt is None else prompt
        if not text:
            raise ValueError("a text prompt is required for SAM3 video")
        arrays = [load_image(frame) for frame in frames]
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be positive")
            arrays = arrays[:max_frames]
        if not arrays:
            return []
        images = [Image.fromarray(array) for array in arrays]
        session = self.processor.init_video_session(
            video=images,
            inference_device=self.device,
            processing_device=self.processing_device,
            video_storage_device=self.processing_device,
        )
        session = self.processor.add_text_prompt(inference_session=session, text=text)
        results: list[SemanticResult] = []
        for output in self.model.propagate_in_video_iterator(
            inference_session=session, max_frame_num_to_track=len(images)
        ):
            processed = self.processor.postprocess_outputs(session, output)
            results.append(self._result_from_output(processed, text))
        return results

    def _result_from_output(
        self, processed: Mapping[str, Any], prompt: str
    ) -> SemanticResult:
        masks = self._array(processed.get("masks", []))
        boxes = self._array(processed.get("boxes", []))
        scores = self._array(processed.get("scores", []))
        object_ids = self._array(processed.get("object_ids", []))
        detections = []
        for index in range(len(boxes)):
            box = boxes[index].reshape(-1).tolist()
            score = float(scores[index]) if index < len(scores) else 1.0
            track_id = int(object_ids[index]) if index < len(object_ids) else None
            mask = masks[index] if index < len(masks) else None
            detections.append(
                Detection(
                    label=prompt,
                    score=min(1.0, max(0.0, score)),
                    box=BoundingBox.from_sequence(box),
                    mask=np.asarray(mask).astype(bool) if mask is not None else None,
                    track_id=track_id,
                )
            )
        frame_index = processed.get("frame_idx")
        if hasattr(frame_index, "item"):
            frame_index = frame_index.item()
        return SemanticResult(
            segmentation=masks,
            detections=detections,
            metadata={
                "backend": "transformers-sam3-video",
                "model_id": self.model_id,
                "prompt": prompt,
                "frame_index": frame_index,
            },
        )

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)


__all__ = [
    "PaddleOCRExtractor",
    "PromptableSegmentationExtractor",
    "TransformersSAM3Extractor",
    "TransformersSAM3VideoExtractor",
    "RFDETRExtractor",
    "TesseractExtractor",
    "TransformersImageEmbeddingExtractor",
    "TransformersGroundingDinoExtractor",
    "TransformersObjectDetectionExtractor",
    "TransformersVLMExtractor",
]
