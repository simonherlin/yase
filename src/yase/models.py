"""Model cards and license-aware model catalog for Yase applications."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ModelCard:
    """Small, serializable description of an external model artifact."""

    name: str
    task: str
    license: str
    source_url: str
    weights_url: Optional[str] = None
    capabilities: tuple[str, ...] = ()
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.task or not self.license or not self.source_url:
            raise ValueError("name, task, license, and source_url are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelCatalog:
    """Explicit registry for models selected by an application.

    Yase does not download or execute models from this catalog. It exists to
    make provenance, capabilities and redistribution constraints inspectable.
    """

    def __init__(self, cards: Optional[Iterable[ModelCard]] = None) -> None:
        self._cards: dict[str, ModelCard] = {}
        for card in cards or ():
            self.register(card)

    def register(self, card: ModelCard, replace: bool = False) -> ModelCard:
        if card.name in self._cards and not replace:
            raise ValueError(f"model '{card.name}' is already registered")
        self._cards[card.name] = card
        return card

    def get(self, name: str) -> ModelCard:
        try:
            return self._cards[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._cards)) or "none"
            raise KeyError(f"unknown model '{name}'; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._cards))

    def search(self, task: Optional[str] = None) -> tuple[ModelCard, ...]:
        return tuple(
            card for card in self._cards.values() if task is None or card.task == task
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: self._cards[name].to_dict() for name in self.names()}


def default_model_catalog() -> ModelCatalog:
    """Return current, externally hosted model cards without loading weights."""
    return ModelCatalog(
        [
            ModelCard(
                name="rf-detr",
                task="detection",
                license="Apache-2.0 (core/Nano-Large; verify checkpoint)",
                source_url="https://github.com/roboflow/rf-detr",
                capabilities=("real-time", "detection", "instance-segmentation"),
                notes="XL/2XL detection variants have separate licensing.",
            ),
            ModelCard(
                name="sam3",
                task="segmentation",
                license="SAM License",
                source_url="https://github.com/facebookresearch/sam3",
                capabilities=("open-vocabulary", "image", "video", "tracking"),
                notes="Review Meta's usage, redistribution, and trade-control terms.",
            ),
            ModelCard(
                name="sam3-video",
                task="video-segmentation",
                license="SAM License",
                source_url="https://github.com/facebookresearch/sam3",
                capabilities=("video", "open-vocabulary", "tracking"),
                notes="Preloaded video mode can use temporal confirmation.",
            ),
            ModelCard(
                name="grounding-dino",
                task="open-vocabulary-detection",
                license="Apache-2.0",
                source_url="https://github.com/IDEA-Research/GroundingDINO",
                capabilities=("text-prompt", "detection"),
            ),
            ModelCard(
                name="qwen3-vl",
                task="vision-language",
                license="Qwen license; verify checkpoint",
                source_url="https://github.com/QwenLM/Qwen3-VL",
                capabilities=("captioning", "ocr", "reasoning", "video"),
            ),
        ]
    )


__all__ = ["ModelCard", "ModelCatalog", "default_model_catalog"]
