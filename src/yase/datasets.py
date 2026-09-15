"""Readers and writers for MOTChallenge and COCO-style annotations."""

import csv
import glob as glob_module
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schema import BoundingBox, Detection


@dataclass(frozen=True)
class CocoImage:
    """Minimal image record from a COCO annotation file."""

    image_id: int
    file_name: str
    width: int
    height: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CocoDataset:
    """COCO annotations converted to ordered, typed detection frames."""

    images: tuple[CocoImage, ...]
    categories: Mapping[int, str]
    annotations: Mapping[int, tuple[Detection, ...]]

    @property
    def image_ids(self) -> tuple[int, ...]:
        return tuple(image.image_id for image in self.images)

    @property
    def category_ids(self) -> Mapping[str, int]:
        return {name: category_id for category_id, name in self.categories.items()}

    def frames(self) -> list[list[Detection]]:
        """Return annotations in the same order as ``images``."""
        return [list(self.annotations.get(image.image_id, ())) for image in self.images]

    def image_paths(self, root: str | Path | None = None) -> list[Path]:
        """Resolve image filenames against an optional dataset root."""
        base = Path(root) if root is not None else Path()
        return [base / image.file_name for image in self.images]


def _read_mot_rows(path: str | Path) -> list[list[str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            rows.append([value.strip() for value in row])
    return rows


def load_mot_sequence(
    path: str | Path,
    *,
    label: str = "object",
    score_column: int | None = None,
    min_score: float = 0.0,
) -> list[list[Detection]]:
    """Load MOT rows into frame-indexed ``Detection`` lists.

    MOT uses ``frame,id,x,y,width,height,confidence,...`` with one-based frame
    numbers. ``score_column`` is zero-based and can be set to 6 for detector
    outputs; ground-truth files can leave it unset.
    """
    if not label:
        raise ValueError("label must not be empty")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be in [0, 1]")
    rows = _read_mot_rows(path)
    parsed: dict[int, list[Detection]] = {}
    for row in rows:
        if len(row) < 6:
            raise ValueError("MOT rows require at least six columns")
        frame = int(float(row[0]))
        track_id = int(float(row[1]))
        score = 1.0
        if score_column is not None:
            if score_column >= len(row):
                raise ValueError("score_column is outside the MOT row")
            score = float(row[score_column])
            if score < min_score:
                continue
        detection = Detection(
            label=label,
            score=max(0.0, min(1.0, score)),
            box=BoundingBox.from_xywh(
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ),
            track_id=track_id,
        )
        parsed.setdefault(frame, []).append(detection)
    return [parsed.get(frame, []) for frame in range(1, max(parsed, default=0) + 1)]


def write_mot_sequence(
    frames: Iterable[Iterable[Detection]],
    destination: str | Path,
    *,
    default_confidence: float = 1.0,
) -> int:
    """Write detections to MOT format and return the number of rows written."""
    if not 0 <= default_confidence <= 1:
        raise ValueError("default_confidence must be in [0, 1]")
    count = 0
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for frame_index, detections in enumerate(frames, start=1):
            for detection in detections:
                if not isinstance(detection, Detection):
                    continue
                writer.writerow(
                    [
                        frame_index,
                        detection.track_id if detection.track_id is not None else -1,
                        detection.box.x1,
                        detection.box.y1,
                        detection.box.width,
                        detection.box.height,
                        detection.score
                        if detection.score is not None
                        else default_confidence,
                        -1,
                        -1,
                        -1,
                    ]
                )
                count += 1
    return count


def _polygon_mask(segmentation: Sequence[Any], width: int, height: int) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - Pillow is a core dependency
        raise ImportError("Pillow is required to rasterize COCO polygons") from exc
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    polygons = (
        segmentation
        if segmentation and isinstance(segmentation[0], (list, tuple))
        else [segmentation]
    )
    for polygon in polygons:
        values = [float(value) for value in polygon]
        if len(values) < 6 or len(values) % 2:
            continue
        points = list(zip(values[::2], values[1::2]))
        draw.polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)


def _rle_mask(segmentation: Mapping[str, Any], width: int, height: int) -> np.ndarray:
    size = segmentation.get("size", [height, width])
    if len(size) != 2 or int(size[0]) != height or int(size[1]) != width:
        raise ValueError("COCO RLE size does not match the image dimensions")
    counts = segmentation.get("counts")
    if isinstance(counts, str):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError(
                "compressed COCO RLE requires the optional pycocotools package"
            ) from exc
        decoded = np.asarray(mask_utils.decode(segmentation), dtype=bool)
        return decoded[..., 0] if decoded.ndim == 3 else decoded
    if not isinstance(counts, Sequence):
        raise ValueError("COCO RLE counts must be a list or compressed string")
    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    value = 0
    for count in counts:
        length = int(count)
        if length < 0 or cursor + length > flat.size:
            raise ValueError("invalid COCO RLE run length")
        if value:
            flat[cursor : cursor + length] = 1
        cursor += length
        value = 1 - value
    return flat.reshape((height, width), order="F").astype(bool)


def _decode_coco_segmentation(
    segmentation: Any, width: int, height: int
) -> np.ndarray | None:
    if segmentation is None:
        return None
    if isinstance(segmentation, Mapping):
        return _rle_mask(segmentation, width, height)
    if isinstance(segmentation, Sequence):
        return _polygon_mask(segmentation, width, height)
    raise ValueError("COCO segmentation must be polygons or RLE")


def _annotation_box(
    annotation: Mapping[str, Any], mask: np.ndarray | None
) -> BoundingBox:
    bbox = annotation.get("bbox")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("COCO bbox must contain x, y, width, height")
        return BoundingBox.from_xywh(*[float(value) for value in bbox])
    if mask is None or not np.any(mask):
        raise ValueError("COCO annotation requires bbox or non-empty segmentation")
    ys, xs = np.nonzero(mask)
    return BoundingBox(
        float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
    )


def load_coco_dataset(
    path: str | Path,
    *,
    include_masks: bool = True,
    include_crowd: bool = False,
    min_area: float = 0.0,
) -> CocoDataset:
    """Load COCO instance annotations without requiring ``pycocotools``.

    Polygon masks and uncompressed RLE are handled with NumPy/Pillow. Compressed
    RLE is delegated to ``pycocotools`` only when such an annotation is found.
    Ground-truth scores are set to ``1.0`` and annotation/category IDs are
    retained in detection attributes.
    """
    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("COCO annotation root must be a JSON object")
    image_records = payload.get("images", [])
    category_records = payload.get("categories", [])
    annotation_records = payload.get("annotations", [])
    if not isinstance(image_records, Sequence) or not isinstance(
        annotation_records, Sequence
    ):
        raise ValueError("COCO images and annotations must be arrays")
    images = []
    image_by_id: dict[int, CocoImage] = {}
    for raw in image_records:
        image = CocoImage(
            image_id=int(raw["id"]),
            file_name=str(raw.get("file_name", "")),
            width=int(raw["width"]),
            height=int(raw["height"]),
            metadata={
                key: value
                for key, value in raw.items()
                if key not in {"id", "file_name", "width", "height"}
            },
        )
        images.append(image)
        image_by_id[image.image_id] = image
    categories = {
        int(raw["id"]): str(raw.get("name", raw["id"])) for raw in category_records
    }
    parsed: dict[int, list[Detection]] = {image.image_id: [] for image in images}
    for annotation in annotation_records:
        image_id = int(annotation["image_id"])
        image = image_by_id.get(image_id)
        if image is None:
            raise ValueError(f"COCO annotation references unknown image {image_id}")
        if annotation.get("iscrowd", 0) and not include_crowd:
            continue
        if float(annotation.get("area", 0.0)) < min_area:
            continue
        mask = (
            _decode_coco_segmentation(
                annotation.get("segmentation"), image.width, image.height
            )
            if include_masks
            else None
        )
        category_id = int(annotation["category_id"])
        attributes = {
            "annotation_id": int(annotation.get("id", -1)),
            "category_id": category_id,
            "iscrowd": bool(annotation.get("iscrowd", 0)),
        }
        if "area" in annotation:
            attributes["area"] = float(annotation["area"])
        parsed.setdefault(image_id, []).append(
            Detection(
                label=categories.get(category_id, str(category_id)),
                score=1.0,
                box=_annotation_box(annotation, mask),
                mask=mask,
                attributes=attributes,
            )
        )
    return CocoDataset(
        images=tuple(images),
        categories=categories,
        annotations={key: tuple(value) for key, value in parsed.items()},
    )


def _encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("COCO masks must be two-dimensional")
    flat = values.reshape(-1, order="F").astype(np.uint8)
    counts: list[int] = []
    previous = 0
    run = 0
    for value in flat:
        current = int(value)
        if current == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = current
    counts.append(run)
    return {"size": [int(values.shape[0]), int(values.shape[1])], "counts": counts}


def write_coco_predictions(
    frames: Iterable[Iterable[Detection]],
    destination: str | Path,
    *,
    image_ids: Sequence[int],
    category_ids: Mapping[str, int] | None = None,
    include_masks: bool = False,
) -> int:
    """Write aligned detections as COCO result JSON and return row count."""
    materialized = [list(frame) for frame in frames]
    if len(materialized) != len(image_ids):
        raise ValueError("image_ids must align with prediction frames")
    rows = []
    for image_id, detections in zip(image_ids, materialized):
        for detection in detections:
            if not isinstance(detection, Detection):
                continue
            category_id = detection.attributes.get("category_id")
            if category_id is None and category_ids is not None:
                category_id = category_ids.get(detection.label)
            if category_id is None:
                try:
                    category_id = int(detection.label)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"category_ids is required for label '{detection.label}'"
                    ) from exc
            row: dict[str, Any] = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [
                    detection.box.x1,
                    detection.box.y1,
                    detection.box.width,
                    detection.box.height,
                ],
                "score": detection.score,
            }
            if include_masks and detection.mask is not None:
                row["segmentation"] = _encode_uncompressed_rle(detection.mask)
            rows.append(row)
    with Path(destination).open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return len(rows)


def load_coco_predictions(
    path: str | Path,
    dataset: CocoDataset,
    *,
    include_masks: bool = False,
) -> list[list[Detection]]:
    """Load COCO result JSON into frames aligned with a ``CocoDataset``."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("COCO predictions must be a JSON array")
    images = {image.image_id: image for image in dataset.images}
    parsed: dict[int, list[Detection]] = {
        image.image_id: [] for image in dataset.images
    }
    for index, prediction in enumerate(payload):
        if not isinstance(prediction, Mapping):
            raise ValueError("each COCO prediction must be an object")
        image_id = int(prediction["image_id"])
        image = images.get(image_id)
        if image is None:
            raise ValueError(f"COCO prediction references unknown image {image_id}")
        bbox = prediction.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise ValueError("COCO prediction requires bbox [x, y, width, height]")
        category_id = int(prediction["category_id"])
        mask = (
            _decode_coco_segmentation(
                prediction.get("segmentation"), image.width, image.height
            )
            if include_masks and prediction.get("segmentation") is not None
            else None
        )
        attributes = {"category_id": category_id, "prediction_index": index}
        parsed.setdefault(image_id, []).append(
            Detection(
                label=dataset.categories.get(category_id, str(category_id)),
                score=max(0.0, min(1.0, float(prediction.get("score", 1.0)))),
                box=BoundingBox.from_xywh(*[float(value) for value in bbox]),
                mask=mask,
                attributes=attributes,
            )
        )
    return [parsed.get(image.image_id, []) for image in dataset.images]


def discover_images(
    inputs: Iterable[str | Path],
    *,
    extensions: Sequence[str] | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Expand files, directories, and glob patterns into sorted image paths."""
    allowed = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in (
            extensions
            or (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")
        )
    }
    discovered: set[Path] = set()
    for raw_input in inputs:
        value = str(raw_input)
        matches = (
            [Path(value)]
            if Path(value).exists()
            else [Path(item) for item in glob_module.glob(value)]
        )
        if not matches:
            raise FileNotFoundError(f"image input does not exist: {value}")
        for match in matches:
            if match.is_file():
                if match.suffix.lower() in allowed:
                    discovered.add(match)
                continue
            if match.is_dir():
                iterator = match.rglob("*") if recursive else match.glob("*")
                discovered.update(
                    item
                    for item in iterator
                    if item.is_file() and item.suffix.lower() in allowed
                )
    return sorted(discovered, key=lambda item: str(item))


__all__ = [
    "CocoDataset",
    "CocoImage",
    "discover_images",
    "load_coco_dataset",
    "load_coco_predictions",
    "load_mot_sequence",
    "write_coco_predictions",
    "write_mot_sequence",
]
