# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""The two numbers this repository reports.

**Mask IoU** is the benchmark's own metric: mean intersection-over-union between
predicted and ground-truth binary masks, over every test item. It is what
:data:`RESULTS` calls 75.70.

**Bounding-box IoU** is the localisation the coordinate objective actually
optimises: the emitted box against the tight box of the ground-truth mask. It
runs about five points above the mask metric, so the two are never
interchangeable -- a delta on one does not convert into a statement about the
other.

Scoring rules that matter for reproducing the headline number:

* An item whose generation yields no usable geometry is **scored zero, not
  dropped**. One test item does this. Dropping it instead reports 75.73.
* The submission format is enforced by default: single-channel 8-bit PNG with
  pixel values in ``{0, 255}``. ``strict_format=False`` relaxes it to ``> 0``
  for debugging only.
* Empty-vs-empty masks score 1.0, which is the convention the official scorer
  uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .geometry import bbox_1000_to_pixels, bbox_iou, mask_to_bbox

__all__ = [
    "RESULTS",
    "BboxEvaluation",
    "MaskFormatError",
    "MaskEvaluation",
    "PairIoU",
    "evaluate_mask_directories",
    "expected_png_name",
    "gt_box_from_mask_png",
    "load_mask_png",
    "score_bbox_predictions",
]

#: Published numbers for the released checkpoint, for regression checks.
RESULTS = {
    "mask_iou": 0.7570,        # 2,373 items, the one failure scored zero
    "mask_iou_gold_prompt": 0.8760,  # same decoder prompted from ground truth
    "bbox_iou": 0.8090,
    "bbox_iou_small": 0.6279,
    "bbox_iou_medium": 0.7640,
    "bbox_iou_large": 0.9176,
    "n_test": 2373,
}


class MaskFormatError(ValueError):
    """Raised when a mask violates the submission format."""


@dataclass(frozen=True)
class PairIoU:
    image_id: str
    width: int
    height: int
    intersection: int
    union: int
    pred_area: int
    gt_area: int
    iou: float


@dataclass(frozen=True)
class MaskEvaluation:
    mean_iou: float
    n: int
    per_sample: list[PairIoU]

    def summary(self) -> dict:
        return {"mean_iou": self.mean_iou, "n": self.n}

    def by_bucket(self, bucket_of: dict[str, str]) -> dict[str, dict]:
        """Mean IoU grouped by an ``image_id -> bucket`` table."""
        groups: dict[str, list[float]] = {}
        for row in self.per_sample:
            bucket = bucket_of.get(row.image_id)
            if bucket:
                groups.setdefault(bucket, []).append(row.iou)
        return {k: {"n": len(v), "mean_iou": float(np.mean(v))} for k, v in sorted(groups.items())}


def expected_png_name(image_id: str) -> str:
    """``VizWiz_test_0000123.jpg -> VizWiz_test_0000123.png``."""
    if image_id.endswith(".jpg"):
        return image_id[:-4] + ".png"
    if image_id.endswith(".png"):
        return image_id
    return image_id + ".png"


def load_mask_png(path: Path, *, strict_format: bool = True) -> np.ndarray:
    """Load a mask as ``uint8``, enforcing the submission format by default."""
    path = Path(path)
    if strict_format and path.suffix.lower() != ".png":
        raise MaskFormatError(f"{path} is not a PNG file")
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        if strict_format and im.format != "PNG":
            raise MaskFormatError(f"{path} is not encoded as PNG")
        if strict_format and im.mode not in {"1", "L", "P"}:
            raise MaskFormatError(f"{path} must be single-channel grayscale; got mode {im.mode}")
        arr = np.asarray(im.convert("L"))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if strict_format:
        values = set(np.unique(arr).tolist())
        if not values <= {0, 255}:
            raise MaskFormatError(f"{path} has non-binary values: {sorted(values)[:12]}")
    return arr


def gt_box_from_mask_png(path: Path):
    """Tight box of a ground-truth mask PNG, or ``None`` when it is empty."""
    with Image.open(path) as im:
        return mask_to_bbox(np.asarray(im.convert("L")))


def _pair(pred: np.ndarray, gt: np.ndarray, image_id: str) -> PairIoU:
    if pred.shape != gt.shape:
        raise MaskFormatError(f"{image_id}: mask shape mismatch {pred.shape} vs {gt.shape}")
    pf, gf = pred > 0, gt > 0
    inter = int(np.logical_and(pf, gf).sum())
    union = int(np.logical_or(pf, gf).sum())
    height, width = gt.shape
    return PairIoU(
        image_id=image_id,
        width=int(width),
        height=int(height),
        intersection=inter,
        union=union,
        pred_area=int(pf.sum()),
        gt_area=int(gf.sum()),
        iou=1.0 if union == 0 else inter / union,
    )


def evaluate_mask_directories(
    pred_dir: Path,
    gt_dir: Path,
    *,
    annotation_path: Path | None = None,
    image_ids: Iterable[str] | None = None,
    strict_format: bool = True,
    missing_scores_zero: bool = True,
) -> MaskEvaluation:
    """Mean mask IoU over a fixed evaluation set.

    The evaluation set is the keys of ``annotation_path`` when given, else
    ``image_ids``, else every PNG in ``gt_dir``. Fixing it from the annotations
    is what makes ``missing_scores_zero`` meaningful: a prediction the model
    failed to produce still appears, and still counts.
    """
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    if annotation_path is not None and image_ids is not None:
        raise ValueError("pass either annotation_path or image_ids, not both")
    if annotation_path is not None:
        ids = sorted(json.loads(Path(annotation_path).read_text()).keys())
    elif image_ids is not None:
        ids = sorted(image_ids)
    else:
        ids = sorted(p.name for p in gt_dir.glob("*.png"))

    rows: list[PairIoU] = []
    for image_id in ids:
        png = expected_png_name(image_id)
        gt_path = gt_dir / png
        if not gt_path.exists():
            continue
        gt = load_mask_png(gt_path, strict_format=strict_format)
        pred_path = pred_dir / png
        if not pred_path.exists():
            if not missing_scores_zero:
                continue
            # A generation that produced no usable geometry is a miss, not an
            # absent sample. Score it zero against the real ground truth.
            pred = np.zeros_like(gt)
        else:
            pred = load_mask_png(pred_path, strict_format=strict_format)
        rows.append(_pair(pred, gt, image_id))

    mean_iou = float(np.mean([r.iou for r in rows])) if rows else float("nan")
    return MaskEvaluation(mean_iou=mean_iou, n=len(rows), per_sample=rows)


@dataclass(frozen=True)
class BboxEvaluation:
    mean_iou: float
    n: int
    per_sample: list[dict]
    n_no_geometry: int

    def by_bucket(self, key: str = "size_class") -> dict[str, dict]:
        groups: dict[str, list[float]] = {}
        for row in self.per_sample:
            bucket = row.get(key)
            if bucket:
                groups.setdefault(bucket, []).append(row["bbox_iou"])
        order = {"small": 0, "medium": 1, "large": 2}
        return {
            k: {"n": len(v), "mean_iou": float(np.mean(v))}
            for k, v in sorted(groups.items(), key=lambda kv: order.get(kv[0], 99))
        }


def score_bbox_predictions(
    predictions: Iterable[dict],
    gt_dir: Path,
    *,
    bucket_of: dict[str, str] | None = None,
) -> BboxEvaluation:
    """Bounding-box IoU of each prediction against the tight box of its GT mask.

    This is the quantity the coordinate objective optimises, and it runs about
    five points above the benchmark's mask metric -- so a delta here does not
    convert into a mask-metric claim.

    A prediction with no usable box scores **zero** against a non-empty ground
    truth rather than being skipped.
    """
    gt_dir = Path(gt_dir)
    bucket_of = bucket_of or {}
    rows: list[dict] = []
    no_geometry = 0

    for pred in predictions:
        image_id = pred["image_id"]
        width, height = int(pred["width"]), int(pred["height"])
        pred_box = bbox_1000_to_pixels(pred.get("bbox_1000"), width, height)
        if pred_box is None:
            no_geometry += 1

        gt_png = gt_dir / expected_png_name(image_id)
        gt_box = gt_box_from_mask_png(gt_png) if gt_png.exists() else None

        rows.append(
            {
                "image_id": image_id,
                "size_class": bucket_of.get(image_id, pred.get("size_class")),
                "pred_bbox_xyxy": pred_box,
                "gt_bbox_xyxy": gt_box,
                "bbox_iou": bbox_iou(pred_box, gt_box),
                "parse_error": pred.get("parse_error"),
            }
        )

    mean_iou = float(np.mean([r["bbox_iou"] for r in rows])) if rows else float("nan")
    return BboxEvaluation(
        mean_iou=mean_iou, n=len(rows), per_sample=rows, n_no_geometry=no_geometry
    )
