# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Polygon rasterisation, boxes, and the coordinate <-> pixel maps.

The rasteriser here is the single source of truth for polygon -> binary mask, so
that the masks we score and the masks we submit are produced by the same code.

Box convention throughout: ``(x1, y1, x2, y2)`` in pixels with ``x2``/``y2``
**exclusive** -- one past the last foreground pixel. :func:`mask_to_bbox` and
:func:`bbox_1000_to_pixels` both produce that form, so their IoU is comparable
without an off-by-one.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from skimage.draw import polygon as sk_polygon

__all__ = [
    "Box",
    "bbox_1000_to_pixels",
    "bbox_area",
    "bbox_intersection",
    "bbox_iou",
    "bbox_rect_mask",
    "mask_area_fraction",
    "mask_iou",
    "mask_to_bbox",
    "point_1000_to_pixel",
    "polygon_to_mask",
    "polygon_xy",
]

Box = tuple[int, int, int, int]


def polygon_xy(points: Sequence[dict]) -> np.ndarray:
    """``[{'x':..,'y':..}, ...] -> (N, 2)`` float array of ``(x, y)``."""
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray([(float(p["x"]), float(p["y"])) for p in points], dtype=np.float64)


def polygon_to_mask(points: Sequence[dict], width: int, height: int) -> np.ndarray:
    """Canonical polygon -> ``uint8`` mask in ``{0, 255}``.

    Coordinates are clipped to the frame before rasterising: worker-drawn
    VizWiz polygons routinely fall a few pixels outside it.
    """
    xy = polygon_xy(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(xy) < 3:
        return mask
    x = np.clip(xy[:, 0], 0, width - 1)
    y = np.clip(xy[:, 1], 0, height - 1)
    rr, cc = sk_polygon(y, x, shape=(height, width))
    mask[rr, cc] = 255
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks; any nonzero pixel counts as foreground."""
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    af, bf = a > 0, b > 0
    inter = int(np.logical_and(af, bf).sum())
    union = int(np.logical_or(af, bf).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def mask_area_fraction(mask: np.ndarray) -> float:
    """Foreground pixels as a fraction of the frame -- the bucketing quantity."""
    total = mask.shape[0] * mask.shape[1]
    return float((mask > 0).sum()) / total if total else 0.0


def mask_to_bbox(mask: np.ndarray) -> Box | None:
    """Tight box of the foreground, ``x2``/``y2`` exclusive. ``None`` if empty."""
    fg = mask > 0
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_area(box: Box | None) -> int:
    if box is None:
        return 0
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_intersection(a: Box | None, b: Box | None) -> int:
    if a is None or b is None:
        return 0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(pred: Box | None, gt: Box | None) -> float:
    """IoU of two boxes. Both missing -> 1.0; exactly one missing -> 0.0."""
    if pred is None and gt is None:
        return 1.0
    if pred is None or gt is None:
        return 0.0
    inter = bbox_intersection(pred, gt)
    union = bbox_area(pred) + bbox_area(gt) - inter
    if union <= 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def _clamp_1000(value) -> int:
    return max(0, min(1000, int(round(float(value)))))


def bbox_1000_to_pixels(raw, width: int, height: int) -> Box | None:
    """Decode ``[x1, y1, x2, y2]`` in ``[0, 1000]`` to an exclusive pixel box.

    Reversed corners are swapped rather than rejected -- the model does emit
    them occasionally, and the intended rectangle is unambiguous. A box that
    collapses to zero width or height *is* rejected, and the caller scores it
    as a miss rather than dropping the item.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1n, y1n, x2n, y2n = (_clamp_1000(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if x2n < x1n:
        x1n, x2n = x2n, x1n
    if y2n < y1n:
        y1n, y2n = y2n, y1n
    x1 = max(0, min(width, int(round(x1n / 1000.0 * width))))
    y1 = max(0, min(height, int(round(y1n / 1000.0 * height))))
    x2 = max(0, min(width, int(round(x2n / 1000.0 * width))))
    y2 = max(0, min(height, int(round(y2n / 1000.0 * height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def point_1000_to_pixel(point, width: int, height: int) -> tuple[int, int]:
    """Decode one ``[x, y]`` in ``[0, 1000]`` to a pixel inside the frame.

    Uses ``floor``, not ``round``: it is the decoder half of the ``+0.5``
    pixel-centre convention in :func:`cnr.prompts.normalize_pixel_point`, and
    switching it to rounding shifts clicks by a pixel across the board.

    The pair round-trips to within one pixel, not exactly -- 1001 bins cannot
    address every pixel of a large image, and near a width of 1000 the ``+0.5``
    can push a point into the next bin. That is why the encoder side is
    :func:`cnr.prompts.safe_normalized_point`, which searches outward until it
    finds a pixel that survives the round trip *as the same class*: a positive
    click that decodes onto background is a prompt that actively misleads SAM 2.
    """
    x = int(_clamp_1000(point[0]) / 1000.0 * width)
    y = int(_clamp_1000(point[1]) / 1000.0 * height)
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def bbox_rect_mask(bbox_1000, width: int, height: int) -> np.ndarray:
    """Rasterise a ``bbox_1000`` rectangle -- the pre-SAM2 proxy mask.

    Reported as a proxy only. The system's real mask comes from
    :mod:`cnr.sam2_decode`; this rectangle scores several points lower and is
    never the headline number.
    """
    box = bbox_1000_to_pixels(bbox_1000, width, height)
    if box is None:
        return np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = box
    corners = [
        {"x": x1, "y": y1},
        {"x": x2, "y": y1},
        {"x": x2, "y": y2},
        {"x": x1, "y": y2},
    ]
    return polygon_to_mask(corners, width, height)
