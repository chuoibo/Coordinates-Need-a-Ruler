# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Deriving the supervision targets from a ground-truth mask (Section 3.7).

The model is trained to emit a box and two kinds of click, because that is what
the mask decoder consumes. All three are read off the annotation mask:

* **box** -- the tight axis-aligned box of the foreground, as edge coordinates.
* **two positive clicks** -- the foreground pixel nearest the centroid, and the
  foreground pixel farthest from it. Near-and-far rather than two near clicks:
  a single centroid click under-determines the extent of an elongated region.
* **up to two negative clicks** -- the background pixel nearest the box centre,
  searched inside the box dilated by ``max(8 px, 25%)`` per side and widening to
  the whole frame if that ring is empty; plus, when a first-pass decoder mask is
  supplied, one *hard* negative mined from its false positives inside the box.
  90.0% of the base records carry a hard negative.

Every derived point goes through :func:`safe_normalized_point`, which searches
outward until it finds a pixel that survives the round trip to ``[0, 1000]`` and
back **as the same class**. Quantisation to 1001 bins on a large image can move
a click across a thin boundary; a positive click that decodes onto background is
worse than no click at all.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "derive_prompt_targets",
    "mask_bbox_1000",
    "normalize_pixel_point",
    "safe_normalized_point",
    "select_background_negative",
    "select_hard_negative",
]


def _clamp_1000(value) -> int:
    return max(0, min(1000, int(value)))


def normalize_pixel_point(x: float, y: float, width: int, height: int) -> list[int]:
    """Pixel -> ``[0, 1000]``, addressing the pixel *centre* (hence ``+ 0.5``)."""
    return [
        _clamp_1000(round((x + 0.5) / width * 1000)),
        _clamp_1000(round((y + 0.5) / height * 1000)),
    ]


def _to_pixel(point, width: int, height: int) -> tuple[int, int]:
    x = int(np.floor(point[0] / 1000.0 * width))
    y = int(np.floor(point[1] / 1000.0 * height))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def mask_bbox_1000(x1: int, y1: int, x2_incl: int, y2_incl: int, width: int, height: int) -> list[int]:
    """Inclusive pixel box -> ``[0, 1000]`` **edge** coordinates.

    ``+1`` on the far corners converts the inclusive last-pixel index into the
    edge past it, so the normalised box covers the pixels rather than stopping
    at their centres.
    """
    return [
        _clamp_1000(round(x1 / width * 1000)),
        _clamp_1000(round(y1 / height * 1000)),
        _clamp_1000(round((x2_incl + 1) / width * 1000)),
        _clamp_1000(round((y2_incl + 1) / height * 1000)),
    ]


def safe_normalized_point(mask: np.ndarray, x: int, y: int, *, foreground: bool) -> list[int]:
    """Nearest point to ``(x, y)`` that round-trips back to the intended class."""
    height, width = mask.shape
    for radius in range(0, 10):
        candidates: list[tuple[int, int, int]] = []
        for yy in range(max(0, y - radius), min(height - 1, y + radius) + 1):
            for xx in range(max(0, x - radius), min(width - 1, x + radius) + 1):
                if bool(mask[yy, xx]) != foreground:
                    continue
                candidates.append(((xx - x) ** 2 + (yy - y) ** 2, xx, yy))
        for _, xx, yy in sorted(candidates):
            point = normalize_pixel_point(float(xx), float(yy), width, height)
            px, py = _to_pixel(point, width, height)
            if bool(mask[py, px]) == foreground:
                return point
    return normalize_pixel_point(float(x), float(y), width, height)


def select_background_negative(mask: np.ndarray, *, bbox_xyxy: tuple[int, int, int, int]) -> list[int] | None:
    """Background click nearest the box centre, inside the dilated box."""
    height, width = mask.shape
    x1, y1, x2, y2 = bbox_xyxy
    bg = ~mask
    if not bool(bg.any()):
        return None

    pad_x = max(8, int(round(max(1, x2 - x1 + 1) * 0.25)))
    pad_y = max(8, int(round(max(1, y2 - y1 + 1) * 0.25)))
    ex1, ey1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    ex2, ey2 = min(width - 1, x2 + pad_x), min(height - 1, y2 + pad_y)
    region = bg[ey1 : ey2 + 1, ex1 : ex2 + 1]
    if not bool(region.any()):  # the dilated box is all foreground: widen to the frame
        ey1, ex1 = 0, 0
        region = bg

    ys, xs = np.nonzero(region)
    xs_abs, ys_abs = xs + ex1, ys + ey1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nearest = int(np.argmin((xs_abs - cx) ** 2 + (ys_abs - cy) ** 2))
    return safe_normalized_point(mask, int(xs_abs[nearest]), int(ys_abs[nearest]), foreground=False)


def select_hard_negative(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None,
    *,
    bbox_xyxy: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    """A negative click mined from the decoder's own false positives.

    Restricted to false positives *inside* the evidence box: those are the
    places the decoder over-segments when prompted well, which is exactly what a
    negative click is for. Picks the largest connected component and returns the
    pixel of it nearest its own centroid, so the click sits in the middle of the
    blob rather than on its fringe.
    """
    if pred_mask is None:
        return None
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"predicted mask shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")

    x1, y1, x2, y2 = bbox_xyxy
    inside = np.zeros_like(gt_mask, dtype=bool)
    inside[y1 : y2 + 1, x1 : x2 + 1] = True
    false_positive = (pred_mask & ~gt_mask) & inside
    if not bool(false_positive.any()):
        return None

    crop = false_positive[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        ys, xs = np.nonzero(crop)
        idx = len(xs) // 2
        return int(xs[idx] + x1), int(ys[idx] + y1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n_labels <= 1:
        ys, xs = np.nonzero(crop)
        idx = len(xs) // 2
        return int(xs[idx] + x1), int(ys[idx] + y1)
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    cx, cy = centroids[largest]
    comp_ys, comp_xs = np.nonzero(labels == largest)
    nearest = int(np.argmin((comp_xs - cx) ** 2 + (comp_ys - cy) ** 2))
    return int(comp_xs[nearest] + x1), int(comp_ys[nearest] + y1)


def derive_prompt_targets(
    mask: np.ndarray,
    pred_mask: np.ndarray | None = None,
    *,
    keep_background_negative_with_hard: bool = False,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Return ``(bbox_1000, positive_points_1000, negative_points_1000)``.

    ``mask`` is boolean. ``pred_mask`` is an optional first-pass decoder mask
    used to mine the hard negative; without it the record gets the background
    negative only.
    """
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0], [], []

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = mask_bbox_1000(x1, y1, x2, y2, width, height)

    # Positive 1: the centroid, snapped onto foreground when the region is
    # concave enough that its centroid falls outside it.
    p1x, p1y = int(round(float(xs.mean()))), int(round(float(ys.mean())))
    if not mask[p1y, p1x]:
        nearest = int(np.argmin((xs - p1x) ** 2 + (ys - p1y) ** 2))
        p1x, p1y = int(xs[nearest]), int(ys[nearest])

    # Positive 2: the farthest foreground pixel from positive 1. Subsampled on
    # huge regions -- the argmax is stable and the full scan is not worth it.
    step = max(1, len(xs) // 20_000)
    sx, sy = xs[::step], ys[::step]
    far = int(np.argmax((sx - p1x) ** 2 + (sy - p1y) ** 2))

    positives = [
        safe_normalized_point(mask, p1x, p1y, foreground=True),
        safe_normalized_point(mask, int(sx[far]), int(sy[far]), foreground=True),
    ]

    if not bool((~mask).any()):
        return bbox, positives, []

    hard_pixel = select_hard_negative(mask, pred_mask, bbox_xyxy=(x1, y1, x2, y2))
    hard = (
        safe_normalized_point(mask, hard_pixel[0], hard_pixel[1], foreground=False)
        if hard_pixel is not None
        else None
    )
    if hard is not None and not keep_background_negative_with_hard:
        return bbox, positives, [hard]

    negatives: list[list[int]] = []
    for point in (select_background_negative(mask, bbox_xyxy=(x1, y1, x2, y2)), hard):
        if point is not None and point not in negatives:
            negatives.append(point)
    return bbox, positives, negatives
