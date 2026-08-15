# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Multi-scale crop augmentation (Section 3.6).

After the processor downsamples a photograph, a small evidence region survives
as very few tokens -- detail no coordinate objective can recover, because it is
no longer in the input. So we add records: the same sample re-rendered cropped
around its region, at a zoom chosen from the region's size.

``rho`` (called ``E`` in the original run) is crop edge over region edge, as a
geometric mean over the two axes, so the region covers about ``1 / rho^2`` of
the crop regardless of how big it started::

    small  (area fraction < 0.05)   ->  2 crops, rho = 2.0 and 3.5
    medium (0.05 .. 0.20)           ->  1 crop,  rho = 2.5
    large  (>= 0.20)                ->  none

Three construction constraints, all of which reject rather than fudge:

* the crop side is at least ``1.15x`` the region side, so the region never
  touches the crop border;
* the crop's short side is floored at 320 px, so we do not manufacture blur;
* a crop covering more than 80% of the image is discarded -- it is the original
  image with extra steps.

**The crop preserves the region's relative position.** ``x0 = alpha * (W - w_c)``
with ``alpha = c_x / W`` the region centre's fractional position. That keeps
directional language ("top right", "on the left") true of the crop, which
matters because the supervised text is copied across unchanged. The placement is
then clamped to keep both region and crop in frame; over the 3,209 crops the
clamp moves the centre by a median of zero on both axes, and contracts the mean
per-axis distance from the frame centre from 158.2 to 137.2 units -- the whole
effect falling on regions that hug an edge.

**Round-trip gate.** Every crop's coordinates are mapped back to the full frame
and must land within :data:`GATE_UNITS` of the source coordinates, else the crop
is dropped. Negative points that fall outside the crop are dropped; a crop left
with no negatives whose parent had some is dropped entirely. We never fabricate
a point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CONTAIN_PAD",
    "CROP_PLANS",
    "GATE_UNITS",
    "LARGE_THRESHOLD",
    "MAX_AREA_FRAC",
    "MIN_SIDE_PX",
    "SMALL_THRESHOLD",
    "CropWindow",
    "classify_size",
    "crop_window",
    "to_crop_1000",
    "to_full_1000",
]

CONTAIN_PAD = 1.15  # crop side >= this * region side
MIN_SIDE_PX = 320  # floor on the crop's short side
MAX_AREA_FRAC = 0.8  # discard crops covering more than this fraction
SMALL_THRESHOLD = 0.05
LARGE_THRESHOLD = 0.20
GATE_UNITS = 3  # round-trip tolerance, in [0, 1000] units

CROP_PLANS: dict[str, list[float]] = {"small": [2.0, 3.5], "medium": [2.5], "large": []}


@dataclass(frozen=True)
class CropWindow:
    """A crop in pixels: origin ``(x0, y0)`` and size ``(w, h)``."""

    x0: int
    y0: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """PIL-style ``(left, upper, right, lower)``."""
        return self.x0, self.y0, self.x0 + self.w, self.y0 + self.h


def classify_size(area_fraction: float) -> str:
    """Bucket a region by its area as a fraction of the frame."""
    if area_fraction < SMALL_THRESHOLD:
        return "small"
    if area_fraction < LARGE_THRESHOLD:
        return "medium"
    return "large"


def crop_window(
    width: int,
    height: int,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
    rho: float,
) -> CropWindow | None:
    """The crop for one region at zoom ``rho``, or ``None`` if pointless.

    The crop keeps the image's aspect ratio, so a single scale ``s`` describes
    it. ``s`` is the largest of: the zoom target, the two containment floors,
    and the resolution floor -- taking the max is what makes the constraints
    hard rather than advisory.
    """
    bw, bh = bx2 - bx1, by2 - by1
    if bw <= 0 or bh <= 0:
        return None
    s = max(
        rho * math.sqrt((bw * bh) / (width * height)),
        CONTAIN_PAD * bw / width,
        CONTAIN_PAD * bh / height,
        MIN_SIDE_PX / min(width, height),
    )
    if s >= 1.0 or s * s > MAX_AREA_FRAC:
        return None

    w_c, h_c = s * width, s * height
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2

    def place(centre: float, lo_b: float, hi_b: float, span: float, side: float) -> float:
        # Position-preserving placement, then clamped so both the region and
        # the crop stay in frame. Image bounds always win.
        x0 = (centre / span) * (span - side)
        margin = min(0.02 * side, (side - (hi_b - lo_b)) / 2 * 0.9)
        lo = max(0.0, hi_b + margin - side)
        hi = min(span - side, lo_b - margin)
        if hi < lo:  # region hugs an edge: relax the margin
            lo = max(0.0, hi_b - side)
            hi = min(span - side, lo_b)
        if hi >= lo:
            x0 = min(max(x0, lo), hi)
        return min(max(x0, 0.0), span - side)

    x0 = int(round(place(cx, bx1, bx2, width, w_c)))
    y0 = int(round(place(cy, by1, by2, height, h_c)))
    w = min(max(int(round(w_c)), 2), width - x0)
    h = min(max(int(round(h_c)), 2), height - y0)
    return CropWindow(x0, y0, w, h)


def to_crop_1000(k: int, full_span: float, origin: float, crop_span: float) -> int:
    """Re-express a ``[0, 1000]`` coordinate of the full frame in the crop frame."""
    px = k / 1000.0 * full_span
    return int(round((px - origin) / crop_span * 1000.0))


def to_full_1000(k: int, full_span: float, origin: float, crop_span: float) -> int:
    """Inverse of :func:`to_crop_1000` -- the round-trip gate uses it."""
    px = k / 1000.0 * crop_span + origin
    return int(round(px / full_span * 1000.0))
