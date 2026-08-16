#!/usr/bin/env python3
"""Derive the ``image_id,size_class`` table from ground-truth masks.

    python scripts/make_size_buckets.py --gt-dir data/gt \
        --annotations data/test_grounding.json --out data/size_class_test.csv

This is **reporting metadata**: it is how the small / medium / large breakdown
in ``scripts/evaluate.py --bucket-csv`` is computed. It is read off the ground
truth, so it says how hard each item was, not what the model was told -- the
prompt is identical for every item and carries none of this.

Buckets are area fractions of the frame: small below 0.05, medium in
[0.05, 0.20), large at or above 0.20 -- the same thresholds the crop plan uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnr.crops import classify_size  # noqa: E402
from cnr.metrics import expected_png_name  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.annotations:
        ids = sorted(json.loads(args.annotations.read_text(encoding="utf-8")).keys())
    else:
        ids = sorted(p.name for p in args.gt_dir.glob("*.png"))

    rows, missing = [], 0
    for image_id in ids:
        png = args.gt_dir / expected_png_name(image_id)
        if not png.exists():
            missing += 1
            continue
        with Image.open(png) as im:
            mask = np.asarray(im.convert("L")) > 0
        fraction = float(mask.sum()) / mask.size if mask.size else 0.0
        rows.append((image_id, classify_size(fraction), f"{fraction:.6f}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "image_id,size_class,mask_area_frac\n" + "".join(f"{a},{b},{c}\n" for a, b, c in rows),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for _, cls, _ in rows:
        counts[cls] = counts.get(cls, 0) + 1
    print(f"wrote {args.out}: {len(rows)} rows ({missing} without a GT mask)")
    print("mix:", {k: counts.get(k, 0) for k in ("small", "medium", "large")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
