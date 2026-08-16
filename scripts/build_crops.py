#!/usr/bin/env python3
"""Add the multi-scale crop records (Section 3.6).

    python scripts/build_crops.py \
        --base data/vizwiz_grounding/train_base.json \
        --size-csv data/vizwiz_grounding/train_base_sizes.csv \
        --data-root . \
        --image-out data/crops/images \
        --out data/vizwiz_grounding/train_ms.json

Reads the base corpus, emits ``rho``-zoomed crops for the small and medium
records, re-expresses every coordinate in the crop frame, and writes base +
crops as one file. The released corpus is 7,625 base + 3,209 crops = 10,834
records.

Which records get cropped comes from ``--size-csv``, the side table
``scripts/build_dataset.py`` writes from the annotation masks. It is metadata
about the corpus, not part of any record: the prompt is identical for every
row, cropped or not.

Images are opened with plain PIL and **no** ``exif_transpose``, which is what
LlamaFactory feeds the model. A source image carrying a non-trivial orientation
tag is therefore a hard error rather than something to silently rotate: the two
stages must agree on the frame, and repointing those records at upright copies
is a data-preparation step, not something to paper over here.

Rejection is the default. A crop is dropped when the box degenerates, when a
positive click leaves the frame, when the round trip drifts more than
``GATE_UNITS``, or when the parent had negative clicks and none survive. We
never invent a replacement point.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnr.crops import (  # noqa: E402
    CROP_PLANS,
    GATE_UNITS,
    classify_size,
    crop_window,
    to_crop_1000,
    to_full_1000,
)
from cnr.dataset import emit_target, parse_target  # noqa: E402

EXIF_ORIENTATION_TAG = 274


def build_one(record: dict, rho: float, data_root: Path, image_out: Path, image_prefix: str):
    """Return ``(new_record, meta)`` or ``(None, reason)``."""
    source = record["images"][0]
    path = data_root / source
    im = Image.open(path)
    if im.getexif().get(EXIF_ORIENTATION_TAG, 1) != 1:
        raise RuntimeError(
            f"{source} carries EXIF orientation. The mask stage and the language-model stage "
            "would disagree on the frame; repoint this record at an upright copy first."
        )
    width, height = im.size

    obj = parse_target(record["output"])
    bbox = obj["bbox_1000"]
    box_px = [bbox[0] / 1000 * width, bbox[1] / 1000 * height, bbox[2] / 1000 * width, bbox[3] / 1000 * height]
    window = crop_window(width, height, *box_px, rho)
    if window is None:
        return None, "skip_covers_full_image"
    x0, y0, w_c, h_c = window.x0, window.y0, window.w, window.h

    new_bbox = [
        to_crop_1000(bbox[0], width, x0, w_c),
        to_crop_1000(bbox[1], height, y0, h_c),
        to_crop_1000(bbox[2], width, x0, w_c),
        to_crop_1000(bbox[3], height, y0, h_c),
    ]
    new_bbox = [min(max(v, 0), 1000) for v in new_bbox]
    if not (new_bbox[2] > new_bbox[0] and new_bbox[3] > new_bbox[1]):
        return None, "skip_degenerate_bbox"

    positives = [
        [to_crop_1000(x, width, x0, w_c), to_crop_1000(y, height, y0, h_c)]
        for x, y in obj["positive_points_1000"]
    ]
    if any(not (0 <= x <= 1000 and 0 <= y <= 1000) for x, y in positives):
        return None, "skip_positive_outside_crop"

    negatives, dropped = [], 0
    for x, y in obj["negative_points_1000"]:
        nx, ny = to_crop_1000(x, width, x0, w_c), to_crop_1000(y, height, y0, h_c)
        if 0 <= nx <= 1000 and 0 <= ny <= 1000:
            negatives.append([nx, ny])
        else:
            dropped += 1
    if obj["negative_points_1000"] and not negatives:
        return None, "skip_no_negative_left"

    drift = 0
    pairs = zip(
        positives + [[new_bbox[0], new_bbox[1]], [new_bbox[2], new_bbox[3]]],
        obj["positive_points_1000"] + [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
    )
    for (kx, ky), (ox, oy) in pairs:
        drift = max(
            drift,
            abs(to_full_1000(kx, width, x0, w_c) - ox),
            abs(to_full_1000(ky, height, y0, h_c) - oy),
        )
    if drift > GATE_UNITS:
        return None, f"skip_roundtrip_drift_{drift}"

    name = f"{Path(source).stem}_ms{int(rho * 10):02d}.jpg"
    image_out.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").crop(window.box).save(image_out / name, quality=95)

    # Recorded in the manifest only: a small region at rho=2.0 covers a medium
    # region's share of its crop, which is what the augmentation is for.
    area_scale = (width * height) / (w_c * h_c)
    old_frac = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / 1_000_000
    new_class = classify_size(min(1.0, old_frac * area_scale))

    new_record = {
        "instruction": record["instruction"],
        "input": record["input"],
        "output": emit_target(obj["answer"], new_bbox, positives, negatives),
        "images": [f"{image_prefix}/{name}"],
    }
    meta = {
        "crop_image": name,
        "parent_image": source,
        "rho": rho,
        "new_class": new_class,
        "full_size": [width, height],
        "crop_xywh": [x0, y0, w_c, h_c],
        "negatives_dropped": dropped,
        "roundtrip_drift": drift,
    }
    return new_record, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--size-csv",
        type=Path,
        required=True,
        help="image_path,size_class table from scripts/build_dataset.py --size-out",
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-out", type=Path, default=Path("data/crops/images"))
    parser.add_argument("--image-prefix", default="data/crops/images", help="path recorded in the record")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    if args.limit:
        base = base[: args.limit]

    with args.size_csv.open(encoding="utf-8") as handle:
        bucket_of = {row["image_path"]: row["size_class"] for row in csv.DictReader(handle)}
    missing = [r["images"][0] for r in base if r["images"][0] not in bucket_of]
    if missing:
        raise SystemExit(
            f"{len(missing)} base records have no entry in {args.size_csv} "
            f"(first: {missing[0]}). Re-run build_dataset.py with --size-out."
        )

    jobs = []
    for record in base:
        for rho in CROP_PLANS[bucket_of[record["images"][0]]]:
            jobs.append((record, rho))

    crops, metas, skips = [], [], Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(
            lambda job: build_one(job[0], job[1], args.data_root, args.image_out, args.image_prefix), jobs
        )
        for record, info in results:
            if record is None:
                skips[info] += 1
            else:
                crops.append(record)
                metas.append(info)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(base + crops, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(metas, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"base {len(base)} + crops {len(crops)} = {len(base) + len(crops)} records -> {args.out}")
    print("crop jobs:", len(jobs), "| skips:", dict(skips))
    print("crop size-class mix:", dict(Counter(m["new_class"] for m in metas)))
    print("max round-trip drift:", max((m["roundtrip_drift"] for m in metas), default=0), f"(gate {GATE_UNITS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
