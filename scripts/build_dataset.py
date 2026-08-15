#!/usr/bin/env python3
"""Build the base supervised corpus from the VizWiz answer-grounding release.

    python scripts/build_dataset.py \
        --grounding data/train_grounding.json --split train \
        --grounding data/val_grounding.json   --split val \
        --data-root . --out data/vizwiz_grounding/train_base.json

Training and validation are both folded in -- 6,494 + 1,131 = 7,625 records --
because with the ablation rows each being the last checkpoint of a fixed epoch
budget there is nothing to hold out for. Add crops with
``scripts/build_crops.py`` to reach the 10,834 the released checkpoint saw.

For each record the box and the clicks are read off the annotation mask (see
:mod:`cnr.prompts`), and the size tag is the mask's own area fraction. Passing
``--pred-mask-dir`` supplies first-pass decoder masks so the negative click can
be mined from the decoder's own false positives; without it every record gets
the background negative only, and 90.0% of the released corpus's hard negatives
are absent.
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
from cnr.dataset import build_record, load_instruction  # noqa: E402
from cnr.geometry import mask_area_fraction, polygon_to_mask  # noqa: E402
from cnr.prompts import derive_prompt_targets  # noqa: E402


def load_gt_mask(data_root: Path, split: str, image_id: str, entry: dict) -> np.ndarray:
    """Official binary PNG when present, else rasterise the stored polygon."""
    png = data_root / "data" / split / Path(image_id).with_suffix(".png").name
    if png.exists():
        with Image.open(png) as im:
            return np.asarray(im.convert("L")) > 0
    return polygon_to_mask(entry.get("answer_grounding") or [], int(entry["width"]), int(entry["height"])) > 0


def load_pred_mask(pred_dir: Path | None, image_id: str, shape: tuple[int, int]) -> np.ndarray | None:
    if pred_dir is None:
        return None
    png = pred_dir / Path(image_id).with_suffix(".png").name
    if not png.exists():
        return None
    height, width = shape
    with Image.open(png) as im:
        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(im.convert("L")) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grounding", type=Path, action="append", required=True)
    parser.add_argument("--split", action="append", required=True, help="one per --grounding")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--instruction", type=Path, default=Path("configs/instruction.txt"))
    parser.add_argument("--pred-mask-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if len(args.grounding) != len(args.split):
        parser.error("--grounding and --split must be given the same number of times")

    instruction = load_instruction(args.instruction)
    records: list[dict] = []
    skipped: dict[str, int] = {}

    for grounding_path, split in zip(args.grounding, args.split):
        data = json.loads(grounding_path.read_text(encoding="utf-8"))
        items = list(data.items())[: args.limit] if args.limit else list(data.items())
        for image_id, entry in items:
            mask = load_gt_mask(args.data_root, split, image_id, entry)
            if not mask.any():
                skipped["empty_mask"] = skipped.get("empty_mask", 0) + 1
                continue
            pred_mask = load_pred_mask(args.pred_mask_dir, image_id, mask.shape)
            bbox, positives, negatives = derive_prompt_targets(mask, pred_mask)
            if not positives:
                skipped["no_positive_click"] = skipped.get("no_positive_click", 0) + 1
                continue
            records.append(
                build_record(
                    instruction=instruction,
                    question=str(entry["question"]),
                    image_path=f"data/{split}/{image_id}",
                    answer=str(entry["most_common_answer"]),
                    bbox_1000=bbox,
                    positive_points_1000=positives,
                    negative_points_1000=negatives,
                    size_class=classify_size(mask_area_fraction(mask)),
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(records)} records")
    if skipped:
        print("skipped:", skipped)
    n_hard = sum(1 for r in records if r["output"].count("[[") and "negative_points_1000\": []" not in r["output"])
    print(f"records carrying at least one negative click: {n_hard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
