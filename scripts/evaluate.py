#!/usr/bin/env python3
"""Score a run: bounding-box IoU, mask IoU, and both by region size.

    # the box metric the coordinate objective optimises
    python scripts/evaluate.py --predictions outputs/test/predictions.json \
        --gt-dir data/gt --bucket-csv data/size_class_test.csv

    # the benchmark's own metric, once masks exist
    python scripts/evaluate.py --predictions outputs/test/predictions.json \
        --gt-dir data/gt --bucket-csv data/size_class_test.csv \
        --mask-dir outputs/test/masks --annotations data/test_grounding.json

The two numbers are not interchangeable: the box proxy runs about five points
above the mask metric, so a delta on one says nothing quantitative about the
other. Both are reported so that neither gets quoted as the other.

``--bucket-csv`` only groups the output. It never reaches a prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnr.infer import load_size_classes  # noqa: E402
from cnr.metrics import RESULTS, evaluate_mask_directories, score_bbox_predictions  # noqa: E402


def _table(title: str, buckets: dict[str, dict]) -> list[str]:
    lines = [f"### {title} by ground-truth region size", "", "| size | n | mean IoU |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v['n']} | {v['mean_iou'] * 100:.2f}% |" for k, v in buckets.items()]
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--bucket-csv", type=Path, default=None, help="reporting buckets only")
    parser.add_argument("--mask-dir", type=Path, default=None, help="decoded masks; enables the mask metric")
    parser.add_argument("--annotations", type=Path, default=None, help="fixes the mask evaluation set")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--lenient", action="store_true", help="skip the submission-format check (debugging only)")
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    buckets = load_size_classes(args.bucket_csv) if args.bucket_csv else None

    bbox = score_bbox_predictions(predictions, args.gt_dir, bucket_of=buckets)
    lines = [
        "# Evaluation", "",
        f"- items scored: **{bbox.n}**",
        f"- predictions with no usable geometry (scored zero): **{bbox.n_no_geometry}**",
        f"- **mean bounding-box IoU: {bbox.mean_iou * 100:.2f}%**"
        f"  _(released checkpoint: {RESULTS['bbox_iou'] * 100:.2f}%)_",
        "",
    ]
    lines += _table("Bounding-box IoU", bbox.by_bucket())

    mask_summary = None
    if args.mask_dir:
        mask = evaluate_mask_directories(
            args.mask_dir,
            args.gt_dir,
            annotation_path=args.annotations,
            image_ids=None if args.annotations else [p["image_id"] for p in predictions],
            strict_format=not args.lenient,
            missing_scores_zero=True,
        )
        lines += [
            f"- **mean mask IoU: {mask.mean_iou * 100:.2f}%** over {mask.n} items"
            f"  _(released checkpoint: {RESULTS['mask_iou'] * 100:.2f}%)_",
            "",
        ]
        if buckets:
            lines += _table("Mask IoU", mask.by_bucket(buckets))
        mask_summary = {"mean_iou": mask.mean_iou, "n": mask.n, "by_bucket": mask.by_bucket(buckets or {})}

    report = "\n".join(lines)
    print(report)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.md").write_text(report + "\n", encoding="utf-8")
        (args.out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "bbox": {
                        "mean_iou": bbox.mean_iou,
                        "n": bbox.n,
                        "n_no_geometry": bbox.n_no_geometry,
                        "by_bucket": bbox.by_bucket(),
                    },
                    "mask": mask_summary,
                    "reference": RESULTS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (args.out_dir / "per_sample_bbox.json").write_text(
            json.dumps(bbox.per_sample, indent=1, default=list), encoding="utf-8"
        )
        print(f"wrote {args.out_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
