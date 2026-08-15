#!/usr/bin/env python3
"""Turn predicted geometry into the binary masks the benchmark scores.

    python scripts/decode_masks.py \
        --predictions outputs/test/predictions.json \
        --image-dir data/test \
        --sam2-checkpoint checkpoints/vizwiz_hiera_tiny_finetune/checkpoint.pt \
        --out-dir outputs/test/masks

Use ``--prompt-source gold`` with ``--gt-dir`` to prompt the same decoder from
the ground-truth region instead. That is the reference the README quotes at
87.60: it is *not* an optimum, because the decoder was fine-tuned on prompts of
exactly that construction, so the reference is in distribution and the model is
not.

An item whose prediction carries no usable geometry still gets a mask file --
an empty one. Writing it is what keeps the scorer honest: skipping it would
quietly drop the item from the mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnr.metrics import expected_png_name  # noqa: E402
from cnr.prompts import derive_prompt_targets  # noqa: E402
from cnr.sam2_decode import (  # noqa: E402
    DEFAULT_MODEL_CFG,
    Sam2Decoder,
    SamPrompt,
    load_image,
    mask_to_png_array,
    prompt_from_prediction,
)


def gold_prompt(gt_png: Path, image_id: str) -> SamPrompt | None:
    with Image.open(gt_png) as im:
        mask = np.asarray(im.convert("L")) > 0
    if not mask.any():
        return None
    bbox, positives, negatives = derive_prompt_targets(mask)
    return SamPrompt(image_id, bbox, positives, negatives)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-source", choices=["predictions", "gold"], default="predictions")
    parser.add_argument("--gt-dir", type=Path, default=None, help="required for --prompt-source gold")
    parser.add_argument("--multimask-output", action="store_true")
    parser.add_argument("--no-box", action="store_true")
    parser.add_argument("--no-negative-points", action="store_true")
    parser.add_argument(
        "--exif-transpose",
        action="store_true",
        help="fix the rotated-frame defect; the result is then NOT comparable to the published 75.70",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.prompt_source == "gold" and args.gt_dir is None:
        parser.error("--prompt-source gold requires --gt-dir")

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if args.limit:
        predictions = predictions[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decoder = Sam2Decoder(
        args.sam2_checkpoint,
        args.sam2_config,
        device=args.device,
        multimask_output=args.multimask_output,
    )

    written = empty = 0
    results = []
    for i, record in enumerate(predictions, 1):
        image_id = record["image_id"]
        png_name = expected_png_name(image_id)
        if args.prompt_source == "gold":
            gt_png = args.gt_dir / png_name
            prompt = gold_prompt(gt_png, image_id) if gt_png.exists() else None
        else:
            prompt = prompt_from_prediction(record)

        image = load_image(args.image_dir / image_id, exif_transpose=args.exif_transpose)
        width, height = image.size
        if prompt is None:
            mask = np.zeros((height, width), dtype=bool)
            empty += 1
        else:
            mask = decoder.decode(
                image,
                prompt,
                include_box=not args.no_box,
                include_negative_points=not args.no_negative_points,
            )
        Image.fromarray(mask_to_png_array(mask)).save(args.out_dir / png_name)
        written += 1
        results.append({"image_id": image_id, "has_prompt": prompt is not None})
        if i % 100 == 0:
            print(f"  {i}/{len(predictions)}", flush=True)

    (args.out_dir.parent / "decode_summary.json").write_text(
        json.dumps(
            {
                "prompt_source": args.prompt_source,
                "sam2_checkpoint": str(args.sam2_checkpoint),
                "sam2_config": args.sam2_config,
                "multimask_output": args.multimask_output,
                "exif_transpose": args.exif_transpose,
                "n": written,
                "n_empty": empty,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {written} masks to {args.out_dir} ({empty} empty, scored as misses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
