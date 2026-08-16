#!/usr/bin/env python3
"""Generate predictions for the test split with a merged checkpoint.

    python scripts/run_inference.py \
        --model-dir checkpoints/qwen3_5_2b_coord_ntl \
        --annotations data/test_grounding.json --image-dir data/test \
        --out-dir outputs/test --resume

Writes ``responses.jsonl`` (raw, resumable), ``predictions.json`` (parsed) and
``run_config.json``. Scoring is a separate step -- ``scripts/evaluate.py`` --
so a decode can be re-scored without re-running the model.

Defaults reproduce the released run: template ``qwen3_5_nothink``,
``image_max_pixels`` 6553600, ``cutoff_len`` 6144, greedy, and special tokens
kept in the decoded string. That last one is not optional: the coordinate
tokens are special, so dropping it returns responses with no geometry at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnr.dataset import load_instruction, load_unique_instruction  # noqa: E402
from cnr.infer import (  # noqa: E402
    GenerationConfig,
    items_from_annotations,
    generate,
    load_size_classes,
    parse_responses,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="merged full model, not a LoRA adapter")
    parser.add_argument("--annotations", type=Path, default=Path("data/test_grounding.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, default=Path("configs/instruction.txt"))
    parser.add_argument(
        "--instruction-from-dataset",
        type=Path,
        default=None,
        help="recover the prompt from a built dataset instead (asserts it is unique)",
    )
    parser.add_argument(
        "--bucket-csv",
        type=Path,
        default=None,
        help="image_id,size_class table used to label predictions for reporting; never enters the prompt",
    )
    parser.add_argument("--template", default="qwen3_5_nothink")
    parser.add_argument("--image-max-pixels", type=int, default=6_553_600)
    parser.add_argument("--cutoff-len", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--infer-backend", default="vllm", choices=["vllm", "huggingface", "sglang"])
    parser.add_argument("--vllm-maxlen", type=int, default=8192)
    parser.add_argument("--vllm-config", default=None, help='e.g. \'{"mm_processor_cache_gb": 0}\'')
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-exif-transpose", action="store_true")
    args = parser.parse_args()

    if args.instruction_from_dataset:
        instruction = load_unique_instruction(args.instruction_from_dataset)
    else:
        instruction = load_instruction(args.instruction)

    bucket_by_id = load_size_classes(args.bucket_csv) if args.bucket_csv else None
    if bucket_by_id:
        print(f"reporting buckets: {len(bucket_by_id)} items labelled (prompt untouched)")

    config = GenerationConfig(
        model_dir=args.model_dir,
        template=args.template,
        image_max_pixels=args.image_max_pixels,
        cutoff_len=args.cutoff_len,
        max_new_tokens=args.max_new_tokens,
        infer_backend=args.infer_backend,
        vllm_maxlen=args.vllm_maxlen,
        vllm_config=args.vllm_config,
        concurrency=args.concurrency,
        exif_transpose=not args.no_exif_transpose,
    )

    def show(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    responses = generate(
        config,
        instruction,
        args.annotations,
        args.image_dir,
        args.out_dir,
        bucket_by_id=bucket_by_id,
        limit=args.limit,
        resume=args.resume,
        progress=show,
    )

    items = items_from_annotations(args.annotations, bucket_by_id)
    if args.limit:
        items = items[: args.limit]
    records = parse_responses(responses, items)
    out = args.out_dir / "predictions.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = sum(1 for r in records if r["parse_error"])
    print(f"wrote {out}: {len(records)} predictions, {failed} without usable geometry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
