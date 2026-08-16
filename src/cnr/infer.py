# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Test-set generation with the trained checkpoint.

The user turn is the prompt of :mod:`cnr.dataset` followed by the question,
joined exactly as the Alpaca converter joins them during training. Nothing else
is appended.

Generation runs through LlamaFactory's own ``ChatModel`` so that the template,
the tokenisation and the image preprocessing are the ones training used. Four
settings are load-bearing, and each of them silently degrades the result rather
than failing when it is wrong:

``skip_special_tokens=False``
    The coordinate tokens are registered as *special*, so the default decoder
    strips them and every response comes back with empty geometry. This is the
    single most common way to "reproduce" a score of zero.

``image_max_pixels`` and ``cutoff_len``
    Must match training. Raising the pixel budget without raising ``cutoff_len``
    truncates the image tokens, which surfaces as a shape mismatch deep inside
    the model's rotary-position code rather than as a helpful error.

EXIF orientation
    Baked upright with ``exif_transpose`` **before** the processor sees the
    image. The mask stage does not do this (see :mod:`cnr.sam2_decode`), which
    is the frame defect the paper reports.

Greedy decoding
    ``do_sample=False``. The reported numbers are single-sample greedy.

Responses stream to ``responses.jsonl`` as they arrive and ``--resume`` skips
what is already there, because a full test pass is long enough that losing it to
an OOM at item 2,100 is a real cost.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageOps

from .dataset import build_input

__all__ = [
    "GenerationConfig",
    "InferenceItem",
    "generate",
    "items_from_annotations",
    "load_size_classes",
    "parse_responses",
]


@dataclass
class GenerationConfig:
    """Everything that must match training, in one place."""

    model_dir: Path
    template: str = "qwen3_5_nothink"
    image_max_pixels: int = 6_553_600
    cutoff_len: int = 6144
    max_new_tokens: int = 384
    infer_backend: str = "vllm"
    vllm_maxlen: int | None = 8192
    vllm_config: str | None = None
    concurrency: int = 4
    keep_special_tokens: bool = True
    exif_transpose: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def chat_model_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "model_name_or_path": str(self.model_dir),
            "template": self.template,
            "infer_backend": self.infer_backend,
            "trust_remote_code": True,
            "image_max_pixels": self.image_max_pixels,
            "cutoff_len": self.cutoff_len,
        }
        if self.vllm_maxlen is not None:
            args["vllm_maxlen"] = self.vllm_maxlen
        if self.vllm_config is not None:
            args["vllm_config"] = self.vllm_config
        args.update(self.extra)
        return args

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        }
        if self.keep_special_tokens:
            kwargs["skip_special_tokens"] = False
        return kwargs


@dataclass(frozen=True)
class InferenceItem:
    image_id: str
    question: str
    width: int
    height: int
    size_class: str | None = None
    """Reporting bucket for this item. Not part of the prompt."""


def load_size_classes(csv_path: Path) -> dict[str, str]:
    """Read an ``image_id,size_class`` table.

    Used for **reporting only** -- it groups the results by region size. It
    never reaches the model.
    """
    import csv

    with Path(csv_path).open(encoding="utf-8") as handle:
        return {row["image_id"]: row["size_class"] for row in csv.DictReader(handle)}


def items_from_annotations(
    annotations: Path, bucket_by_id: dict[str, str] | None = None
) -> list[InferenceItem]:
    """Read the evaluation set. ``bucket_by_id`` only labels items for reporting."""
    data = json.loads(Path(annotations).read_text(encoding="utf-8"))
    bucket_by_id = bucket_by_id or {}
    return [
        InferenceItem(
            image_id=image_id,
            question=str(entry["question"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
            size_class=bucket_by_id.get(image_id),
        )
        for image_id, entry in data.items()
    ]


def _already_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["image_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def generate(
    config: GenerationConfig,
    instruction: str,
    annotations: Path,
    image_dir: Path,
    out_dir: Path,
    *,
    bucket_by_id: dict[str, str] | None = None,
    limit: int | None = None,
    resume: bool = False,
    failure_limit: int = 50,
    progress: Callable[[int, int], None] | None = None,
    chat_model_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> Path:
    """Generate over the test split; returns the path to ``responses.jsonl``.

    ``chat_model_factory`` exists so tests can drive the loop with a stub; the
    default imports LlamaFactory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.jsonl"

    items = items_from_annotations(annotations, bucket_by_id)
    if limit is not None:
        items = items[:limit]
    done = _already_done(responses_path) if resume else set()
    pending = [item for item in items if item.image_id not in done]

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": {**asdict(config), "model_dir": str(config.model_dir)},
                "instruction_len": len(instruction),
                "n_items": len(items),
                "n_pending": len(pending),
                "reporting_buckets": len(bucket_by_id or {}),
                "decoding": "greedy (do_sample=False)",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if not pending:
        return responses_path

    if chat_model_factory is None:
        # LlamaFactory pins an upper bound on transformers that a current vLLM
        # exceeds; the check is advisory and blocks an otherwise working stack.
        os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
        from llamafactory.chat import ChatModel

        chat_model = ChatModel(config.chat_model_args())
    else:
        chat_model = chat_model_factory(config.chat_model_args())

    gen_kwargs = config.generation_kwargs()
    failures = 0

    def run_one(item: InferenceItem) -> dict[str, Any]:
        started = time.time()
        error: str | None = None
        raw = ""
        try:
            with Image.open(Path(image_dir) / item.image_id) as im:
                image = (ImageOps.exif_transpose(im) if config.exif_transpose else im).convert("RGB")
            content = f"{instruction}\n{build_input(item.question)}"
            responses = chat_model.chat([{"role": "user", "content": content}], images=[image], **gen_kwargs)
            raw = responses[0].response_text if responses else ""
        except Exception as exc:  # noqa: BLE001 - one bad item must not kill the pass
            error = f"chat failed: {exc}"
        return {
            "image_id": item.image_id,
            "question": item.question,
            "width": item.width,
            "height": item.height,
            "size_class": item.size_class,
            "raw_response": raw,
            "api_error": error,
            "elapsed_sec": round(time.time() - started, 3),
        }

    mode = "a" if resume else "w"
    with responses_path.open(mode, encoding="utf-8") as handle:
        completed = 0
        chunk = max(config.concurrency * 2, 8)
        with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
            for start in range(0, len(pending), chunk):
                batch = pending[start : start + chunk]
                for result in [f.result() for f in [pool.submit(run_one, i) for i in batch]]:
                    if result["api_error"]:
                        failures += 1
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed += 1
                    if progress is not None:
                        progress(completed, len(pending))
                if failure_limit and failures >= failure_limit:
                    raise RuntimeError(
                        f"aborting after {failures} chat failures; re-run with resume=True once fixed"
                    )
    return responses_path


def parse_responses(
    responses_path: Path,
    items: Iterable[InferenceItem],
) -> list[dict[str, Any]]:
    """Turn raw responses into per-item prediction records.

    An unparseable response becomes a record with ``bbox_1000 = None``. It stays
    in the list: downstream that becomes a mask of zeros and an IoU of zero,
    which is the honest score for a generation that failed.
    """
    from .dataset import parse_target

    raw_by_id = {}
    with Path(responses_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                raw_by_id[row["image_id"]] = row

    records: list[dict[str, Any]] = []
    for item in items:
        row = raw_by_id.get(item.image_id)
        if row is None:
            continue
        prediction: dict[str, Any] | None = None
        error = row.get("api_error")
        if error is None:
            try:
                prediction = parse_target(row.get("raw_response", ""))
            except ValueError as exc:
                error = str(exc)
        records.append(
            {
                "image_id": item.image_id,
                "question": item.question,
                "width": item.width,
                "height": item.height,
                "size_class": item.size_class,
                "answer": (prediction or {}).get("answer"),
                "bbox_1000": (prediction or {}).get("bbox_1000"),
                "positive_points_1000": (prediction or {}).get("positive_points_1000"),
                "negative_points_1000": (prediction or {}).get("negative_points_1000"),
                "parse_error": error,
                "raw_response": row.get("raw_response", ""),
            }
        )
    return records
