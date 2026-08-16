# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""The supervised record: prompt in, coordinate tokens out.

Every training record is an Alpaca-style multimodal row::

    {"instruction": <the prompt, identical for every record>,
     "input":       "Question: ...",
     "output":      '{"answer": "...", "bbox_1000": [<coord_412>, ...], ...}',
     "images":      ["data/train/VizWiz_train_00007841.jpg"]}

Two invariants hold the pipeline together, and both are enforced here rather
than trusted:

**One prompt, byte for byte.** The instruction lives in ``configs/instruction.txt``
and is checked against :data:`INSTRUCTION_SHA256` on load. Training and
inference read the same file; there is no second copy of the prompt to drift
out of sync. Inference additionally asserts the dataset it was pointed at holds
exactly one unique instruction, so a mixed corpus fails loudly instead of
silently evaluating the wrong prompt.

**The target is not valid JSON, on purpose.** ``<coord_412>`` is an unquoted
atomic token, which is the entire point: one softmax per coordinate. Both the
builder and the parser round-trip through
:func:`cnr.coord_tokens.decode_coord_tokens`, so nothing downstream ever tries
to ``json.loads`` the raw form.

Field order in the emitted object is fixed -- answer, box, positives, negatives.
The box comes before the points because the points are constrained to lie
inside it, so generating it first gives the later tokens something to condition
on.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .coord_tokens import coord_token, decode_coord_tokens

__all__ = [
    "INSTRUCTION_LENGTH",
    "INSTRUCTION_SHA256",
    "build_input",
    "build_record",
    "emit_target",
    "load_instruction",
    "load_unique_instruction",
    "parse_target",
]

#: sha256 of the shipped prompt. Training and inference read the same bytes.
INSTRUCTION_SHA256 = "b69254da75481739cedcf7aee68102114f25b5481d7490afa5746e97a8df5cde"
INSTRUCTION_LENGTH = 727


def load_instruction(path: Path, *, verify: bool = True) -> str:
    """Read the prompt and (by default) verify it is the one we trained with."""
    text = Path(path).read_text(encoding="utf-8")
    if verify:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != INSTRUCTION_SHA256:
            raise ValueError(
                f"instruction at {path} does not match the shipped prompt.\n"
                f"  expected sha256 {INSTRUCTION_SHA256} (len {INSTRUCTION_LENGTH})\n"
                f"  got      sha256 {digest} (len {len(text)})\n"
                "Editing the prompt is fine, but then the published numbers no longer apply; "
                "pass verify=False to proceed."
            )
    if not text.startswith("<image>"):
        raise ValueError("instruction must begin with the '<image>' placeholder")
    return text


def load_unique_instruction(dataset_json: Path) -> str:
    """Recover the prompt from a built dataset, asserting it is unambiguous."""
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    found = {r["instruction"] for r in data if "instruction" in r}
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one unique instruction in {dataset_json}, found {len(found)}. "
            "Refusing to guess which one the checkpoint was trained with."
        )
    instruction = found.pop()
    if not instruction.startswith("<image>"):
        raise ValueError("instruction does not begin with '<image>'")
    return instruction


def build_input(question: str) -> str:
    """The ``input`` field -- and, at inference, the tail of the user turn.

    The Alpaca converter joins ``instruction`` and ``input`` with a newline, so
    inference must build the user turn the same way or the model sees a prompt
    it was never trained on.
    """
    return f"Question: {question}"


def _points(points: Iterable[Sequence[float]]) -> str:
    return "[" + ", ".join(f"[{coord_token(p[0])}, {coord_token(p[1])}]" for p in points) + "]"


def emit_target(
    answer: str,
    bbox_1000: Sequence[float],
    positive_points_1000: Iterable[Sequence[float]],
    negative_points_1000: Iterable[Sequence[float]],
) -> str:
    """Render the supervised target with coordinates as atomic tokens.

    Hand-rendered rather than ``json.dumps``-ed because the coordinate tokens
    must stay unquoted; ``json.dumps`` would emit ``"<coord_412>"`` and the
    model would learn to produce a quoted string.
    """
    return (
        '{"answer": ' + json.dumps(answer, ensure_ascii=False)
        + ', "bbox_1000": [' + ", ".join(coord_token(v) for v in bbox_1000) + "]"
        + ', "positive_points_1000": ' + _points(positive_points_1000)
        + ', "negative_points_1000": ' + _points(negative_points_1000) + "}"
    )


def parse_target(text: str) -> dict[str, Any]:
    """Parse an emitted target (or a model response) back into plain integers.

    Tolerant of the things a model actually does: a code fence, a leading BOM,
    prose before the object. Scans for the first position that parses as a JSON
    object rather than assuming the response *is* one.
    """
    decoded = decode_coord_tokens(text) or ""
    cleaned = decoded.strip().lstrip("﻿")
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in text")


def build_record(
    instruction: str,
    question: str,
    image_path: str,
    answer: str,
    bbox_1000: Sequence[float],
    positive_points_1000: Iterable[Sequence[float]],
    negative_points_1000: Iterable[Sequence[float]],
) -> dict[str, Any]:
    """Assemble one Alpaca multimodal row."""
    return {
        "instruction": instruction,
        "input": build_input(question),
        "output": emit_target(answer, bbox_1000, positive_points_1000, negative_points_1000),
        "images": [image_path],
    }
