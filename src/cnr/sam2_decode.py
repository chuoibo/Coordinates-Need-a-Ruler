# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Turning the emitted geometry into the mask the benchmark scores (Section 3.7).

SAM 2.1 ``hiera-tiny`` consumes the model's box and clicks and returns a binary
mask. **This stage is an adapter to the benchmark's output format, not a
contribution**: any promptable segmenter would serve. The decoder is fine-tuned
once on the training masks with the image encoder frozen, then frozen itself, so
every later difference in mask quality comes from the prompts rather than from
the segmenter.

Prompt assembly: positive clicks get label 1, negative clicks label 0, and the
box is passed alongside. When ``multimask_output`` is on, the highest-scoring
of the returned masks is taken; the reported run leaves it off, which makes the
decode deterministic given the prompt.

.. warning::
   :func:`load_image` does **not** apply EXIF orientation, and that is
   deliberate: it reproduces the reported numbers. The language-model stage
   *does* apply it, so on the 205 of 2,373 test items whose stored and upright
   dimensions differ, the decoder is prompted in the rotated frame. The system
   averages 66.34 mask IoU on those against 76.59 elsewhere, so the published
   75.70 is an **underestimate**. Pass ``exif_transpose=True`` to fix the frame;
   the result is then no longer comparable to the published number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps

from .geometry import bbox_1000_to_pixels, point_1000_to_pixel

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_MODEL_CFG",
    "Sam2Decoder",
    "SamPrompt",
    "load_image",
    "prompt_from_prediction",
]

DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CHECKPOINT = "checkpoints/vizwiz_hiera_tiny_finetune/checkpoint.pt"


@dataclass(frozen=True)
class SamPrompt:
    """One decode prompt, in the model's ``[0, 1000]`` frame."""

    image_id: str
    bbox_1000: list[int]
    positive_points_1000: list[list[int]]
    negative_points_1000: list[list[int]]

    def is_usable(self) -> bool:
        """A prompt with no box and no positive click cannot produce a mask."""
        return bool(self.bbox_1000) or bool(self.positive_points_1000)


def prompt_from_prediction(record: dict[str, Any]) -> SamPrompt | None:
    """Build a prompt from one row of ``detailed_answers.json``.

    Returns ``None`` when the generation produced no usable geometry. The caller
    must score that item **zero**, not drop it -- one test item does exactly
    this, and dropping it inflates the reported mask IoU from 75.70 to 75.73.
    """
    bbox = record.get("bbox_1000") or []
    positives = record.get("positive_points_1000") or []
    negatives = record.get("negative_points_1000") or []
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = []
    prompt = SamPrompt(
        image_id=record["image_id"],
        bbox_1000=[int(v) for v in bbox] if bbox else [],
        positive_points_1000=[[int(p[0]), int(p[1])] for p in positives if len(p) >= 2],
        negative_points_1000=[[int(p[0]), int(p[1])] for p in negatives if len(p) >= 2],
    )
    return prompt if prompt.is_usable() else None


def load_image(path: Path, *, exif_transpose: bool = False) -> Image.Image:
    """Open an image for the decoder. See the module-level EXIF warning."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im) if exif_transpose else im
        return im.convert("RGB")


def _pixel_points(points: Sequence[Sequence[int]], width: int, height: int) -> np.ndarray:
    coords = [list(point_1000_to_pixel(p, width, height)) for p in points if len(p) >= 2]
    return np.asarray(coords, dtype=np.float32)


class Sam2Decoder:
    """Thin wrapper over ``SAM2ImagePredictor``.

    SAM 2 is imported lazily so that the rest of this package -- the loss, the
    tokenisation, the dataset builder, the metrics -- stays importable and
    testable on a machine with no segmentation stack installed.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        model_cfg: str = DEFAULT_MODEL_CFG,
        device: str = "cuda",
        *,
        multimask_output: bool = False,
    ) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint}")
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, str(checkpoint), device=device))
        self.multimask_output = multimask_output

    def decode(
        self,
        image: Image.Image,
        prompt: SamPrompt,
        *,
        include_box: bool = True,
        include_negative_points: bool = True,
    ) -> np.ndarray:
        """Return a boolean mask of ``image``'s size for one prompt."""
        width, height = image.size
        self.predictor.set_image(image)

        positive = _pixel_points(prompt.positive_points_1000, width, height)
        negative = (
            _pixel_points(prompt.negative_points_1000, width, height)
            if include_negative_points
            else np.zeros((0, 2), dtype=np.float32)
        )
        if len(positive) and len(negative):
            coords = np.concatenate([positive, negative], axis=0)
            labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32)
        elif len(positive):
            coords, labels = positive, np.ones(len(positive), dtype=np.int32)
        elif len(negative):
            coords, labels = negative, np.zeros(len(negative), dtype=np.int32)
        else:
            coords = labels = None

        box = None
        if include_box and prompt.bbox_1000:
            pixels = bbox_1000_to_pixels(prompt.bbox_1000, width, height)
            if pixels is not None:
                box = np.asarray(pixels, dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box,
            multimask_output=self.multimask_output,
        )
        best = int(np.argmax(scores)) if len(scores) else 0
        return masks[best] > 0


def mask_to_png_array(mask: np.ndarray) -> np.ndarray:
    """Boolean mask -> the submission format: ``uint8`` in ``{0, 255}``."""
    return (mask.astype(np.uint8) * 255)
