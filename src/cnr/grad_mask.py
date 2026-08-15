# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Gradient-masked embedding training (Section 3.3, Eq. 3).

Resizing the vocabulary to make room for the 1001 coordinate tokens leaves all
~248k *pre-existing* rows trainable. At our data scale (10,834 records) that is
a fast route to forgetting: gradient from the output head reaches every row and
drags the language distribution around. So we mask it::

    g~[i] = g[i] * 1[lo <= i <= hi]                                    (Eq. 3)

Three details that are easy to get wrong, and each of which we hit:

**The quantifier matters.** The base checkpoint *ties* the input embedding and
the output head, but the LoRA adapter wraps them as separate modules, so a hook
on one does not cover the other. :func:`attach_coord_grad_mask` hooks *every*
trainable 2-D parameter whose name marks it as an embedding or head.

**Index by vocabulary row.** Eq. (3) indexes rows of the vocabulary, so the hook
belongs only on parameters whose first axis *is* the vocabulary. On a low-rank
factor of the head it would select nothing and silently discard that gradient.
Our configuration never produces such a parameter, and nothing in Eq. (3)
enforces it, so :func:`attach_coord_grad_mask` refuses any candidate whose first
axis does not match the vocabulary size when one is supplied.

**Verify afterwards.** ``scripts/verify_embedding_freeze.py`` compares the
trained checkpoint against the base model row by row and asserts every row
outside the block is bit-identical. Run it; a hook that silently attached to
nothing looks exactly like a hook that worked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import TrainerCallback as _TrainerCallback
else:  # pragma: no cover - import shim
    _TrainerCallback = object

__all__ = ["EMBEDDING_PARAM_MARKERS", "CoordEmbeddingGradMaskCallback", "attach_coord_grad_mask"]

EMBEDDING_PARAM_MARKERS = ("embed_tokens", "lm_head")


def _make_hook(lo: int, hi: int, cache: dict[int, "torch.Tensor"]):
    def hook(grad: "torch.Tensor") -> "torch.Tensor":
        rows = grad.shape[0]
        mask = cache.get(rows)
        if mask is None:
            mask = torch.zeros(rows, 1, device=grad.device, dtype=grad.dtype)
            mask[lo : hi + 1] = 1.0
            cache[rows] = mask
        return grad * mask.to(grad.device, grad.dtype)

    return hook


def attach_coord_grad_mask(
    model: Any,
    coord_lo: int,
    coord_hi: int,
    *,
    vocab_size: int | None = None,
    markers: Iterable[str] = EMBEDDING_PARAM_MARKERS,
) -> list[Any]:
    """Register Eq. (3) on every trainable vocabulary-shaped parameter.

    Args:
        model: any ``torch.nn.Module``.
        coord_lo, coord_hi: inclusive bounds of the coordinate block.
        vocab_size: when given, a candidate parameter whose first axis differs
            is rejected loudly rather than masked into a no-op.
        markers: substrings that identify embedding / head parameters.

    Returns:
        The hook handles, so a caller can remove them. An empty list means
        nothing was hooked, which is always a bug -- callers should treat it as
        one.
    """
    markers = tuple(markers)
    cache: dict[int, "torch.Tensor"] = {}
    handles: list[Any] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.dim() != 2:
            continue
        if not any(marker in name for marker in markers):
            continue
        if vocab_size is not None and param.shape[0] != vocab_size:
            raise ValueError(
                f"parameter {name!r} has shape {tuple(param.shape)}; its first axis is not the "
                f"vocabulary ({vocab_size}). Eq. (3) indexes vocabulary rows, so masking this "
                "parameter would select nothing and silently discard its gradient."
            )
        if param.shape[0] <= coord_hi:
            raise ValueError(
                f"parameter {name!r} has only {param.shape[0]} rows, which cannot hold the "
                f"coordinate block [{coord_lo}, {coord_hi}]."
            )
        handles.append(param.register_hook(_make_hook(coord_lo, coord_hi, cache)))
    return handles


class CoordEmbeddingGradMaskCallback(_TrainerCallback):
    """:func:`attach_coord_grad_mask` as a ``transformers`` callback.

    Attaching at ``on_train_begin`` rather than at construction time is
    deliberate: the model the Trainer actually optimises is the one produced
    *after* LoRA wrapping and accelerator preparation, and only that one has the
    final set of trainable parameters.
    """

    def __init__(self, coord_lo: int, coord_hi: int, *, vocab_size: int | None = None) -> None:
        self.coord_lo = int(coord_lo)
        self.coord_hi = int(coord_hi)
        self.vocab_size = vocab_size
        self._handles: list[Any] = []

    def on_train_begin(self, args: Any = None, state: Any = None, control: Any = None, **kwargs: Any) -> None:
        model = kwargs.get("model")
        if model is None:
            return
        self._handles = attach_coord_grad_mask(
            model, self.coord_lo, self.coord_hi, vocab_size=self.vocab_size
        )
        if not self._handles:
            raise RuntimeError(
                "gradient mask attached to no parameter: the embedding and output head are not "
                "trainable, so Eq. (3) would be a no-op and the ~248k pre-existing rows would "
                "either be frozen entirely or (worse) trainable and unmasked."
            )

    def on_train_end(self, args: Any = None, state: Any = None, control: Any = None, **kwargs: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
