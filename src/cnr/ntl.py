# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""The number token loss over coordinate bins (Section 3.4).

Cross-entropy is a classification loss over an unordered label set: where the
target is bin 437, putting mass on 436 costs exactly what putting it on 900
costs. IoU is a continuous function of coordinate error, so what is optimised
and what is measured come apart. This module closes that gap with one extra
term, computed from the model's **own output softmax** -- no auxiliary head, no
decoder, no reward.

Restricting the softmax to the coordinate columns gives a distribution over the
ordered bins and nothing else::

    p_t(k) = softmax(z_t[lo : hi + 1])_k,    k = 0 .. K-1              (Eq. 4)

With the normalised ground metric ``d(k, k') = |k - k'| / (K - 1)`` and a target
that is a point mass, the Wasserstein-1 distance is closed form -- it is just
the expected absolute bin error, rescaled to ``[0, 1]``::

    NTL_t = W1(p_t, delta_c) = E_{k ~ p_t}[|k - c_t|] / (K - 1)        (Eq. 5)

and the objective is::

    L = (1/N) sum_t CE_t  +  lambda * (1/N) sum_{t in C} NTL_t         (Eq. 6)

**Both terms share the denominator N** -- the number of supervised tokens in the
*global* batch, which HF Trainer passes as ``num_items_in_batch``. This is not a
cosmetic choice: normalising only the numeric term per micro-batch would
multiply its effective weight by roughly ``gradient_accumulation_steps`` (32 in
our recipe), so ``lambda = 1`` would silently mean ``lambda = 32``.

Division of labour between the two terms: cross-entropy over the *full*
vocabulary decides **whether** a coordinate belongs at this position; the
numeric term decides **which value**, and because it sees only the coordinate
columns it is not diluted by the other ~248k.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = ["IGNORE_INDEX", "ntl_from_logits", "ntl_loss_func", "wasserstein1_to_point_mass"]

IGNORE_INDEX = -100


def wasserstein1_to_point_mass(probs: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    """Eq. (5) for a batch of restricted distributions.

    Args:
        probs: ``[M, K]`` rows summing to 1 over the ordered bins.
        targets: ``[M]`` integer target bins in ``[0, K - 1]``.

    Returns:
        ``[M]`` normalised expected absolute bin error, each in ``[0, 1]``.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D [M, K]; got shape {tuple(probs.shape)}")
    n_bins = probs.size(-1)
    if n_bins < 2:
        raise ValueError("need at least two bins for a distance term")
    bins = torch.arange(n_bins, device=probs.device, dtype=probs.dtype)
    dist = (bins.unsqueeze(0) - targets.to(probs.dtype).unsqueeze(1)).abs() / (n_bins - 1)
    return (probs * dist).sum(dim=-1)


def ntl_from_logits(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    *,
    coord_lo: int,
    coord_hi: int,
    ignore_index: int = IGNORE_INDEX,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Per-position numeric error on the coordinate positions of a flat batch.

    Args:
        logits: ``[M_all, V]`` already-shifted, already-flattened logits.
        labels: ``[M_all]`` already-shifted, already-flattened target ids.

    Returns:
        ``(per_position, mask)`` where ``per_position`` holds Eq. (5) for the
        coordinate positions only and ``mask`` selects them out of ``labels``.
    """
    mask = (labels != ignore_index) & (labels >= coord_lo) & (labels <= coord_hi)
    if not bool(mask.any()):
        empty = logits.new_zeros(0)
        return empty, mask
    coord_logits = logits[mask][:, coord_lo : coord_hi + 1]
    probs = torch.softmax(coord_logits, dim=-1)
    targets = labels[mask] - coord_lo
    return wasserstein1_to_point_mass(probs, targets), mask


def ntl_loss_func(
    outputs: Any,
    labels: "torch.Tensor",
    num_items_in_batch: "torch.Tensor | int | None" = None,
    *,
    ntl_weight: float = 1.0,
    coord_lo: int = 0,
    coord_hi: int = 0,
    ignore_index: int = IGNORE_INDEX,
) -> "torch.Tensor":
    """Eq. (6): ``CE + lambda * NTL``, as a HF Trainer ``compute_loss_func``.

    The signature is fixed by ``transformers`` -- bind the keyword-only
    arguments with :func:`functools.partial` before handing it over. See
    :mod:`cnr.trainer_hooks`.

    ``labels`` arrive unshifted; the shift performed here is the same one
    ``transformers`` performs internally, so the two losses see identical
    positions.
    """
    logits = outputs.get("logits") if hasattr(outputs, "get") else getattr(outputs, "logits", None)
    if logits is None:
        fallback = outputs.get("loss") if hasattr(outputs, "get") else None
        return fallback if fallback is not None else torch.tensor(0.0)

    logits = logits.float()
    vocab_size = logits.size(-1)
    labels = F.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous().to(logits.device).view(-1)
    flat_logits = logits.view(-1, vocab_size)

    valid = shift_labels != ignore_index
    ce_sum = F.cross_entropy(flat_logits, shift_labels, ignore_index=ignore_index, reduction="sum")
    if num_items_in_batch is not None:
        denom = (
            num_items_in_batch.to(flat_logits.device)
            if torch.is_tensor(num_items_in_batch)
            else num_items_in_batch
        )
    else:
        denom = valid.sum().clamp(min=1)
    ce = ce_sum / denom

    per_position, _ = ntl_from_logits(
        flat_logits, shift_labels, coord_lo=coord_lo, coord_hi=coord_hi, ignore_index=ignore_index
    )
    # Same denominator as CE. See the module docstring: normalising per
    # micro-batch here would scale lambda by gradient_accumulation_steps.
    ntl = per_position.sum() / denom if per_position.numel() else ce.new_zeros(())

    return ce + ntl_weight * ntl
