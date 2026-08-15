# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Numeric-description initialisation of the coordinate rows (Section 3.3, Eq. 2).

Each new row ``<coord_k>`` is seeded with the mean of the *pre-resize*
embeddings of the tokens that spell ``k``::

    E[lo + k] <- mean_{s in tok("k")} E~[s]                            (Eq. 2)

which is Fast Vocabulary Transfer's mean-of-subword rule applied to a numeric
description. ``E~`` is a snapshot taken before any row is written, so the loop
never reads a row it has already overwritten.

**This is a measured null result.** Against 1001 rows drawn from
``N(0, 0.02^2)``, Eq. (2) scores 80.90 vs 80.57 bounding-box IoU -- a paired
difference of -0.33 with interval [-0.91, +0.27], one run per arm at one seed.
It is kept because it is the arm the reported checkpoint was trained with, not
because it helps. :func:`gaussian_init` is the comparison arm, shipped so the
A/B can be re-run.

Two failure modes this module exists to prevent:

**Row wraparound.** Never index new rows as ``E[-num_new + i]``. Qwen ships a
pre-padded vocabulary (embedding rows > ``len(tokenizer)``), so a resize can
create *fewer* new rows than there are descriptions. Position-based indexing
then lands descriptions on the wrong tokens and wraps into positive indices,
overwriting the embeddings of real low-id tokens. We resolve every row through
``convert_tokens_to_ids``.

**Untied export.** If the base checkpoint ties the input embedding and the
output head, both must be written -- and the export must not silently re-tie
them. ``scripts/verify_embedding_freeze.py`` checks both matrices.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedTokenizerBase

__all__ = ["description_init", "gaussian_init", "mean_of_description"]


def mean_of_description(
    description: str,
    tokenizer: "PreTrainedTokenizerBase",
    snapshot: "torch.Tensor",
    *,
    exclude_ids: frozenset[int] | set[int] = frozenset(),
) -> "torch.Tensor":
    """Mean of the snapshot embeddings of the tokens spelling ``description``.

    Tokens in ``exclude_ids`` -- the new rows themselves -- are dropped, since
    they carry no meaning yet. If nothing survives, falls back to the mean of
    the whole snapshot rather than leaving the row at whatever resize produced.
    """
    ids = tokenizer(description, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    keep = [int(i) for i in ids.tolist() if int(i) not in exclude_ids]
    if not keep:
        return snapshot.mean(dim=0)
    index = torch.tensor(keep, device=snapshot.device, dtype=torch.long)
    return snapshot[index].mean(dim=0)


@torch.no_grad()
def description_init(
    embed_weight: "torch.Tensor",
    descriptions: Mapping[str, str],
    tokenizer: "PreTrainedTokenizerBase",
    *,
    add_noise: bool = False,
) -> int:
    """Write Eq. (2) into ``embed_weight`` in place. Returns the rows written.

    Args:
        embed_weight: ``[vocab, dim]`` matrix to initialise.
        descriptions: ``token string -> description``; see
            :func:`cnr.coord_tokens.build_descriptions`.
        tokenizer: already extended with the new tokens.
        add_noise: add ``N(0, 1/dim)`` on top of the mean (the
            ``desc_init_w_noise`` arm).
    """
    dim = embed_weight.size(1)
    snapshot = embed_weight.detach().clone()
    new_ids = {
        tid
        for tid in (tokenizer.convert_tokens_to_ids(tok) for tok in descriptions)
        if tid is not None
    }

    written = 0
    for token, description in descriptions.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id >= embed_weight.size(0):
            raise ValueError(
                f"token {token!r} resolves to id {token_id}, outside the embedding matrix "
                f"({embed_weight.size(0)} rows). Resize the vocabulary before initialising."
            )
        row = mean_of_description(description, tokenizer, snapshot, exclude_ids=new_ids)
        if add_noise:
            row = row + torch.randn_like(row) * (1.0 / math.sqrt(dim))
        embed_weight[token_id] = row
        written += 1
    return written


@torch.no_grad()
def gaussian_init(
    embed_weight: "torch.Tensor",
    token_ids: "list[int]",
    *,
    sigma: float = 0.02,
    generator: "torch.Generator | None" = None,
) -> int:
    """The comparison arm: draw each named row from ``N(0, sigma^2)``.

    Rows are addressed by id, never by position, for the wraparound reason in
    the module docstring.
    """
    for token_id in token_ids:
        if token_id < 0 or token_id >= embed_weight.size(0):
            raise ValueError(f"token id {token_id} is outside the embedding matrix")
        noise = torch.randn(
            embed_weight.size(1), generator=generator, device=embed_weight.device, dtype=embed_weight.dtype
        )
        embed_weight[token_id] = noise * sigma
    return len(token_ids)
