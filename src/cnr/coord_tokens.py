# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Atomic coordinate tokens: the vocabulary block of Section 3.3.

A coordinate in ``[0, 1000]`` is written as a *single* token ``<coord_k>`` rather
than as a string of digits. The 1001 tokens occupy a contiguous block of
vocabulary indices ``B = [lo, hi]``, so a token's value is affine in its index::

    v(i) = i - lo,      i in B                                        (Eq. 1)

That affine map is the whole point: it is what lets a loss read the *numeric*
distance between two tokens straight off their indices, which
:mod:`cnr.ntl` then does.

Two things are built here:

* :func:`coord_token` / :func:`parse_coord_token` -- the string form.
* :func:`build_descriptions` -- the ``token -> description`` table consumed by
  the numeric-description initialisation of :mod:`cnr.desc_init`. Each
  ``<coord_k>`` is described by the plain decimal string of ``k``, so its
  starting embedding is the mean of the digit embeddings that spell it.

:func:`resolve_block` recovers ``(lo, hi)`` from a tokenizer and *asserts* the
block really is contiguous and affine. Every entry point that consumes the block
calls it; a silently non-contiguous block would make the numeric term optimise
the wrong distance, which is the kind of bug that shows up only as a slightly
worse number.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "COORD_MAX",
    "COORD_MIN",
    "COORD_TOKEN_RE",
    "NUM_COORD_TOKENS",
    "CoordBlock",
    "build_descriptions",
    "coord_token",
    "decode_coord_tokens",
    "parse_coord_token",
    "quantize",
    "resolve_block",
]

COORD_MIN = 0
COORD_MAX = 1000
NUM_COORD_TOKENS = COORD_MAX - COORD_MIN + 1  # K = 1001

COORD_TOKEN_PREFIX = "<coord_"
COORD_TOKEN_RE = re.compile(r"<coord_(\d+)>")


class CoordBlock:
    """The resolved vocabulary block ``[lo, hi]`` of coordinate tokens."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: int, hi: int) -> None:
        if hi - lo != COORD_MAX - COORD_MIN:
            raise ValueError(f"coordinate block must span {NUM_COORD_TOKENS} ids; got [{lo}, {hi}]")
        self.lo = int(lo)
        self.hi = int(hi)

    def __len__(self) -> int:
        return NUM_COORD_TOKENS

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CoordBlock(lo={self.lo}, hi={self.hi})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CoordBlock) and (self.lo, self.hi) == (other.lo, other.hi)

    def value(self, token_id: int) -> int:
        """Eq. (1): the coordinate value carried by a vocabulary index."""
        if not self.lo <= token_id <= self.hi:
            raise ValueError(f"token id {token_id} is outside the coordinate block [{self.lo}, {self.hi}]")
        return int(token_id) - self.lo

    def token_id(self, value: int) -> int:
        """Inverse of :meth:`value`."""
        if not COORD_MIN <= value <= COORD_MAX:
            raise ValueError(f"coordinate value {value} is outside [{COORD_MIN}, {COORD_MAX}]")
        return self.lo + int(value)


def quantize(value: float) -> int:
    """Round a coordinate to the nearest bin and clamp it into ``[0, 1000]``.

    Worker-drawn polygons can fall slightly outside the frame, so clamping here
    rather than asserting is deliberate.
    """
    k = int(round(float(value)))
    return max(COORD_MIN, min(COORD_MAX, k))


def coord_token(value: float) -> str:
    """``437 -> '<coord_437>'`` (quantising and clamping first)."""
    return f"{COORD_TOKEN_PREFIX}{quantize(value)}>"


def parse_coord_token(token: str) -> int:
    """``'<coord_437>' -> 437``. Raises :class:`ValueError` on anything else."""
    m = COORD_TOKEN_RE.fullmatch(token)
    if m is None:
        raise ValueError(f"not a coordinate token: {token!r}")
    k = int(m.group(1))
    if not COORD_MIN <= k <= COORD_MAX:
        raise ValueError(f"coordinate token out of range: {token!r}")
    return k


def decode_coord_tokens(text: str | None) -> str | None:
    """Rewrite every ``<coord_k>`` back to the bare integer ``k``.

    The supervised target is deliberately *not* valid JSON -- ``<coord_412>`` is
    unquoted -- so both the training-data builder and the evaluator round-trip
    through this function. A no-op on text that holds no coordinate tokens, so
    it is safe to call on the output of a digit-based checkpoint too.
    """
    if not text:
        return text
    return COORD_TOKEN_RE.sub(lambda m: m.group(1), text)


def build_descriptions() -> dict[str, str]:
    """The ``<coord_k> -> "k"`` table used by numeric-description initialisation.

    Describing ``<coord_437>`` as ``"437"`` means :mod:`cnr.desc_init` seeds its
    row with the mean of the embeddings of the tokens spelling 437 -- Eq. (2).
    """
    return {f"{COORD_TOKEN_PREFIX}{k}>": str(k) for k in range(COORD_MIN, COORD_MAX + 1)}


def resolve_block(tokenizer: "PreTrainedTokenizerBase") -> CoordBlock:
    """Resolve and validate the coordinate block on a tokenizer.

    Raises if the block is missing, non-contiguous, or not affine. Checking the
    midpoint as well as the endpoints catches a block that is the right *size*
    but has been permuted -- endpoints alone would not.
    """
    unk = getattr(tokenizer, "unk_token_id", None)

    def _id(value: int) -> int:
        tid = tokenizer.convert_tokens_to_ids(f"{COORD_TOKEN_PREFIX}{value}>")
        if tid is None or (unk is not None and tid == unk):
            raise ValueError(
                f"coordinate token <coord_{value}> is not in the tokenizer. "
                "Add the block first (see scripts/make_coord_tokens.py)."
            )
        return int(tid)

    lo, hi = _id(COORD_MIN), _id(COORD_MAX)
    if hi - lo != COORD_MAX - COORD_MIN:
        raise ValueError(f"coordinate tokens are not contiguous: <coord_0>={lo}, <coord_1000>={hi}")
    for probe in (1, 500, 999):
        if _id(probe) != lo + probe:
            raise ValueError(
                f"coordinate tokens are not ordered: <coord_{probe}>={_id(probe)}, expected {lo + probe}. "
                "The numeric term would optimise the wrong distance."
            )
    return CoordBlock(lo, hi)
