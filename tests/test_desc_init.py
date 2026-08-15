"""Eq. (2): seed each coordinate row from the digits that spell it."""

import pytest

torch = pytest.importorskip("torch")

from cnr.desc_init import description_init, gaussian_init, mean_of_description  # noqa: E402

VOCAB, DIM = 40, 6
NEW = {"<coord_0>": "0", "<coord_1>": "1", "<coord_12>": "12"}


class FakeTokenizer:
    """Digits 0-9 are ids 0-9; the new tokens sit at 30, 31, 32."""

    def __init__(self):
        self.ids = {"<coord_0>": 30, "<coord_1>": 31, "<coord_12>": 32}

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([[int(ch) for ch in text]])}


def test_row_is_the_mean_of_its_digit_rows():
    weight = torch.arange(VOCAB * DIM, dtype=torch.float32).reshape(VOCAB, DIM)
    expected = weight[torch.tensor([1, 2])].mean(dim=0).clone()
    description_init(weight, NEW, FakeTokenizer())
    torch.testing.assert_close(weight[32], expected)   # "12" -> mean of rows 1 and 2


def test_single_digit_row_is_copied():
    weight = torch.randn(VOCAB, DIM)
    expected = weight[1].clone()
    description_init(weight, NEW, FakeTokenizer())
    torch.testing.assert_close(weight[31], expected)


def test_rows_are_addressed_by_id_never_by_position():
    """The wraparound bug: position-based indexing overwrites real low-id rows
    when the vocabulary was pre-padded. Everything outside the new ids must be
    untouched."""
    weight = torch.randn(VOCAB, DIM)
    before = weight.clone()
    description_init(weight, NEW, FakeTokenizer())
    for row in range(VOCAB):
        if row in (30, 31, 32):
            continue
        torch.testing.assert_close(weight[row], before[row])


def test_snapshot_prevents_cascading_reads():
    """Row 0 is written first; row 12's mean must still use the ORIGINAL row 1,
    not a value this same loop has already replaced."""
    weight = torch.arange(VOCAB * DIM, dtype=torch.float32).reshape(VOCAB, DIM)
    original = weight.clone()
    ordered = {"<coord_1>": "1", "<coord_12>": "12"}

    class Tok(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.ids = {"<coord_1>": 1 + 30, "<coord_12>": 32}

    description_init(weight, ordered, Tok())
    torch.testing.assert_close(weight[32], original[torch.tensor([1, 2])].mean(dim=0))


def test_description_tokens_that_are_new_are_excluded():
    weight = torch.randn(VOCAB, DIM)
    snapshot = weight.clone()
    # ids 30-32 are the new rows; excluding them leaves only 1 and 2.
    value = mean_of_description("12", FakeTokenizer(), snapshot, exclude_ids={30, 31, 32})
    torch.testing.assert_close(value, snapshot[torch.tensor([1, 2])].mean(dim=0))


def test_out_of_range_token_is_an_error_not_a_wraparound():
    weight = torch.randn(VOCAB, DIM)

    class Bad(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.ids = {"<coord_0>": VOCAB + 5}

    with pytest.raises(ValueError, match="outside the embedding matrix"):
        description_init(weight, {"<coord_0>": "0"}, Bad())


def effective_rank(block: "torch.Tensor", rtol: float = 1e-6) -> int:
    """Singular values above ``rtol`` times the largest.

    An explicit relative tolerance rather than ``matrix_rank``'s default: the
    tail here is float rounding noise around 1e-7 of the leading value, which
    the default treats as signal.
    """
    sv = torch.linalg.svdvals(block.double())
    return int((sv > sv[0] * rtol).sum())


def test_description_init_collapses_the_block_onto_a_low_rank_subspace():
    """The measured null result rests on this contrast: every coordinate row is
    a mean of the ten digit rows, so the whole block lives in a ten-dimensional
    subspace. The noise arm is full rank -- and the two score within a point of
    each other, which is what bounds the effect of the starting point."""
    dim = 16
    desc_weight = torch.randn(1200, dim, dtype=torch.float64)
    tokens = {f"<coord_{k}>": str(k) for k in range(0, 100)}

    class Tok:
        def convert_tokens_to_ids(self, token):
            return 1000 + int(token[len("<coord_") : -1])

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.tensor([[int(ch) for ch in text]])}

    description_init(desc_weight, tokens, Tok())
    desc_rank = effective_rank(desc_weight[1000:1100])

    noise_weight = torch.zeros(1200, dim, dtype=torch.float64)
    gaussian_init(noise_weight, list(range(1000, 1100)), sigma=0.02)
    noise_rank = effective_rank(noise_weight[1000:1100])

    assert desc_rank <= 10          # spanned by the ten digit embeddings
    assert noise_rank == dim        # full rank
    assert desc_rank < noise_rank


def test_description_init_produces_far_fewer_distinct_rows_than_tokens():
    """1001 tokens do not give 1001 distinct starting points: rows sharing a
    digit multiset share an embedding, because the mean is order-blind."""
    weight = torch.randn(1200, 8, dtype=torch.float64)
    tokens = {f"<coord_{k}>": str(k) for k in range(0, 100)}

    class Tok:
        def convert_tokens_to_ids(self, token):
            return 1000 + int(token[len("<coord_") : -1])

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.tensor([[int(ch) for ch in text]])}

    description_init(weight, tokens, Tok())
    distinct = {tuple(row.tolist()) for row in weight[1000:1100]}
    assert len(distinct) < 100
    # "12" and "21" are the same mean, so they collide.
    torch.testing.assert_close(weight[1000 + 12], weight[1000 + 21])


def test_gaussian_arm_writes_the_named_rows_only():
    weight = torch.zeros(VOCAB, DIM)
    gaussian_init(weight, [30, 31], sigma=0.02)
    assert torch.count_nonzero(weight[30]) > 0
    assert torch.count_nonzero(weight[31]) > 0
    assert torch.count_nonzero(weight[32]).item() == 0
