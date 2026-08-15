"""The vocabulary block: Eq. (1), and the validation that protects it."""

import pytest

from cnr.coord_tokens import (
    COORD_MAX,
    NUM_COORD_TOKENS,
    CoordBlock,
    build_descriptions,
    coord_token,
    decode_coord_tokens,
    parse_coord_token,
    quantize,
    resolve_block,
)


class FakeTokenizer:
    """A tokenizer whose coordinate block we control, to test the guards."""

    unk_token_id = 0

    def __init__(self, lo=1000, *, permute=False, missing=False, gap=False):
        self.lo = lo
        self.permute = permute
        self.missing = missing
        self.gap = gap

    def convert_tokens_to_ids(self, token):
        k = int(token[len("<coord_") : -1])
        if self.missing:
            return self.unk_token_id
        if self.gap and k == COORD_MAX:
            return self.lo + k + 5
        if self.permute and k == 500:
            return self.lo + 501
        return self.lo + k


def test_block_has_1001_tokens():
    assert NUM_COORD_TOKENS == 1001
    assert len(build_descriptions()) == 1001


def test_description_is_the_decimal_string():
    # Eq. (2) seeds <coord_437> from the tokens spelling "437", so the
    # description must be exactly that and nothing more.
    descriptions = build_descriptions()
    assert descriptions["<coord_437>"] == "437"
    assert descriptions["<coord_0>"] == "0"
    assert descriptions["<coord_1000>"] == "1000"


def test_quantize_rounds_and_clamps():
    assert quantize(436.6) == 437
    assert quantize(-12) == 0        # worker polygons do fall outside the frame
    assert quantize(1200) == 1000


def test_token_round_trip():
    for value in (0, 1, 437, 999, 1000):
        assert parse_coord_token(coord_token(value)) == value


@pytest.mark.parametrize("bad", ["<coord_>", "coord_437", "<coord_1001>", "437", ""])
def test_parse_rejects_non_tokens(bad):
    with pytest.raises(ValueError):
        parse_coord_token(bad)


def test_decode_is_a_no_op_without_coordinate_tokens():
    text = '{"answer": "79 calories", "bbox_1000": [1, 2, 3, 4]}'
    assert decode_coord_tokens(text) == text


def test_decode_leaves_digits_in_the_answer_alone():
    # The answer field routinely contains numbers ("106.8"); a bare-integer
    # rewrite would corrupt them, which is why the token form is used.
    raw = '{"answer": "106.8", "bbox_1000": [<coord_12>, <coord_0>, <coord_999>, <coord_1000>]}'
    assert decode_coord_tokens(raw) == '{"answer": "106.8", "bbox_1000": [12, 0, 999, 1000]}'


def test_affine_value_map():
    block = CoordBlock(248077, 249077)
    assert block.value(248077) == 0
    assert block.value(248077 + 437) == 437
    assert block.token_id(437) == 248077 + 437
    with pytest.raises(ValueError):
        block.value(248076)


def test_block_must_span_1001_ids():
    with pytest.raises(ValueError):
        CoordBlock(100, 200)


def test_resolve_block_accepts_a_contiguous_block():
    assert resolve_block(FakeTokenizer(lo=248077)) == CoordBlock(248077, 249077)


def test_resolve_block_rejects_a_missing_block():
    with pytest.raises(ValueError, match="not in the tokenizer"):
        resolve_block(FakeTokenizer(missing=True))


def test_resolve_block_rejects_a_gap():
    with pytest.raises(ValueError, match="not contiguous"):
        resolve_block(FakeTokenizer(gap=True))


def test_resolve_block_rejects_a_permutation_the_endpoints_would_hide():
    # Endpoints alone cannot catch this: only the interior probe does, and a
    # permuted block would make the numeric term optimise the wrong distance.
    with pytest.raises(ValueError, match="not ordered"):
        resolve_block(FakeTokenizer(permute=True))
