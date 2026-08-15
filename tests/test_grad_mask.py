"""Eq. (3): only the coordinate rows learn, and the hook refuses to no-op."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from cnr.grad_mask import CoordEmbeddingGradMaskCallback, attach_coord_grad_mask  # noqa: E402

VOCAB, DIM, LO, HI = 64, 8, 40, 50


class TiedModel(nn.Module):
    """Input embedding and output head as separate modules, as the adapter wraps them."""

    def __init__(self, vocab=VOCAB, dim=DIM):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.other = nn.Linear(dim, dim, bias=False)

    def forward(self, ids):
        return self.lm_head(self.other(self.embed_tokens(ids)))


def test_hooks_both_matrices_not_just_one():
    """The quantifier is load-bearing: the base checkpoint ties the embedding and
    the head, but the adapter wraps them separately, so one hook is not enough."""
    model = TiedModel()
    handles = attach_coord_grad_mask(model, LO, HI, vocab_size=VOCAB)
    assert len(handles) == 2


def test_gradient_is_zero_outside_the_block():
    model = TiedModel()
    attach_coord_grad_mask(model, LO, HI, vocab_size=VOCAB)
    model(torch.tensor([[1, 2, 45]])).sum().backward()

    for weight in (model.embed_tokens.weight, model.lm_head.weight):
        grad = weight.grad
        assert torch.count_nonzero(grad[:LO]).item() == 0
        assert torch.count_nonzero(grad[HI + 1 :]).item() == 0


def test_coordinate_rows_still_receive_gradient():
    model = TiedModel()
    attach_coord_grad_mask(model, LO, HI, vocab_size=VOCAB)
    model(torch.tensor([[45]])).sum().backward()
    assert torch.count_nonzero(model.lm_head.weight.grad[LO : HI + 1]).item() > 0


def test_unrelated_parameters_are_untouched():
    model = TiedModel()
    attach_coord_grad_mask(model, LO, HI, vocab_size=VOCAB)
    model(torch.tensor([[1, 45]])).sum().backward()
    assert torch.count_nonzero(model.other.weight.grad).item() > 0


def test_a_wrongly_shaped_parameter_is_rejected_loudly():
    """A low-rank factor of the head has the wrong first axis; masking it would
    select nothing and silently discard that gradient. Refuse instead."""
    model = TiedModel()
    model.lm_head_lora = nn.Parameter(torch.randn(4, DIM))
    with pytest.raises(ValueError, match="not the vocabulary"):
        attach_coord_grad_mask(model, LO, HI, vocab_size=VOCAB)


def test_frozen_embeddings_make_the_callback_fail_rather_than_pretend():
    model = TiedModel()
    for param in model.parameters():
        param.requires_grad_(False)
    callback = CoordEmbeddingGradMaskCallback(LO, HI, vocab_size=VOCAB)
    with pytest.raises(RuntimeError, match="attached to no parameter"):
        callback.on_train_begin(model=model)


def test_callback_attaches_and_detaches():
    model = TiedModel()
    callback = CoordEmbeddingGradMaskCallback(LO, HI, vocab_size=VOCAB)
    callback.on_train_begin(model=model)
    model(torch.tensor([[1, 45]])).sum().backward()
    assert torch.count_nonzero(model.embed_tokens.weight.grad[:LO]).item() == 0

    callback.on_train_end(model=model)
    model.zero_grad(set_to_none=True)
    model(torch.tensor([[1, 45]])).sum().backward()
    assert torch.count_nonzero(model.embed_tokens.weight.grad[:LO]).item() > 0
