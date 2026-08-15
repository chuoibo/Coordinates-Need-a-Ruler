"""The number token loss: Eq. (4)-(6), and the properties that make it worth having."""

import math

import pytest

torch = pytest.importorskip("torch")

from cnr.ntl import IGNORE_INDEX, ntl_from_logits, ntl_loss_func, wasserstein1_to_point_mass  # noqa: E402


def make_outputs(logits):
    return {"logits": logits}


def test_zero_on_a_confident_correct_prediction():
    probs = torch.zeros(1, 5)
    probs[0, 2] = 1.0
    assert wasserstein1_to_point_mass(probs, torch.tensor([2])).item() == pytest.approx(0.0)


def test_equals_normalised_expected_absolute_error():
    # Half the mass on bin 0, half on bin 4, target 2 -> mean |k - 2| = 2,
    # normalised by (K - 1) = 4 -> 0.5.
    probs = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.5]])
    assert wasserstein1_to_point_mass(probs, torch.tensor([2])).item() == pytest.approx(0.5)


def test_bounded_in_unit_interval():
    probs = torch.zeros(1, 1001)
    probs[0, 1000] = 1.0
    assert wasserstein1_to_point_mass(probs, torch.tensor([0])).item() == pytest.approx(1.0)


def test_monotone_in_distance():
    """The property cross-entropy does not have: being nearer costs less."""
    target = torch.tensor([500])
    costs = []
    for predicted in (500, 510, 600, 900):
        probs = torch.zeros(1, 1001)
        probs[0, predicted] = 1.0
        costs.append(wasserstein1_to_point_mass(probs, target).item())
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]


def test_cross_entropy_is_blind_to_the_same_thing():
    """Contrast: CE assigns the identical cost to a near miss and a far one."""
    logits = torch.full((1, 1001), -10.0)
    near, far = logits.clone(), logits.clone()
    near[0, 501] = 10.0
    far[0, 900] = 10.0
    target = torch.tensor([500])
    ce_near = torch.nn.functional.cross_entropy(near, target)
    ce_far = torch.nn.functional.cross_entropy(far, target)
    assert ce_near.item() == pytest.approx(ce_far.item(), rel=1e-6)


def test_symmetric_mass_still_costs():
    """Mass spread evenly about the target has the right mean and a nonzero cost.

    This is what separates the Wasserstein term from a squared error on the
    decoded expectation, which would read this distribution as perfect.
    """
    probs = torch.zeros(1, 1001)
    probs[0, 400] = 0.5
    probs[0, 600] = 0.5
    value = wasserstein1_to_point_mass(probs, torch.tensor([500])).item()
    assert value == pytest.approx(100 / 1000)
    assert value > 0


def test_only_coordinate_positions_contribute():
    vocab, lo, hi = 60, 20, 40
    logits = torch.randn(6, vocab)
    labels = torch.tensor([5, 25, IGNORE_INDEX, 55, 30, 19])
    per_position, mask = ntl_from_logits(logits, labels, coord_lo=lo, coord_hi=hi)
    assert mask.tolist() == [False, True, False, False, True, False]
    assert per_position.numel() == 2


def test_loss_reduces_to_cross_entropy_without_coordinates():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 50)
    labels = torch.tensor([[1, 2, 3, IGNORE_INDEX]])
    total = ntl_loss_func(make_outputs(logits), labels, None, ntl_weight=1.0, coord_lo=45, coord_hi=49)

    shifted = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].reshape(-1)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 50).float(), shifted, ignore_index=IGNORE_INDEX)
    assert total.item() == pytest.approx(ce.item(), rel=1e-5)


def test_lambda_scales_only_the_numeric_term():
    torch.manual_seed(0)
    vocab, lo, hi = 30, 10, 20
    logits = torch.randn(1, 3, vocab)
    labels = torch.tensor([[1, 15, 16]])
    kwargs = dict(coord_lo=lo, coord_hi=hi)
    at0 = ntl_loss_func(make_outputs(logits), labels, None, ntl_weight=0.0, **kwargs).item()
    at1 = ntl_loss_func(make_outputs(logits), labels, None, ntl_weight=1.0, **kwargs).item()
    at2 = ntl_loss_func(make_outputs(logits), labels, None, ntl_weight=2.0, **kwargs).item()
    assert at2 - at1 == pytest.approx(at1 - at0, rel=1e-5)


def test_both_terms_share_the_denominator():
    """Eq. (6): if the numeric term used its own count, lambda would silently
    scale with gradient accumulation. Doubling num_items_in_batch must halve
    the whole loss, not part of it."""
    torch.manual_seed(0)
    vocab, lo, hi = 30, 10, 20
    logits = torch.randn(1, 4, vocab)
    labels = torch.tensor([[1, 15, 16, 17]])
    kwargs = dict(ntl_weight=1.0, coord_lo=lo, coord_hi=hi)
    single = ntl_loss_func(make_outputs(logits), labels, torch.tensor(4.0), **kwargs)
    doubled = ntl_loss_func(make_outputs(logits), labels, torch.tensor(8.0), **kwargs)
    assert doubled.item() == pytest.approx(single.item() / 2, rel=1e-6)


def test_numeric_term_touches_only_the_coordinate_columns():
    """The numeric term reads a softmax restricted to the block, so its gradient
    cannot reach any other column. Cross-entropy still does, of course -- that is
    the division of labour: CE decides *whether* a coordinate belongs here, the
    numeric term decides *which value*."""
    vocab, lo, hi = 30, 10, 20
    logits = torch.randn(2, vocab, requires_grad=True)
    labels = torch.tensor([IGNORE_INDEX, 15])
    per_position, _ = ntl_from_logits(logits, labels, coord_lo=lo, coord_hi=hi)
    per_position.sum().backward()

    grad = logits.grad[1]
    assert torch.count_nonzero(grad[lo : hi + 1]) > 0
    outside = torch.cat([grad[:lo], grad[hi + 1 :]])
    assert torch.count_nonzero(outside).item() == 0
    # ...and a non-coordinate position gets no numeric gradient at all.
    assert torch.count_nonzero(logits.grad[0]).item() == 0


def test_full_loss_still_trains_the_rest_of_the_vocabulary():
    vocab, lo, hi = 30, 10, 20
    logits = torch.randn(1, 2, vocab, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, 15]])
    ntl_loss_func(make_outputs(logits), labels, None, ntl_weight=1.0, coord_lo=lo, coord_hi=hi).backward()
    grad = logits.grad[0, 0]
    assert torch.count_nonzero(torch.cat([grad[:lo], grad[hi + 1 :]])).item() > 0


def test_finite_on_a_realistic_block():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 1200)
    labels = torch.randint(100, 1100, (2, 8))
    loss = ntl_loss_func(make_outputs(logits), labels, None, ntl_weight=1.0, coord_lo=100, coord_hi=1100)
    assert math.isfinite(loss.item())
