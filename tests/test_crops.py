"""Multi-scale crop augmentation: the constraints, and what they reject."""

import pytest

from cnr.crops import (
    CONTAIN_PAD,
    CROP_PLANS,
    MAX_AREA_FRAC,
    MIN_SIDE_PX,
    classify_size,
    crop_window,
    to_crop_1000,
    to_full_1000,
)


def test_plan_matches_the_paper():
    assert CROP_PLANS["small"] == [2.0, 3.5]
    assert CROP_PLANS["medium"] == [2.5]
    assert CROP_PLANS["large"] == []


@pytest.mark.parametrize(
    "fraction,expected",
    [(0.0, "small"), (0.049, "small"), (0.05, "medium"), (0.19, "medium"), (0.20, "large"), (0.9, "large")],
)
def test_size_thresholds(fraction, expected):
    assert classify_size(fraction) == expected


def test_crop_keeps_the_image_aspect_ratio():
    window = crop_window(2000, 1000, 900, 450, 1000, 550, 3.0)
    assert window is not None
    assert window.w / window.h == pytest.approx(2000 / 1000, rel=0.02)


def test_crop_contains_the_region_with_margin():
    W, H = 1600, 1200
    box = (700, 500, 800, 600)
    window = crop_window(W, H, *box, 3.0)
    assert window is not None
    x0, y0, x1, y1 = window.box
    assert x0 <= box[0] and y0 <= box[1] and x1 >= box[2] and y1 >= box[3]
    assert window.w >= CONTAIN_PAD * (box[2] - box[0]) * 0.99


def test_crop_respects_the_resolution_floor():
    # A tiny region at high zoom would produce a crop far below 320 px; the
    # floor takes over rather than manufacturing blur.
    window = crop_window(4000, 3000, 1990, 1490, 2010, 1510, 3.5)
    assert window is not None
    assert min(window.w, window.h) >= MIN_SIDE_PX - 1


def test_crop_rejected_when_it_would_cover_the_whole_image():
    # A region already filling much of the frame has nothing to zoom into.
    assert crop_window(1000, 1000, 50, 50, 950, 950, 2.0) is None


def test_crop_area_stays_under_the_cap():
    window = crop_window(1200, 900, 500, 400, 700, 600, 2.0)
    if window is not None:
        assert (window.w * window.h) / (1200 * 900) <= MAX_AREA_FRAC + 1e-6


def test_crop_stays_inside_the_frame():
    for box in [(0, 0, 60, 60), (1540, 1140, 1600, 1200), (0, 570, 60, 630)]:
        window = crop_window(1600, 1200, *box, 3.0)
        assert window is not None
        x0, y0, x1, y1 = window.box
        assert 0 <= x0 < x1 <= 1600
        assert 0 <= y0 < y1 <= 1200


def test_relative_position_is_preserved_for_an_interior_region():
    """A region in the right half must stay in the right half of its crop.

    This is what keeps directional language in the supervised text true after
    the crop, which matters because that text is copied across unchanged.
    """
    W, H = 2000, 1000
    box = (1500, 300, 1600, 400)
    window = crop_window(W, H, *box, 3.0)
    assert window is not None
    centre_before = (box[0] + box[2]) / 2 / W
    centre_after = ((box[0] + box[2]) / 2 - window.x0) / window.w
    assert centre_after > 0.5
    assert centre_after == pytest.approx(centre_before, abs=0.25)


def test_coordinate_round_trip_is_within_the_gate():
    W, w_c, x0 = 1600, 400, 300
    for k in (0, 250, 500, 750, 1000):
        back = to_full_1000(to_crop_1000(k, W, x0, w_c), W, x0, w_c)
        # k must first be inside the crop for the round trip to be meaningful
        if 0 <= to_crop_1000(k, W, x0, w_c) <= 1000:
            assert abs(back - k) <= 3


def test_crop_frame_maps_the_window_edges_to_the_full_range():
    W, w_c, x0 = 1600, 400, 300
    left = to_crop_1000(int(round(x0 / W * 1000)), W, x0, w_c)
    right = to_crop_1000(int(round((x0 + w_c) / W * 1000)), W, x0, w_c)
    assert left == pytest.approx(0, abs=3)
    assert right == pytest.approx(1000, abs=3)


def test_degenerate_region_is_rejected():
    assert crop_window(1000, 1000, 500, 500, 500, 500, 2.0) is None
