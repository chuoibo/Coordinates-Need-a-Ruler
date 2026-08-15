"""Boxes, masks, the [0, 1000] <-> pixel maps, and the scoring rules."""

import numpy as np
import pytest
from PIL import Image

from cnr.geometry import (
    bbox_1000_to_pixels,
    bbox_iou,
    bbox_rect_mask,
    mask_area_fraction,
    mask_iou,
    mask_to_bbox,
    point_1000_to_pixel,
    polygon_to_mask,
)
from cnr.metrics import RESULTS, MaskFormatError, evaluate_mask_directories, score_bbox_predictions
from cnr.prompts import derive_prompt_targets, normalize_pixel_point


def square_mask(size=100, x0=20, y0=30, w=40, h=25):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0 : y0 + h, x0 : x0 + w] = True
    return mask


def test_polygon_rasterises_to_the_submission_format():
    pts = [{"x": 10, "y": 10}, {"x": 30, "y": 10}, {"x": 30, "y": 20}, {"x": 10, "y": 20}]
    mask = polygon_to_mask(pts, 50, 40)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert mask.sum() > 0


def test_polygon_clips_points_outside_the_frame():
    # VizWiz workers routinely draw past the edge; clipping beats crashing.
    pts = [{"x": -50, "y": -50}, {"x": 200, "y": -50}, {"x": 200, "y": 200}, {"x": -50, "y": 200}]
    assert polygon_to_mask(pts, 30, 20).mean() == 255.0


def test_degenerate_polygon_is_empty_not_an_error():
    assert polygon_to_mask([{"x": 1, "y": 1}, {"x": 2, "y": 2}], 10, 10).sum() == 0


def test_mask_to_bbox_is_exclusive_on_the_far_corner():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    assert mask_to_bbox(mask) == (3, 2, 7, 5)
    assert mask_to_bbox(np.zeros((4, 4), dtype=np.uint8)) is None


def test_bbox_iou_conventions():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert bbox_iou((0, 0, 10, 10), (0, 0, 20, 10)) == pytest.approx(0.5)
    assert bbox_iou(None, None) == 1.0       # nothing to find, nothing found
    assert bbox_iou(None, (0, 0, 5, 5)) == 0.0
    assert bbox_iou((0, 0, 5, 5), None) == 0.0


def test_bbox_1000_decode_swaps_reversed_corners():
    assert bbox_1000_to_pixels([500, 500, 100, 100], 1000, 1000) == (100, 100, 500, 500)


def test_bbox_1000_decode_rejects_collapsed_and_malformed():
    assert bbox_1000_to_pixels([500, 500, 500, 500], 1000, 1000) is None
    assert bbox_1000_to_pixels([1, 2, 3], 100, 100) is None
    assert bbox_1000_to_pixels(None, 100, 100) is None
    assert bbox_1000_to_pixels(["a", 2, 3, 4], 100, 100) is None


def test_point_round_trip_lands_within_one_pixel():
    """Quantising to 1001 bins is lossy, and near W = 1000 the +0.5 pixel-centre
    convention can push a point into the next bin. The round trip is therefore
    only guaranteed to within a pixel -- which is exactly why
    ``safe_normalized_point`` exists rather than trusting the arithmetic."""
    for width, height in [(640, 480), (1000, 1000), (37, 91), (1600, 1200)]:
        for x, y in [(0, 0), (width - 1, height - 1), (width // 3, height // 7)]:
            normalised = normalize_pixel_point(x, y, width, height)
            px, py = point_1000_to_pixel(normalised, width, height)
            assert abs(px - x) <= 1 and abs(py - y) <= 1


def test_safe_point_is_what_guarantees_the_class_survives():
    """The guarantee the pipeline actually relies on: a click keeps its class
    after the round trip, even where the arithmetic alone would not."""
    from cnr.prompts import safe_normalized_point

    mask = square_mask(size=1000, x0=400, y0=400, w=60, h=40)
    height, width = mask.shape
    for x, y in [(400, 400), (459, 439), (430, 420)]:
        point = safe_normalized_point(mask, x, y, foreground=True)
        px, py = point_1000_to_pixel(point, width, height)
        assert mask[py, px]
    for x, y in [(10, 10), (900, 900)]:
        point = safe_normalized_point(mask, x, y, foreground=False)
        px, py = point_1000_to_pixel(point, width, height)
        assert not mask[py, px]


def test_bbox_rect_mask_covers_the_box():
    mask = bbox_rect_mask([0, 0, 1000, 1000], 20, 10)
    assert mask.shape == (10, 20)
    assert mask.mean() == 255.0
    assert bbox_rect_mask(None, 20, 10).sum() == 0


def test_mask_iou_and_area_fraction():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[0:5, :] = 255
    b[0:10, :] = 255
    assert mask_iou(a, b) == pytest.approx(0.5)
    assert mask_iou(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0
    assert mask_area_fraction(a) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mask_iou(np.zeros((4, 4)), np.zeros((5, 5)))


def test_prompt_targets_land_where_they_should():
    mask = square_mask()
    bbox, positives, negatives = derive_prompt_targets(mask)
    assert len(positives) == 2
    assert len(negatives) >= 1
    height, width = mask.shape
    for point in positives:
        px, py = point_1000_to_pixel(point, width, height)
        assert mask[py, px]              # positives must decode onto foreground
    for point in negatives:
        px, py = point_1000_to_pixel(point, width, height)
        assert not mask[py, px]          # and negatives onto background
    assert bbox[2] > bbox[0] and bbox[3] > bbox[1]


def test_prompt_targets_on_an_empty_mask():
    assert derive_prompt_targets(np.zeros((10, 10), dtype=bool)) == ([0, 0, 0, 0], [], [])


def _write_mask(path, arr):
    Image.fromarray(arr.astype(np.uint8) * 255).save(path)


def test_missing_prediction_scores_zero_rather_than_vanishing(tmp_path):
    """The rule behind 75.70 rather than 75.73: a failed generation is a miss."""
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    full = np.ones((8, 8), dtype=bool)
    _write_mask(gt_dir / "a.png", full)
    _write_mask(gt_dir / "b.png", full)
    _write_mask(pred_dir / "a.png", full)      # b.png deliberately absent

    scored = evaluate_mask_directories(pred_dir, gt_dir, missing_scores_zero=True)
    assert scored.n == 2
    assert scored.mean_iou == pytest.approx(0.5)

    dropped = evaluate_mask_directories(pred_dir, gt_dir, missing_scores_zero=False)
    assert dropped.n == 1
    assert dropped.mean_iou == pytest.approx(1.0)


def test_strict_format_rejects_a_non_binary_mask(tmp_path):
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    _write_mask(gt_dir / "a.png", np.ones((4, 4), dtype=bool))
    Image.fromarray(np.full((4, 4), 128, dtype=np.uint8)).save(pred_dir / "a.png")

    with pytest.raises(MaskFormatError, match="non-binary"):
        evaluate_mask_directories(pred_dir, gt_dir, strict_format=True)
    assert evaluate_mask_directories(pred_dir, gt_dir, strict_format=False).mean_iou == pytest.approx(1.0)


def test_bbox_scoring_counts_a_geometry_failure_as_zero(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:60, 10:60] = True
    _write_mask(gt_dir / "a.png", mask)
    _write_mask(gt_dir / "b.png", mask)

    predictions = [
        {"image_id": "a.jpg", "width": 100, "height": 100, "bbox_1000": [100, 100, 600, 600], "size_class": "large"},
        {"image_id": "b.jpg", "width": 100, "height": 100, "bbox_1000": None, "parse_error": "no JSON",
         "size_class": "large"},
    ]
    report = score_bbox_predictions(predictions, gt_dir)
    assert report.n == 2
    assert report.n_no_geometry == 1
    assert report.per_sample[0]["bbox_iou"] > 0.9
    assert report.per_sample[1]["bbox_iou"] == 0.0
    assert report.by_bucket()["large"]["n"] == 2


def test_published_reference_numbers_are_self_consistent():
    # The gold-prompt reference is an upper estimate, so it must sit above the
    # system; and the box proxy runs above the mask metric.
    assert RESULTS["mask_iou_gold_prompt"] > RESULTS["mask_iou"]
    assert RESULTS["bbox_iou"] > RESULTS["mask_iou"]
    assert RESULTS["bbox_iou_small"] < RESULTS["bbox_iou_medium"] < RESULTS["bbox_iou_large"]
