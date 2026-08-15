"""The supervised record: the target format, and the one canonical prompt."""

from pathlib import Path

import pytest

from cnr.dataset import (
    INSTRUCTION_LENGTH,
    INSTRUCTION_SHA256,
    build_input,
    build_record,
    emit_target,
    load_instruction,
    load_unique_instruction,
    parse_target,
    retag_size,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_shipped_instruction_matches_the_released_checkpoint():
    instruction = load_instruction(CONFIGS / "instruction.txt")
    assert len(instruction) == INSTRUCTION_LENGTH
    assert instruction.startswith("<image>")


def test_edited_instruction_is_rejected(tmp_path):
    path = tmp_path / "instruction.txt"
    path.write_text("<image>\nsomething else")
    with pytest.raises(ValueError, match=INSTRUCTION_SHA256[:12]):
        load_instruction(path)
    # ...but opting out is allowed, loudly.
    assert load_instruction(path, verify=False).startswith("<image>")


def test_instruction_must_carry_the_image_placeholder(tmp_path):
    path = tmp_path / "instruction.txt"
    path.write_text("no placeholder here")
    with pytest.raises(ValueError, match="<image>"):
        load_instruction(path, verify=False)


def test_target_is_not_valid_json_until_decoded():
    import json

    target = emit_target("fruit punch", [412, 233, 508, 366], [[455, 291]], [[120, 700]])
    assert "<coord_412>" in target
    with pytest.raises(json.JSONDecodeError):
        json.loads(target)          # unquoted atomic tokens, by design
    assert parse_target(target)["bbox_1000"] == [412, 233, 508, 366]


def test_target_round_trip():
    obj = parse_target(emit_target("yes", [1, 2, 3, 4], [[5, 6], [7, 8]], [[9, 10], [11, 12]]))
    assert obj == {
        "answer": "yes",
        "bbox_1000": [1, 2, 3, 4],
        "positive_points_1000": [[5, 6], [7, 8]],
        "negative_points_1000": [[9, 10], [11, 12]],
    }


def test_field_order_puts_the_box_before_the_points():
    target = emit_target("x", [1, 2, 3, 4], [[5, 6]], [[7, 8]])
    assert target.index("bbox_1000") < target.index("positive_points_1000") < target.index("negative_points_1000")


def test_answer_digits_survive_the_round_trip():
    obj = parse_target(emit_target("106.8", [1, 2, 3, 4], [[5, 6]], []))
    assert obj["answer"] == "106.8"


def test_coordinates_are_clamped_not_dropped():
    obj = parse_target(emit_target("x", [-5, 0, 1200, 1000], [[1500, -2]], []))
    assert obj["bbox_1000"] == [0, 0, 1000, 1000]
    assert obj["positive_points_1000"] == [[1000, 0]]


def test_empty_negatives_render_and_parse():
    assert parse_target(emit_target("x", [1, 2, 3, 4], [[5, 6]], []))["negative_points_1000"] == []


@pytest.mark.parametrize(
    "wrapped",
    [
        '```json\n{"answer": "a", "bbox_1000": [1, 2, 3, 4]}\n```',
        'Here you go: {"answer": "a", "bbox_1000": [1, 2, 3, 4]}',
        '﻿{"answer": "a", "bbox_1000": [1, 2, 3, 4]}',
    ],
)
def test_parse_tolerates_what_models_actually_emit(wrapped):
    assert parse_target(wrapped)["bbox_1000"] == [1, 2, 3, 4]


def test_parse_rejects_text_with_no_object():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_target("I could not find the region.")


def test_input_matches_the_trained_form():
    assert build_input("What time is it?") == "Question: What time is it?"
    assert build_input("What time is it?", "small") == "Question: What time is it?\nAnswer region size: small"


def test_retag_size_rewrites_only_the_tag():
    text = "Question: is it 5 small ones?\nAnswer region size: small"
    assert retag_size(text, "large") == "Question: is it 5 small ones?\nAnswer region size: large"


def test_build_record_shape():
    record = build_record(
        "<image>\nprompt", "q?", "data/train/x.jpg", "yes", [1, 2, 3, 4], [[5, 6]], [[7, 8]], "medium"
    )
    assert set(record) == {"instruction", "input", "output", "images"}
    assert record["images"] == ["data/train/x.jpg"]


def test_unique_instruction_guard(tmp_path):
    import json

    path = tmp_path / "mixed.json"
    path.write_text(json.dumps([{"instruction": "<image>\na"}, {"instruction": "<image>\nb"}]))
    with pytest.raises(ValueError, match="exactly one unique instruction"):
        load_unique_instruction(path)

    path.write_text(json.dumps([{"instruction": "<image>\na"}, {"instruction": "<image>\na"}]))
    assert load_unique_instruction(path) == "<image>\na"
