"""Drive the command-line pipeline on a synthetic dataset.

The unit tests cover the arithmetic; this covers the wiring -- that the scripts
parse their arguments, find each other's outputs, and produce files with the
shape the next stage expects. It runs in a couple of seconds on eight fabricated
images and needs no GPU, no model and no real data.

The two stages that need a checkpoint (``run_inference.py``) or a segmenter
(``decode_masks.py``) are exercised only as far as their argument parsing;
everything downstream of them is fed a hand-written prediction file, which is
also how you would debug a real run without re-generating.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N_IMAGES = 8


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"{script} failed:\n{result.stdout}\n{result.stderr}")
    return result


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A miniature VizWiz: images, masks, and a grounding JSON.

    Region sizes are spread across the three buckets on purpose, so the crop
    plan has something to do in each of them.
    """
    root = tmp_path_factory.mktemp("corpus")
    (root / "data" / "train").mkdir(parents=True)
    entries = {}
    rng = np.random.default_rng(0)

    for i in range(N_IMAGES):
        width, height = 400, 300
        image_id = f"VizWiz_train_{i:08d}.jpg"
        Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(
            root / "data" / "train" / image_id
        )
        # side 20 -> ~0.3% (small), 90 -> ~6.8% (medium), 200 -> ~33% (large)
        side = (20, 90, 200)[i % 3]
        x0, y0 = 60 + (i * 7) % 100, 40 + (i * 5) % 60
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0 : y0 + side, x0 : x0 + side] = 255
        Image.fromarray(mask).save(root / "data" / "train" / image_id.replace(".jpg", ".png"))
        entries[image_id] = {
            "question": f"what is item {i}?",
            "most_common_answer": f"answer {i}",
            "width": width,
            "height": height,
        }

    grounding = root / "data" / "train_grounding.json"
    grounding.write_text(json.dumps(entries), encoding="utf-8")
    return root, grounding


def test_make_coord_tokens_emits_the_full_block(tmp_path):
    out = tmp_path / "coord_tokens_desc.yaml"
    run("make_coord_tokens.py", "--out", str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1001
    assert lines[0] == "<coord_0>: '0'"
    assert lines[-1] == "<coord_1000>: '1000'"


def test_shipped_token_config_matches_a_fresh_build(tmp_path):
    """The committed config must be exactly what the generator produces."""
    out = tmp_path / "fresh.yaml"
    run("make_coord_tokens.py", "--out", str(out))
    assert out.read_text(encoding="utf-8") == (ROOT / "configs" / "coord_tokens_desc.yaml").read_text(
        encoding="utf-8"
    )


def test_build_dataset_then_crops(corpus, tmp_path):
    root, grounding = corpus
    base = tmp_path / "base.json"

    run(
        "build_dataset.py",
        "--grounding", str(grounding),
        "--split", "train",
        "--data-root", str(root),
        "--instruction", str(ROOT / "configs" / "instruction.txt"),
        "--out", str(base),
    )
    records = json.loads(base.read_text(encoding="utf-8"))
    assert len(records) == N_IMAGES

    for record in records:
        assert set(record) == {"instruction", "input", "output", "images"}
        assert record["instruction"].startswith("<image>")
        assert "<coord_" in record["output"]
        assert "Answer region size: " in record["input"]

    # every bucket is represented, so the crop plan is genuinely exercised
    tags = {r["input"].rpartition("Answer region size: ")[2] for r in records}
    assert tags == {"small", "medium", "large"}

    out = tmp_path / "ms.json"
    result = run(
        "build_crops.py",
        "--base", str(base),
        "--data-root", str(root),
        "--image-out", str(tmp_path / "crops"),
        "--image-prefix", "crops",
        "--out", str(out),
        "--manifest", str(tmp_path / "manifest.json"),
        "--workers", "2",
    )
    assert "max round-trip drift" in result.stdout

    combined = json.loads(out.read_text(encoding="utf-8"))
    assert len(combined) >= len(records)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for meta in manifest:
        assert meta["roundtrip_drift"] <= 3           # the gate held
        assert Path(tmp_path / "crops" / meta["crop_image"]).exists()
        x0, y0, w, h = meta["crop_xywh"]
        full_w, full_h = meta["full_size"]
        assert 0 <= x0 and x0 + w <= full_w
        assert 0 <= y0 and y0 + h <= full_h

    # crops carry a coordinate target and point at the crop image, not the parent
    crops = combined[len(records) :]
    for crop in crops:
        assert crop["images"][0].startswith("crops/")
        assert "<coord_" in crop["output"]

    # large records are never cropped
    parents = {m["parent_image"] for m in manifest}
    large = {r["images"][0] for r in records if r["input"].endswith("large")}
    assert parents.isdisjoint(large)


def test_make_size_tags_from_masks(corpus, tmp_path):
    root, grounding = corpus
    out = tmp_path / "size.csv"
    run(
        "make_size_tags.py",
        "--gt-dir", str(root / "data" / "train"),
        "--annotations", str(grounding),
        "--out", str(out),
    )
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "image_id,size_class,mask_area_frac"
    assert len(lines) == N_IMAGES + 1
    assert {line.split(",")[1] for line in lines[1:]} == {"small", "medium", "large"}


def test_evaluate_scores_a_prediction_file(corpus, tmp_path):
    root, grounding = corpus
    entries = json.loads(grounding.read_text(encoding="utf-8"))

    # A perfect box for every item except one, which failed to produce geometry.
    predictions = []
    for i, (image_id, entry) in enumerate(entries.items()):
        with Image.open(root / "data" / "train" / image_id.replace(".jpg", ".png")) as im:
            mask = np.asarray(im.convert("L")) > 0
        ys, xs = np.nonzero(mask)
        w, h = entry["width"], entry["height"]
        bbox = None if i == 0 else [
            round(xs.min() / w * 1000), round(ys.min() / h * 1000),
            round((xs.max() + 1) / w * 1000), round((ys.max() + 1) / h * 1000),
        ]
        predictions.append(
            {"image_id": image_id, "width": w, "height": h, "bbox_1000": bbox,
             "parse_error": None if bbox else "no JSON object found"}
        )
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    size_csv = tmp_path / "size.csv"
    run("make_size_tags.py", "--gt-dir", str(root / "data" / "train"),
        "--annotations", str(grounding), "--out", str(size_csv))

    out_dir = tmp_path / "eval"
    result = run(
        "evaluate.py",
        "--predictions", str(pred_path),
        "--gt-dir", str(root / "data" / "train"),
        "--bucket-csv", str(size_csv),
        "--out-dir", str(out_dir),
    )
    assert "mean bounding-box IoU" in result.stdout

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["bbox"]["n"] == N_IMAGES
    assert summary["bbox"]["n_no_geometry"] == 1
    # 7 near-perfect boxes and one scored zero
    assert 0.8 < summary["bbox"]["mean_iou"] < 1.0
    assert set(summary["bbox"]["by_bucket"]) == {"small", "medium", "large"}
    assert (out_dir / "results.md").exists()


@pytest.mark.parametrize("script", ["run_inference.py", "decode_masks.py", "apply_llamafactory_patch.py"])
def test_gpu_scripts_at_least_parse_their_arguments(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_patch_manifest_matches_the_vendored_files():
    import hashlib

    manifest = json.loads((ROOT / "third_party" / "llamafactory" / "MANIFEST.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / "third_party" / "llamafactory" / "files" / entry["path"]
        assert path.exists(), entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_no_shipped_file_is_accidentally_git_ignored():
    """An unanchored ignore rule once swallowed
    ``third_party/llamafactory/files/data/template.py`` -- present on disk,
    absent from the clone, and no test noticed. Ask git directly."""
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=ROOT
    )
    if tracked.returncode != 0:  # not a git checkout (e.g. installed from a tarball)
        pytest.skip("not a git working tree")
    files = set(tracked.stdout.split())

    manifest = json.loads((ROOT / "third_party" / "llamafactory" / "MANIFEST.json").read_text())
    required = {f"third_party/llamafactory/files/{e['path']}" for e in manifest["files"]}
    required |= {
        "configs/instruction.txt",
        "configs/coord_tokens_desc.yaml",
        "configs/train_stage1.yaml",
        "configs/train_stage2.yaml",
        "configs/merge_lora.yaml",
    }
    missing = sorted(required - files)
    assert not missing, f"present on disk but not tracked by git: {missing}"
