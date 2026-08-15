#!/usr/bin/env python3
"""Verify the gradient mask of Eq. (3): every vocabulary row outside the coordinate
block B is bit-identical, in both the input embedding and the output head, between
the trained checkpoint and the base model it was grafted onto.

The comparison is on raw bytes, so it is dtype-agnostic and genuinely bit-exact --
no tolerance, no dtype round-trip. Only the base model's embedding tensor is
fetched, by HTTP range request against the safetensors byte offsets, so this does
not download the whole base checkpoint.

This is the check that makes Eq. (3) a fact rather than an intention: a hook
that silently attached to nothing looks exactly like a hook that worked.

Usage:
    python scripts/verify_embedding_freeze.py \
        --merged checkpoints/sft_run_no_think_merged \
        --base Qwen/Qwen3.5-2B

Expected on the released checkpoint:

    coordinate block B = [248077, 249077], 1001 tokens, contiguous, v(i) = i - l holds
    base vocab rows 248320, trained vocab rows 249088
    rows of B already present in the base model: 243
    input embedding : rows 0..248076 bit-identical
    output head     : rows 0..248076 bit-identical
    PASS
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url

EMBED_KEYS = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
HEAD_KEYS = ("lm_head.weight",)
CHUNK_ROWS = 4096


def read_safetensors_header(fh) -> tuple[dict, int]:
    """Return (header dict, offset of the data section)."""
    (n,) = struct.unpack("<Q", fh.read(8))
    header = json.loads(fh.read(n))
    return header, 8 + n


def local_tensor(model_dir: Path, candidates: tuple[str, ...]):
    """Locate a tensor across the shards of a local checkpoint.

    Returns (path, absolute byte start, absolute byte end, shape).
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = {weight_map[k]: k for k in candidates if k in weight_map}
    else:
        shards = {"model.safetensors": None}
    for shard, key in shards.items():
        path = model_dir / shard
        with path.open("rb") as fh:
            header, data_start = read_safetensors_header(fh)
        for name in candidates:
            if name in header:
                meta = header[name]
                start, end = meta["data_offsets"]
                return path, data_start + start, data_start + end, tuple(meta["shape"])
    raise KeyError(f"none of {candidates} found in {model_dir}")


def remote_tensor(repo_id: str, candidates: tuple[str, ...]):
    """Same, for a hub repo, using range requests only."""
    try:
        index = json.loads(Path(hf_hub_download(repo_id, "model.safetensors.index.json")).read_text())
        weight_map = index["weight_map"]
        shards = [weight_map[k] for k in candidates if k in weight_map]
    except Exception:
        shards = []
    if not shards:
        from huggingface_hub import HfApi

        shards = [s.rfilename for s in HfApi().model_info(repo_id).siblings if s.rfilename.endswith(".safetensors")]
    for shard in shards:
        url = hf_hub_url(repo_id, shard)
        raw = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60)
        raw.raise_for_status()
        (n,) = struct.unpack("<Q", raw.content[:8])
        head = requests.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=120)
        head.raise_for_status()
        header = json.loads(head.content)
        data_start = 8 + n
        for name in candidates:
            if name in header:
                meta = header[name]
                start, end = meta["data_offsets"]
                return url, data_start + start, data_start + end, tuple(meta["shape"])
    raise KeyError(f"none of {candidates} found in {repo_id}")


def coord_block(model_dir: Path) -> tuple[int, int, int]:
    """Return (lo, hi, count) of the contiguous <coord_k> id block, asserting Eq. (1)."""
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    coord = {t["content"]: t["id"] for t in tok.get("added_tokens", []) if t["content"].startswith("<coord_")}
    if not coord:
        coord = {k: i for k, i in tok["model"]["vocab"].items() if k.startswith("<coord_")}
    ids = sorted(coord.values())
    assert ids == list(range(ids[0], ids[-1] + 1)), "coordinate ids are not contiguous"
    for name, i in coord.items():
        k = int(name[len("<coord_") : -1])
        assert i - ids[0] == k, f"v(i) = i - l fails at {name}: {i - ids[0]} != {k}"
    return ids[0], ids[-1], len(ids)


def stream_rows(source, start: int, end: int, row_bytes: int, n_rows: int):
    """Yield the raw bytes of rows [0, n_rows) in CHUNK_ROWS blocks."""
    if isinstance(source, Path):
        with source.open("rb") as fh:
            fh.seek(start)
            for lo in range(0, n_rows, CHUNK_ROWS):
                hi = min(lo + CHUNK_ROWS, n_rows)
                yield lo, hi, fh.read((hi - lo) * row_bytes)
        return
    stop = start + n_rows * row_bytes - 1
    assert stop < end
    resp = requests.get(source, headers={"Range": f"bytes={start}-{stop}"}, stream=True, timeout=600)
    resp.raise_for_status()
    buf = b""
    lo = 0
    for piece in resp.iter_content(chunk_size=1 << 22):
        buf += piece
        while len(buf) >= CHUNK_ROWS * row_bytes and lo < n_rows:
            hi = min(lo + CHUNK_ROWS, n_rows)
            take = (hi - lo) * row_bytes
            yield lo, hi, buf[:take]
            buf = buf[take:]
            lo = hi
    if lo < n_rows:
        yield lo, n_rows, buf


def compare(label: str, local_spec, remote_spec, n_rows: int, row_bytes: int) -> list[int]:
    lpath, lstart, lend, _ = local_spec
    rurl, rstart, rend, _ = remote_spec
    mismatches: list[int] = []
    left = stream_rows(lpath, lstart, lend, row_bytes, n_rows)
    right = stream_rows(rurl, rstart, rend, row_bytes, n_rows)
    done = 0
    for (lo, hi, a), (_, _, b) in zip(left, right):
        if a != b:
            for r in range(hi - lo):
                s = r * row_bytes
                if a[s : s + row_bytes] != b[s : s + row_bytes]:
                    mismatches.append(lo + r)
        done = hi
        if done % (CHUNK_ROWS * 16) == 0:
            print(f"    {label}: {done}/{n_rows} rows", end="\r", flush=True)
    print(f"    {label}: {done}/{n_rows} rows compared, {len(mismatches)} differ")
    return mismatches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, type=Path)
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    lo, hi, count = coord_block(args.merged)
    print(f"coordinate block B = [{lo}, {hi}], {count} tokens, contiguous, v(i) = i - l holds")

    base_cfg = json.loads(Path(hf_hub_download(args.base, "config.json")).read_text())
    base_vocab = base_cfg.get("vocab_size") or base_cfg["text_config"]["vocab_size"]
    merged_cfg = json.loads((args.merged / "config.json").read_text())
    merged_vocab = merged_cfg.get("vocab_size") or merged_cfg["text_config"]["vocab_size"]
    print(f"base vocab rows {base_vocab}, trained vocab rows {merged_vocab}")
    print(f"rows of B already present in the base model: {base_vocab - lo}")
    print(f"rows compared (outside B, present in both): 0 .. {lo - 1}")

    remote_embed = remote_tensor(args.base, EMBED_KEYS)
    row_bytes = (remote_embed[2] - remote_embed[1]) // remote_embed[3][0]
    print(f"row width {remote_embed[3][1]} values, {row_bytes} bytes\n")

    results = {}
    for label, keys in (("input embedding", EMBED_KEYS), ("output head", HEAD_KEYS)):
        local_spec = local_tensor(args.merged, keys)
        print(f"  {label}: local {local_spec[0].name} vs base (tied)" if label == "output head" else f"  {label}: local {local_spec[0].name} vs base")
        results[label] = compare(label, local_spec, remote_embed, lo, row_bytes)

    print()
    ok = all(not v for v in results.values())
    for label, bad in results.items():
        verdict = "bit-identical" if not bad else f"{len(bad)} rows differ (first: {bad[:5]})"
        print(f"{label:16s}: rows 0..{lo - 1} {verdict}")
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
