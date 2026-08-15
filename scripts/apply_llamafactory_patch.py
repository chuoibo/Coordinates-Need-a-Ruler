#!/usr/bin/env python3
"""Install our modified LlamaFactory files into a LlamaFactory checkout.

    git clone https://github.com/hiyouga/LLaMA-Factory.git   # v0.9.5.dev0
    python scripts/apply_llamafactory_patch.py --llamafactory ../LLaMA-Factory
    python scripts/apply_llamafactory_patch.py --llamafactory ../LLaMA-Factory --verify

Seven files carry the changes described in ``third_party/llamafactory/README.md``:
the number token loss, its wiring into the SFT trainer, the gradient-mask
callback, the description-based embedding initialisation, the two argument
dataclasses that expose them, and the no-``<think>`` chat template.

``--backup`` (default) writes ``<file>.orig`` beside anything it overwrites, so
the operation is reversible without a second clone. ``--verify`` only reports
which destinations already match, which is the fast way to answer "did this env
actually get patched?" after a fresh ``pip install``.

These files are derived from LlamaFactory (Apache-2.0) and keep its licence
headers; see ``third_party/llamafactory/README.md`` for attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PATCH_ROOT = HERE / "third_party" / "llamafactory"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llamafactory", type=Path, required=True, help="root of a LlamaFactory checkout")
    parser.add_argument("--verify", action="store_true", help="report only; change nothing")
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    args = parser.parse_args()

    manifest = json.loads((PATCH_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    dest_root = args.llamafactory / manifest["dest_root"]
    if not dest_root.is_dir():
        raise SystemExit(
            f"{dest_root} is not a directory. Point --llamafactory at the repository root, "
            "the folder that contains src/llamafactory."
        )

    print(f"LlamaFactory {manifest['llamafactory_version']} -> {dest_root}")
    applied = matched = missing = 0
    for entry in manifest["files"]:
        source = PATCH_ROOT / "files" / entry["path"]
        target = dest_root / entry["path"]
        if source.exists() and sha256(source) != entry["sha256"]:
            raise SystemExit(f"vendored file {source} does not match MANIFEST.json; the patch tree is corrupt")

        if not target.exists():
            print(f"  MISSING  {entry['path']}  (destination absent -- wrong version?)")
            missing += 1
            continue
        if sha256(target) == entry["sha256"]:
            print(f"  ok       {entry['path']}")
            matched += 1
            continue
        if args.verify:
            print(f"  DIFFERS  {entry['path']}")
            continue
        if args.backup:
            shutil.copy2(target, target.with_suffix(target.suffix + ".orig"))
        shutil.copy2(source, target)
        print(f"  patched  {entry['path']}" + ("  (backup: .orig)" if args.backup else ""))
        applied += 1

    if args.verify:
        print(f"\n{matched}/{len(manifest['files'])} files already match" + (f", {missing} missing" if missing else ""))
        return 0 if matched == len(manifest["files"]) else 1

    print(f"\npatched {applied}, already current {matched}" + (f", missing {missing}" if missing else ""))
    if missing:
        print(
            "Some destinations were absent. Check that the checkout is LlamaFactory "
            f"{manifest['llamafactory_version']} and re-run."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
