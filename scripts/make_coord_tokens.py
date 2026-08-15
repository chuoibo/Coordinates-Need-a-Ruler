#!/usr/bin/env python3
"""Write the ``<coord_k> -> "k"`` description table used to seed the new rows.

    python scripts/make_coord_tokens.py --out configs/coord_tokens_desc.yaml

The file is a flat YAML mapping consumed by the training config's
``new_special_tokens_config``. It is emitted rather than committed by hand
because 1001 entries typed once is 1001 chances to typo one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnr.coord_tokens import NUM_COORD_TOKENS, build_descriptions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("configs/coord_tokens_desc.yaml"))
    args = parser.parse_args()

    descriptions = build_descriptions()
    assert len(descriptions) == NUM_COORD_TOKENS, len(descriptions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Quote the value so YAML reads "0" as a string, not the integer 0.
    args.out.write_text(
        "".join(f"{token}: '{desc}'\n" for token, desc in descriptions.items()), encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(descriptions)} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
