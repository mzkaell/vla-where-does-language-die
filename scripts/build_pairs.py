#!/usr/bin/env python
"""Build the LIBERO-Goal contrastive stimulus set (M1).

    python scripts/build_pairs.py --suite libero_goal --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_pairs import (  # noqa: E402
    SCHEMA_VERSION,
    build_pairs,
    discover_tasks,
    write_pairs,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-per-demo", type=int, default=2)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-hash",
        action="store_true",
        help="skip source file hashing (faster; not for a release build)",
    )
    args = ap.parse_args()

    suite_dir = args.data_root / args.suite
    out_path = args.out or REPO_ROOT / "stimuli" / f"{args.suite}_pairs_v1.jsonl"

    print(f"suite dir : {suite_dir}")
    tasks = discover_tasks(suite_dir, compute_hash=not args.no_hash)
    print(f"tasks     : {len(tasks)}")
    for t in tasks:
        print(f"  - {t.instruction!r}  ({t.num_demos} demos)")

    pairs, report = build_pairs(
        tasks,
        n=args.n,
        max_per_demo=args.max_per_demo,
        stride=args.stride,
        seed=args.seed,
    )

    print(f"\nminimal instruction pairings: {len(report['instruction_pairings'])}")
    for p in report["instruction_pairings"]:
        print(f"  [{p['family']:17s}] {p['swap']}")
    if report["rejected_pairings"]:
        print(f"rejected pairings: {len(report['rejected_pairings'])} (not single-referent swaps)")

    if not pairs:
        print("\nFAIL: produced 0 pairs", file=sys.stderr)
        return 1

    content_hash = write_pairs(pairs, out_path, report)

    print(f"\nproduced  : {len(pairs)} pairs (requested {args.n})")
    print(f"by family : {json.dumps(report['counts_by_family'])}")
    print(f"schema    : {SCHEMA_VERSION}")
    print(f"written   : {out_path}")
    print(f"sha256    : {content_hash}")

    if len(pairs) < args.n:
        print(
            f"\nWARNING: produced {len(pairs)} < requested {args.n}. "
            "Increase --max-per-demo or --stride coverage, or add task files.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
