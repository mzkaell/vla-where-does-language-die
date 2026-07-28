#!/usr/bin/env python
"""Fetch the LIBERO demonstration files the stimulus set references.

    python scripts/download_data.py                 # the 6 tasks used by v1 (3.3 GB)
    python scripts/download_data.py --all           # all 10 LIBERO-Goal tasks (5.9 GB)
    python scripts/download_data.py --check         # verify what is present, download nothing

These are not in git (far too large). Everything else in the repo is, including the
stimulus set itself, which stores references into these files plus their SHA-256 so a
teammate can prove they have the same bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPO_ID = "yifengzhu-hf/LIBERO-datasets"

# The six tasks that yield the v1 minimal pairings. The other four LIBERO-Goal tasks
# ("open the middle drawer...", "turn on the stove", "push the plate...", "open the top
# drawer and put the bowl inside") share no single-referent swap with these, so they add
# download size without adding pairs.
DEFAULT_TASKS = [
    "put_the_bowl_on_the_plate_demo.hdf5",
    "put_the_bowl_on_the_stove_demo.hdf5",
    "put_the_bowl_on_top_of_the_cabinet_demo.hdf5",
    "put_the_cream_cheese_in_the_bowl_demo.hdf5",
    "put_the_wine_bottle_on_the_rack_demo.hdf5",
    "put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5",
]

EXTRA_TASKS = [
    "open_the_middle_drawer_of_the_cabinet_demo.hdf5",
    "open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5",
    "push_the_plate_to_the_front_of_the_stove_demo.hdf5",
    "turn_on_the_stove_demo.hdf5",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--all", action="store_true", help="all 10 tasks instead of the 6 used")
    ap.add_argument("--check", action="store_true", help="report presence only")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--verify", action="store_true", help="re-hash local files (slow)")
    args = ap.parse_args()

    tasks = DEFAULT_TASKS + (EXTRA_TASKS if args.all else [])
    dest = args.out / args.suite

    if args.check or args.verify:
        missing = [t for t in tasks if not (dest / t).exists()]
        present = [t for t in tasks if (dest / t).exists()]
        total = sum((dest / t).stat().st_size for t in present)
        print(f"present : {len(present)}/{len(tasks)} files ({total / 1e9:.2f} GB) in {dest}")
        for t in missing:
            print(f"  MISSING {t}")
        if args.verify and present:
            from src.data.build_pairs import sha256_file

            print("\nhashing (this reads every byte) ...")
            for t in present:
                print(f"  {sha256_file(dest / t)}  {t}")
        return 1 if missing else 0

    from huggingface_hub import hf_hub_download

    print(f"repo   : {REPO_ID}")
    print(f"suite  : {args.suite}")
    print(f"files  : {len(tasks)}")
    print(f"dest   : {dest}\n")

    for i, task in enumerate(tasks, 1):
        target = dest / task
        if target.exists():
            print(f"[{i}/{len(tasks)}] have {task}")
            continue
        print(f"[{i}/{len(tasks)}] downloading {task} ...", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"{args.suite}/{task}",
            local_dir=args.out,
        )

    total = sum((dest / t).stat().st_size for t in tasks if (dest / t).exists())
    print(f"\ndone. {total / 1e9:.2f} GB in {dest}")
    print("\nnext:  python scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
