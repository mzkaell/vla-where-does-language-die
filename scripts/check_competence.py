#!/usr/bin/env python
"""Is the policy actually competent at these states? (diagnostic)

    python scripts/check_competence.py --run m0_v2_msv6

A directional IFR at chance has two very different explanations: the policy does not
ground language, or the policy cannot do the task at all and its predictions are noise.
Those demand opposite responses -- the first is a finding, the second means the checkpoint
is unusable -- so they must be separated before any grounding claim is made.

The test needs no forward passes. `per_pair.jsonl` already stores the distance from each
predicted chunk to the demonstrated chunk. We compare that against a **data-only
baseline**: the distance between two *unrelated* demonstrated chunks drawn from the same
stimulus states. That baseline is what "a plausible trajectory for this scene, but not this
one" scores.

    ratio = median ||prediction - demo|| / median ||other demo - demo||

    ratio << 1   prediction tracks the specific demonstration -> competent
    ratio ~ 1    no better than an unrelated trajectory       -> not competent here
    ratio >> 1   worse than an unrelated trajectory           -> badly broken or misaligned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_pairs import load_observation, load_pairs  # noqa: E402
from src.eval.stats import bootstrap_mean  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run id under results/")
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--n-baseline", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "results" / args.run
    per_pair_path = run_dir / "per_pair.jsonl"
    if not per_pair_path.exists():
        print(f"no per_pair.jsonl in {run_dir}", file=sys.stderr)
        return 1

    outcomes = [json.loads(line) for line in per_pair_path.read_text().splitlines() if line.strip()]
    pairs = load_pairs(args.pairs or REPO_ROOT / "stimuli" / "libero_goal_pairs_v1.jsonl")
    by_id = {p["pair_id"]: p for p in pairs}

    # Distance from the prediction under the CORRECT (demonstrated) instruction.
    correct = np.array(
        [o["dist_a_to_demo"] if o["source_is_a"] else o["dist_b_to_demo"] for o in outcomes],
        dtype=np.float64,
    )
    correct = correct[np.isfinite(correct)]
    if not correct.size:
        print("no finite distances; the run had no usable reference", file=sys.stderr)
        return 1

    # Data-only baseline: unrelated demonstrated chunks from the same stimulus states.
    rng = np.random.default_rng(args.seed)
    ids = [o["pair_id"] for o in outcomes if o["pair_id"] in by_id]
    sample = rng.choice(len(ids), size=min(200, len(ids)), replace=False)
    chunks = []
    for i in sample:
        obs = load_observation(by_id[ids[int(i)]], args.data_root, chunk_size=args.chunk_size)
        chunks.append(np.asarray(obs["action_chunk"][:, : args.action_dim], dtype=np.float64))
    stack = np.stack(chunks)

    a = rng.integers(0, len(stack), args.n_baseline)
    b = rng.integers(0, len(stack), args.n_baseline)
    keep = a != b
    baseline = np.linalg.norm(
        (stack[a[keep]] - stack[b[keep]]).reshape(keep.sum(), -1), axis=1
    )

    med_c, med_b = float(np.median(correct)), float(np.median(baseline))
    ratio = med_c / med_b if med_b else float("nan")
    est = bootstrap_mean(correct, seed=args.seed)

    print(f"run                : {args.run}")
    print(f"n predictions      : {correct.size}")
    print(f"median ||pred-demo||           : {med_c:.4f}")
    print(f"median ||other demo - demo||   : {med_b:.4f}   (data-only baseline)")
    print(f"ratio                          : {ratio:.3f}")
    print(f"mean ||pred-demo||             : {est}")

    if ratio < 0.7:
        verdict = "COMPETENT -- predictions track the specific demonstration"
    elif ratio < 1.05:
        verdict = "NOT COMPETENT here -- no better than an unrelated trajectory"
    else:
        verdict = "WORSE THAN BASELINE -- likely misaligned inputs or a broken checkpoint"
    print(f"\nverdict: {verdict}")

    (run_dir / "competence.json").write_text(
        json.dumps(
            {
                "median_pred_to_demo": med_c,
                "median_baseline": med_b,
                "ratio": ratio,
                "n": int(correct.size),
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
