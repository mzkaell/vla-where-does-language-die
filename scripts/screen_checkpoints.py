#!/usr/bin/env python
"""Screen public checkpoints for competence before investing in any experiment.

    python scripts/screen_checkpoints.py --n 50

Why this runs first
-------------------
A grounding score is only interpretable if the policy can do the task. M0 spent hours
producing clean, significant numbers from two checkpoints, one of which turned out to be
worse than predicting a random trajectory -- and the other only became competent after an
input-orientation bug was fixed. Screening is cheap (~5 min per checkpoint at n=50) and
tells you which checkpoints are worth an experiment at all.

It also answers a prior question: the project's standing rule requires two independent
things to agree, so **at least two competent checkpoints must exist** or no claim can ever
ship regardless of how good the experiment is.

Metric
------
Each checkpoint predicts an action chunk under the CORRECT (demonstrated) instruction. We
report

    ratio = median ||prediction - demonstration|| / median ||other demo - demonstration||

where the denominator is a data-only baseline: the distance between two *unrelated*
demonstrated chunks from the same stimulus states. That is what "a plausible trajectory
for this scene, but not this one" scores.

    < 0.70   competent -- tracks the specific demonstration
    < 1.05   no better than an unrelated trajectory
    >= 1.05  worse than baseline; misaligned preprocessing or a broken checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_ifr import build_images, build_state  # noqa: E402
from src.data.build_pairs import load_observation, load_pairs  # noqa: E402

# Compatible schemas only: 2 cameras, 8-dim state, 7-dim action. Checkpoints declaring
# 32-dim padded features or malformed shapes are excluded by --checkpoints defaults.
DEFAULT_CHECKPOINTS = [
    "k1000dai/smolvla_libero_finetune",
    "msv6/smolvla_meta_libero",
    "bicmol/smolvla-libero",
    "jadechoghari/smolvla-libero-ckpts",
    "k1000dai/smolvla_libero_scratch_80k",
    "xainyuxxx/my-smolvla-libero",
    "AustineJohnBreaker/smolvla_stratch_libero_spatial",
    "k1000dai/smolvla-libero-pick-up-the-black-bowl-20k",
]

COMPETENT, MARGINAL = 0.70, 1.05


def verdict_for(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "ERROR"
    if ratio < COMPETENT:
        return "COMPETENT"
    if ratio < MARGINAL:
        return "NOT COMPETENT"
    return "WORSE THAN BASELINE"


def baseline_distance(pairs, data_root: Path, chunk_size: int, action_dim: int, seed: int) -> float:
    """Median distance between two unrelated demonstrated chunks. Model-independent."""
    rng = np.random.default_rng(seed)
    chunks = [
        np.asarray(
            load_observation(p, data_root, chunk_size=chunk_size)["action_chunk"][:, :action_dim],
            dtype=np.float64,
        )
        for p in pairs
    ]
    stack = np.stack(chunks)
    i = rng.integers(0, len(stack), 4000)
    j = rng.integers(0, len(stack), 4000)
    keep = i != j
    d = np.linalg.norm((stack[i[keep]] - stack[j[keep]]).reshape(keep.sum(), -1), axis=1)
    return float(np.median(d))


def screen_one(checkpoint: str, pairs, data_root: Path, device: str) -> dict:
    from src.models.smolvla import SmolVLA, make_batch

    t0 = time.time()
    model = SmolVLA.load(checkpoint, device=device)
    cfg = model.config
    keys = list(cfg.image_features)
    size = cfg.image_features[keys[0]].shape[-1]
    adim = cfg.action_feature.shape[0]
    sdim = cfg.robot_state_feature.shape[0]

    dists = []
    for p in pairs:
        obs = load_observation(p, data_root, chunk_size=cfg.chunk_size)
        demo = torch.from_numpy(
            np.asarray(obs["action_chunk"][:, :adim], dtype=np.float32)
        ).unsqueeze(0)
        state = model.normalize_state(build_state(obs, sdim))
        # The instruction the demonstration was actually following.
        src_is_a = p["source_task"] in p["instruction_a"].replace(" ", "_")
        instruction = p["instruction_a"] if src_is_a else p["instruction_b"]
        batch = make_batch(build_images(obs, keys, size), state, instruction, model.policy, device)
        pred = model.unnormalize_action(model.predict_action(batch, noise=model.make_noise(1)))
        dists.append(float(torch.linalg.vector_norm((pred - demo).reshape(-1))))

    return {
        "checkpoint": checkpoint,
        "n": len(dists),
        "median_pred_to_demo": float(np.median(dists)),
        "has_norm_stats": model.has_norm_stats,
        "action_dim": int(adim),
        "state_dim": int(sdim),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50, help="pairs per checkpoint")
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "checkpoint_screen.json")
    args = ap.parse_args()

    pairs_file = args.pairs or REPO_ROOT / "stimuli" / "libero_goal_pairs_v1.jsonl"
    pairs = load_pairs(pairs_file)[: args.n]
    if not pairs:
        print(f"no pairs in {pairs_file}", file=sys.stderr)
        return 1

    baseline = baseline_distance(pairs, args.data_root, 50, 7, args.seed)
    print(f"pairs            : {len(pairs)} from {pairs_file.name}")
    print(f"data baseline    : {baseline:.4f}  (median distance between unrelated demos)")
    print(f"thresholds       : <{COMPETENT} competent, <{MARGINAL} not competent, else worse\n")

    rows = []
    for i, ckpt in enumerate(args.checkpoints, 1):
        print(f"[{i}/{len(args.checkpoints)}] {ckpt} ...", flush=True)
        try:
            row = screen_one(ckpt, pairs, args.data_root, args.device)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            rows.append({"checkpoint": ckpt, "error": f"{type(exc).__name__}: {exc}"})
            continue
        row["baseline"] = baseline
        row["ratio"] = row["median_pred_to_demo"] / baseline
        row["verdict"] = verdict_for(row["ratio"])
        rows.append(row)
        print(f"    ratio {row['ratio']:.3f}  -> {row['verdict']}  ({row['seconds']}s)")

    print("\n" + "=" * 78)
    print(f"{'checkpoint':<52}{'ratio':>8}  verdict")
    print("=" * 78)
    for r in sorted(rows, key=lambda x: x.get("ratio", float("inf"))):
        if "error" in r:
            print(f"{r['checkpoint']:<52}{'--':>8}  ERROR")
        else:
            print(f"{r['checkpoint']:<52}{r['ratio']:>8.3f}  {r['verdict']}")

    competent = [r for r in rows if r.get("verdict") == "COMPETENT"]
    print("=" * 78)
    print(f"\ncompetent checkpoints: {len(competent)}")
    for r in competent:
        print(f"  {r['checkpoint']}  (ratio {r['ratio']:.3f})")

    if len(competent) < 2:
        print(
            "\nFEWER THAN TWO COMPETENT CHECKPOINTS.\n"
            "The standing rule requires two independent sources agreeing, so no headline\n"
            "claim can ship on SmolVLA regardless of experiment quality. Consider OpenVLA,\n"
            "which has official LIBERO checkpoints.",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"baseline": baseline, "n_pairs": len(pairs), "results": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
