#!/usr/bin/env python
"""Layer-wise probe: is the named destination encoded at all?

    python scripts/run_probe.py --checkpoint k1000dai/smolvla_libero_finetune --device cuda

Answers the RQ2 question the transplant failed to: **readout failure** (the destination is
in the representation but the action ignores it) or **encoding failure** (it never gets
bound in). Method and controls: `src/interp/probe.py`.

Cheap by construction. One forward pass per (state, instruction), activations cached for
every site at once, then probes fit on CPU. No per-site forward passes, so this costs a
small fraction of the M2 sweep.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_ifr import STATE_KEYS, build_images, build_state  # noqa: E402
from scripts.run_composition import collect_states  # noqa: E402
from src.eval.composition import (  # noqa: E402
    DESTINATIONS,
    OBJECT_SOURCE_TASKS,
    TRAINED,
    instruction_for,
)
from src.interp.probe import ProbeResult, fit_probe, pool_activation, verdict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--n-states", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sites-limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    suite_dir = args.data_root / args.suite
    run_id = args.run_id or f"probe_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPO_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from src.models.smolvla import SmolVLA, make_batch

    print(f"run_id     : {run_id}\ncheckpoint : {args.checkpoint}\ndevice     : {args.device}")
    model = SmolVLA.load(args.checkpoint, device=args.device)
    cfg = model.config
    keys = list(cfg.image_features)
    size = cfg.image_features[keys[0]].shape[-1]
    sdim = cfg.robot_state_feature.shape[0]

    sites = model.sites()
    if args.sites_limit:
        sites = sites[: args.sites_limit]
    site_names = [s.name for s in sites]
    dests = list(DESTINATIONS)
    chance = 1.0 / len(dests)
    print(f"sites      : {len(sites)}   destinations: {len(dests)}   chance: {chance:.3f}")

    (out_dir / "config.yaml").write_text(
        "\n".join(
            f"{k}: {json.dumps(v)}"
            for k, v in {
                "run_id": run_id, "experiment": "layerwise_destination_probe",
                "checkpoint": args.checkpoint, "n_states": args.n_states,
                "destinations": dests, "chance": chance, "seed": args.seed,
                "device": args.device, "image_orientation": "rot180",
                "state_composition": list(STATE_KEYS), "platform": platform.platform(),
            }.items()
        ),
        encoding="utf-8",
    )

    pooled = [t for ts in OBJECT_SOURCE_TASKS.values() for t in ts]
    states = collect_states(suite_dir, pooled, args.n_states, args.seed)[: args.n_states]
    print(f"states     : {len(states)}")

    # activations[site] -> list of vectors; parallel label/condition lists
    acts: dict[str, list[np.ndarray]] = {n: [] for n in site_names}
    labels: list[int] = []
    is_novel: list[bool] = []

    t0 = time.time()
    for i, st in enumerate(states):
        obs = {k: st[k] for k in ("agentview_rgb", "eye_in_hand_rgb")}
        obs |= {k: st[k] for k in STATE_KEYS}
        images = build_images(obs, keys, size)
        state_t = model.normalize_state(build_state(obs, sdim))
        noise = model.make_noise(1)

        for obj in OBJECT_SOURCE_TASKS:
            for di, dest in enumerate(dests):
                batch = make_batch(
                    images, state_t, instruction_for(obj, dest), model.policy, args.device
                )
                run = model.forward_with_cache(batch, sites=site_names, noise=noise)
                for n in site_names:
                    # First occurrence: the prefix pass, where the instruction is encoded.
                    acts[n].append(pool_activation(run.occurrences(n)[0][0].float().cpu().numpy()))
                labels.append(di)
                is_novel.append(dest not in TRAINED[obj])
        if (i + 1) % 5 == 0:
            per = (time.time() - t0) / (i + 1)
            print(f"    {i + 1}/{len(states)} states  ({per:.1f}s/state, "
                  f"eta {per * (len(states) - i - 1) / 60:.1f} min)", flush=True)

    y = np.array(labels)
    novel_mask = np.array(is_novel)
    print(f"\nexamples: {len(y)} total, {(~novel_mask).sum()} trained, {novel_mask.sum()} novel")

    # Train on TRAINED pairings only (held-out split), test on trained-heldout AND novel.
    trained_idx = np.flatnonzero(~novel_mask)
    rng.shuffle(trained_idx)
    cut = int(0.7 * len(trained_idx))
    tr, te = trained_idx[:cut], trained_idx[cut:]
    nv = np.flatnonzero(novel_mask)

    results = []
    for s in sites:
        X = np.stack(acts[s.name])
        acc_tr = fit_probe(X[tr], y[tr], X[te], y[te], seed=args.seed)
        acc_nv = fit_probe(X[tr], y[tr], X[nv], y[nv], seed=args.seed)
        y_shuf = y[tr].copy()
        rng.shuffle(y_shuf)
        acc_sh = fit_probe(X[tr], y_shuf, X[te], y[te], seed=args.seed)
        results.append(
            ProbeResult(
                site=s.name, layer=s.layer, tower=s.tower,
                n_train=len(tr), n_test_trained=len(te), n_test_novel=len(nv),
                acc_trained=acc_tr, acc_novel=acc_nv, acc_shuffled=acc_sh, chance=chance,
            )
        )

    v = verdict(results)
    (out_dir / "metrics.json").write_text(
        json.dumps({"verdict": v, "sites": [r.as_dict() for r in results]}, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 74)
    print(f"DESTINATION PROBE  ->  {v['verdict']}")
    print("=" * 74)
    print(f"chance = {chance:.3f}\n")
    print(f"{'site':<26}{'trained':>9}{'novel':>9}{'shuffled':>10}")
    for r in sorted(results, key=lambda x: -x.acc_novel)[:15]:
        print(f"{r.site:<26}{r.acc_trained:>9.3f}{r.acc_novel:>9.3f}{r.acc_shuffled:>10.3f}")
    print(f"\n{v['reason']}")
    print(f"\nwritten -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
