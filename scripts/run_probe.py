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
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
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
from src.interp.probe import (  # noqa: E402
    ProbeResult,
    fit_probe,
    pool_activation,
    verdict,
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _fit_nonlinear_in_subprocess(
    out_dir: Path,
    site_names: list[str],
    acts: dict[str, list[np.ndarray]],
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    nv: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Fit XGBoost probes in a fresh process.

    On macOS/arm64, XGBoost can segfault if it receives NumPy labels after Torch/MPS has
    already been active in the process. Keeping the boosted trees in a CPU-only child
    preserves the probe definition while letting the parent own the Metal forward pass.
    """
    input_path = out_dir / "nonlinear_input.npz"
    output_path = out_dir / "nonlinear_metrics.json"
    np.savez_compressed(
        input_path,
        y=y,
        tr=tr.astype(int),
        te=te.astype(int),
        nv=nv.astype(int),
        site_names=np.asarray(site_names),
        **{name: np.stack(acts[name]) for name in site_names},
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.fit_probe_nonlinear",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--seed",
            str(seed),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


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
    ap.add_argument(
        "--nonlinear",
        action="store_true",
        help=(
            "also fit a gradient-boosted-tree probe at every site, on the same split. "
            "Distinguishes 'the destination is not encoded here' from 'it is encoded but "
            "not linearly readable' -- the two readings the linear null cannot separate."
        ),
    )
    ap.add_argument(
        "--cache-activations",
        action="store_true",
        help="write pooled activations to activations.npz so later probes need no forward passes",
    )
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
                "xgboost_version": _package_version("xgboost"),
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
    state_of: list[int] = []   # which state each example came from, for the group split

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
                state_of.append(i)
        if (i + 1) % 5 == 0:
            per = (time.time() - t0) / (i + 1)
            print(f"    {i + 1}/{len(states)} states  ({per:.1f}s/state, "
                  f"eta {per * (len(states) - i - 1) / 60:.1f} min)", flush=True)

    y = np.array(labels)
    novel_mask = np.array(is_novel)
    print(f"\nexamples: {len(y)} total, {(~novel_mask).sum()} trained, {novel_mask.sum()} novel")

    # Train on TRAINED pairings only, test on trained-heldout AND novel.
    #
    # SPLIT BY STATE, not by example. Each state contributes 8 examples (2 objects x 4
    # destinations) that share an observation and a noise draw, so an example-level split
    # puts near-identical activations in both train and test and lets the probe memorise
    # state-specific patterns. That leak showed up unmistakably: acc_trained came out at
    # 1.000 at EVERY site, including expert layers where the instruction signal is ~3e-2
    # and one where the activation is byte-identical across instructions. Grouping by
    # state removes it.
    groups = np.asarray(state_of, dtype=int)
    trained_states = np.unique(groups[~novel_mask])
    rng.shuffle(trained_states)
    cut = int(0.7 * len(trained_states))
    train_states, test_states = set(trained_states[:cut]), set(trained_states[cut:])

    tr = np.array([i for i in np.flatnonzero(~novel_mask) if groups[i] in train_states])
    te = np.array([i for i in np.flatnonzero(~novel_mask) if groups[i] in test_states])
    # Novel examples are held out by construction; also drop any whose state was trained on,
    # so no probe ever sees a test state.
    nv = np.array([i for i in np.flatnonzero(novel_mask) if groups[i] in test_states])
    if nv.size < 10:  # too few novel test examples once states are held out
        nv = np.flatnonzero(novel_mask)
        print("  note: too few novel examples in held-out states; using all novel examples")
    print(f"  split by state: {len(train_states)} train / {len(test_states)} test states")

    # Activations are the expensive part and are identical for every probe family, so
    # cache them once. A later probe variant then costs seconds instead of a full sweep of
    # forward passes -- which is what made the non-linear probe below cheap to add.
    if args.cache_activations:
        np.savez_compressed(
            out_dir / "activations.npz",
            y=y, groups=np.asarray(groups), novel_mask=novel_mask,
            **{s.name: np.stack(acts[s.name]) for s in sites},
        )
        print(f"  cached activations -> {out_dir / 'activations.npz'}")

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

    nonlinear = {}
    if args.nonlinear:
        # Same split, same standardisation, same readout -- only the hypothesis class
        # differs, so any gap over the linear probe is attributable to linearity. The
        # boosted fit runs in a fresh CPU-only process because XGBoost can segfault after
        # Torch/MPS has been active on macOS/arm64.
        nonlinear = _fit_nonlinear_in_subprocess(
            out_dir, site_names, acts, y, tr, te, nv, args.seed
        )

    v = verdict(results)
    payload: dict = {"verdict": v, "sites": [r.as_dict() for r in results]}
    if nonlinear:
        payload["nonlinear"] = nonlinear
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"DESTINATION PROBE  ->  {v['verdict']}")
    print("=" * 74)
    print(f"chance = {chance:.3f}\n")
    print(f"{'site':<26}{'trained':>9}{'novel':>9}{'shuffled':>10}")
    for r in sorted(results, key=lambda x: -x.acc_novel)[:15]:
        print(f"{r.site:<26}{r.acc_trained:>9.3f}{r.acc_novel:>9.3f}{r.acc_shuffled:>10.3f}")
    print(f"\n{v['reason']}")

    if nonlinear:
        # The comparison that matters is linear vs boosted at the SAME site on the SAME
        # split. A boosted probe that gains only where its own shuffled control also rises
        # is fitting noise rather than finding structure, so all three are printed.
        print("\n" + "=" * 74)
        print("NON-LINEAR PROBE (gradient-boosted trees) vs LINEAR, novel pairings")
        print("=" * 74)
        print(f"{'site':<26}{'linear':>9}{'boosted':>9}{'delta':>8}{'bst-shuf':>10}")
        for r in sorted(results, key=lambda x: -nonlinear[x.site]["acc_novel"])[:15]:
            nl = nonlinear[r.site]
            print(f"{r.site:<26}{r.acc_novel:>9.3f}{nl['acc_novel']:>9.3f}"
                  f"{nl['acc_novel'] - r.acc_novel:>+8.3f}{nl['acc_shuffled']:>10.3f}")
        deltas = [nonlinear[r.site]["acc_novel"] - r.acc_novel for r in results]
        exp = [d for r, d in zip(results, deltas, strict=True) if r.tower == "expert"]
        print(f"\nmean(boosted - linear): all sites {np.mean(deltas):+.3f}, "
              f"expert sites {np.mean(exp):+.3f}")
        print(f"expert sites where boosted beats linear by >0.10: "
              f"{sum(1 for d in exp if d > 0.10)}/{len(exp)}")
        print("If that count is near zero, the linear null inside the expert is a genuine")
        print("absence rather than a limit of the probe's hypothesis class.")

    print(f"\nwritten -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
