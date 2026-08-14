#!/usr/bin/env python
"""M3 — binding transplant. Encoding failure or readout failure?

    python scripts/run_transplant.py --checkpoint k1000dai/smolvla_libero_finetune \
        --device cuda --run-id transplant_finetune

**Auto-targets** the site M2 found strongest, read from `results/loc_*/metrics.json`, so it
can run unattended straight after the sweep. Override with `--site`.

Method and the controls that make a positive result meaningful: see
`src/interp/transplant.py`. Verdict threshold is CLAUDE.md §8: readout if recovery ≥ 0.50.
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
from scripts.run_composition import collect_states, task_endpoints  # noqa: E402
from scripts.run_localization import CONTRASTS  # noqa: E402
from src.eval.composition import OBJECT_SOURCE_TASKS, build_anchors, instruction_for  # noqa: E402
from src.eval.stats import bootstrap_mean  # noqa: E402
from src.interp.localization import direction_cosine, net_translation  # noqa: E402
from src.interp.transplant import (  # noqa: E402
    TransplantPoint,
    binding_direction,
    inject,
    random_direction_like,
    verdict,
)


def pick_target_site(explicit: str | None, prefer: str = "expert") -> tuple[str, str]:
    """Use M2's strongest site unless told otherwise. Returns (site, provenance)."""
    if explicit:
        return explicit, "specified on the command line"

    best, best_run, best_val = None, None, -np.inf
    for m in sorted((REPO_ROOT / "results").glob("loc_*/metrics.json")):
        data = json.loads(m.read_text())
        for s in data.get("sites", []):
            r = s.get("recovery", {}).get("value")
            if r is None or not np.isfinite(r) or s.get("degenerate"):
                continue
            # The hypothesis is about the VLM->expert interface, so prefer expert-side
            # sites when they are competitive; fall back to the global best otherwise.
            score = r + (0.05 if s.get("tower") == prefer else 0.0)
            if score > best_val:
                best, best_run, best_val = s["site"], m.parent.name, r
    if best is None:
        raise SystemExit(
            "No M2 results to target. Run scripts/run_localization.py first, or pass --site."
        )
    return best, f"top site from {best_run} (recovery {best_val:+.3f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--site", default=None, help="override the auto-selected M2 target")
    ap.add_argument("--n-trials", type=int, default=40, help="states per contrast")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.25, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    suite_dir = args.data_root / args.suite
    run_id = args.run_id or f"transplant_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPO_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    site, provenance = pick_target_site(args.site)
    anchors = build_anchors(task_endpoints(suite_dir))

    from src.models.smolvla import SmolVLA, make_batch

    print(f"run_id     : {run_id}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"target site: {site}   ({provenance})")
    model = SmolVLA.load(args.checkpoint, device=args.device)
    cfg = model.config
    keys = list(cfg.image_features)
    size = cfg.image_features[keys[0]].shape[-1]
    sdim = cfg.robot_state_feature.shape[0]

    (out_dir / "config.yaml").write_text(
        "\n".join(
            f"{k}: {json.dumps(v)}"
            for k, v in {
                "run_id": run_id,
                "experiment": "M3_binding_transplant",
                "checkpoint": args.checkpoint,
                "target_site": site,
                "target_provenance": provenance,
                "alphas": args.alphas,
                "n_trials_per_contrast": args.n_trials,
                "seed": args.seed,
                "device": args.device,
                "image_orientation": "rot180",
                "state_composition": list(STATE_KEYS),
                "platform": platform.platform(),
            }.items()
        ),
        encoding="utf-8",
    )

    pooled = [t for ts in OBJECT_SOURCE_TASKS.values() for t in ts]

    # ---- pass 1: collect activations and baselines -------------------------------
    est_w, est_f, trials = [], [], []
    for contrast in CONTRASTS:
        dest = contrast["destination"]
        anchor = anchors[dest]
        for st in collect_states(suite_dir, pooled, args.n_trials, args.seed)[: args.n_trials]:
            obs = {k: st[k] for k in ("agentview_rgb", "eye_in_hand_rgb")}
            obs |= {k: st[k] for k in STATE_KEYS}
            images = build_images(obs, keys, size)
            state_t = model.normalize_state(build_state(obs, sdim))
            ee = np.asarray(st["ee_pos"], dtype=np.float64)
            noise = model.make_noise(1)

            bw = make_batch(images, state_t, instruction_for(contrast["working_object"], dest),
                            model.policy, args.device)
            bf = make_batch(images, state_t, instruction_for(contrast["failing_object"], dest),
                            model.policy, args.device)

            rw = model.forward_with_cache(bw, sites=[site], noise=noise)
            rf = model.forward_with_cache(bf, sites=[site], noise=noise)
            cw = direction_cosine(net_translation(model.unnormalize_action(rw.action)), ee, anchor)
            cf = direction_cosine(net_translation(model.unnormalize_action(rf.action)), ee, anchor)
            if not (np.isfinite(cw) and np.isfinite(cf)) or (cw - cf) <= 0.05:
                continue
            trials.append(
                {"batch_f": bf, "noise": noise, "ee": ee, "anchor": anchor,
                 "cos_w": cw, "cos_f": cf, "act_f": rf.occurrences(site)}
            )
            est_w.append(rw.occurrences(site)[0])
            est_f.append(rf.occurrences(site)[0])

    if len(trials) < 10:
        print(f"only {len(trials)} usable trials; need >=10", file=sys.stderr)
        return 1

    # Estimate the direction on the FIRST half, evaluate on the SECOND, so recovery cannot
    # be memorisation of the trials that defined the direction.
    split = len(trials) // 2
    d = binding_direction(est_w[:split], est_f[:split])
    d_ctrl = random_direction_like(d, seed=args.seed)
    held_out = trials[split:]
    print(f"trials     : {len(trials)} usable, direction from {split}, evaluated on "
          f"{len(held_out)} held out")
    print(f"direction  : norm {float(d.norm()):.4f}")

    # ---- pass 2: sweep alpha ------------------------------------------------------
    points = []
    for alpha in args.alphas:
        recs, ctrls = [], []
        for tr in held_out:
            for direction, sink in ((d, recs), (d_ctrl, ctrls)):
                occ = [inject(a, direction, alpha) for a in tr["act_f"]]
                out = model.patch(tr["batch_f"], patches={site: occ}, noise=tr["noise"])
                c = direction_cosine(
                    net_translation(model.unnormalize_action(out.action)), tr["ee"], tr["anchor"]
                )
                if np.isfinite(c):
                    sink.append((c - tr["cos_f"]) / (tr["cos_w"] - tr["cos_f"]))
        if len(recs) < 5:
            continue
        est = bootstrap_mean(np.array(recs), resamples=args.resamples, seed=args.seed)
        points.append(
            TransplantPoint(
                alpha=alpha, n=len(recs), recovery=est.value,
                recovery_lo=est.lo, recovery_hi=est.hi,
                control_recovery=float(np.mean(ctrls)) if ctrls else float("nan"),
            )
        )
        print(f"  alpha={alpha:<5g} recovery {est.value:+.3f} [{est.lo:+.3f},{est.hi:+.3f}]  "
              f"random-direction control {points[-1].control_recovery:+.3f}")

    v = verdict(points)
    payload = {
        "target_site": site,
        "target_provenance": provenance,
        "n_trials_total": len(trials),
        "n_held_out": len(held_out),
        "direction_norm": float(d.norm()),
        "points": [p.as_dict() for p in points],
        **v,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"M3 BINDING TRANSPLANT  ->  {v['verdict']}")
    print("=" * 74)
    print(v["reason"])
    print(f"\nwritten -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
