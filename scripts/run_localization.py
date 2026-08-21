#!/usr/bin/env python
"""M2 — causal localization sweep over (layer x tower x component).

    # smoke test first, on CPU, ~5 min:
    python scripts/run_localization.py --checkpoint k1000dai/smolvla_libero_finetune \
        --n-trials 3 --sites-limit 8 --run-id _loc_smoke

    # full sweep, GPU:
    python scripts/run_localization.py --checkpoint k1000dai/smolvla_libero_finetune \
        --n-trials 40 --device cuda --run-id loc_finetune

Method and the two hazards it guards against: see src/interp/localization.py.

Short version. The behavioural result gives a matched pair of runs that differ only in
whether the object<->destination pairing was demonstrated:

    working:  "put the wine bottle on the rack"   (trained)
    failing:  "put the bowl on the rack"          (novel)

Same destination word, same state. We patch each site from the working run into the failing
run and ask how much of the gap toward the named destination it closes.

COST. Each trial costs 2 baseline forwards + one forward per site. With 130 sites that is
~132 forwards per trial, so `--n-trials 40` is ~5300 forwards. Budget from the smoke test's
reported rate before launching; the script prints an ETA after the first trial.
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
from src.eval.composition import build_anchors, instruction_for  # noqa: E402
from src.interp.localization import (  # noqa: E402
    TrialBaseline,
    aggregate_site,
    apply_fdr,
    direction_cosine,
    localization_summary,
    net_translation,
    recovery_fraction,
)

# The matched contrast. Same destination word; the pairing is trained for one object and
# never demonstrated for the other. `target` is the destination whose anchor we score
# movement toward.
CONTRASTS = [
    {"target": "the rack",  "working": ("wine bottle", "the rack"),
     "failing": ("bowl", "the rack")},
    {"target": "the plate", "working": ("bowl", "the plate"),
     "failing": ("wine bottle", "the plate")},
    {"target": "the stove", "working": ("bowl", "the stove"),
     "failing": ("wine bottle", "the stove")},
]

# CONTROL: both arms are TRAINED pairings, so there is no binding failure to recover --
# only the ordinary causal structure of the network.
#
# This exists because the raw recovery profile is confounded. Patching an early layer
# propagates through everything downstream while patching a late layer changes almost
# nothing, so recovery rises toward the output for reasons that have nothing to do with
# where the binding lives. The first sweep showed exactly that shape, and it would look
# the same in a model with no binding failure at all.
#
# Running the identical sweep on a contrast the model handles correctly measures that
# artefact directly. Subtracting the two profiles cancels it and leaves the component
# specific to the novel-pairing failure.
CONTROL_CONTRASTS = [
    {"target": "the plate", "working": ("bowl", "the plate"),
     "failing": ("bowl", "the stove")},
    {"target": "the stove", "working": ("bowl", "the stove"),
     "failing": ("bowl", "the plate")},
    {"target": "the rack",  "working": ("wine bottle", "the rack"),
     "failing": ("wine bottle", "top of the cabinet")},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--n-trials", type=int, default=40, help="states per contrast")
    ap.add_argument("--sites-limit", type=int, default=None, help="first N sites (smoke test)")
    ap.add_argument("--components", default=None,
                    help="comma-separated component filter, e.g. 'resid_post' or "
                         "'resid_post,final_norm'. Cuts cost ~4x while keeping the full "
                         "depth profile of both towers, which is what the drop-off "
                         "question needs. Applied before --sites-limit.")
    ap.add_argument("--towers", default=None,
                    help="comma-separated tower filter: 'vlm', 'expert', or both")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--null-sites", type=int, default=12, help="sites used to build the null")
    ap.add_argument("--min-trials", type=int, default=5,
                    help="minimum scored trials before a site gets a CI (lower for smoke tests)")
    ap.add_argument(
        "--contrast-mode",
        choices=["novel", "control"],
        default="novel",
        help=(
            "novel: trained-vs-novel pairing (the effect). "
            "control: both arms trained, which isolates the causal-proximity artefact "
            "so it can be subtracted from the novel profile."
        ),
    )
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    suite_dir = args.data_root / args.suite
    run_id = args.run_id or f"loc_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPO_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    contrasts = CONTRASTS if args.contrast_mode == "novel" else CONTROL_CONTRASTS
    anchors = build_anchors(task_endpoints(suite_dir))

    from src.models.smolvla import SmolVLA, make_batch, pair_pad_length

    print(f"run_id     : {run_id}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {args.device}")
    model = SmolVLA.load(args.checkpoint, device=args.device)
    cfg = model.config
    keys = list(cfg.image_features)
    size = cfg.image_features[keys[0]].shape[-1]
    sdim = cfg.robot_state_feature.shape[0]

    sites = model.sites()
    # names validate against the model's FULL site table, not the residue of an
    # earlier filter -- otherwise a valid tower name can be refused as "unknown"
    # just because the component filter emptied it first
    known_components = {s.component for s in sites}
    known_towers = {s.tower for s in sites}
    if args.components:
        wanted = {c.strip() for c in args.components.split(",")}
        if wanted - known_components:
            sys.exit(f"unknown component(s) {sorted(wanted - known_components)}; "
                     f"known: {sorted(known_components)}")
        sites = [s for s in sites if s.component in wanted]
    if args.towers:
        wanted_towers = {c.strip() for c in args.towers.split(",")}
        if wanted_towers - known_towers:
            sys.exit(f"unknown tower(s) {sorted(wanted_towers - known_towers)}; "
                     f"known: {sorted(known_towers)}")
        sites = [s for s in sites if s.tower in wanted_towers]
    if not sites:
        sys.exit("the filter combination selects zero sites")
    if args.sites_limit is not None:
        sites = sites[: args.sites_limit]
    print(f"sites      : {len(sites)}   norm stats: {model.has_norm_stats}")

    (out_dir / "config.yaml").write_text(
        "\n".join(
            f"{k}: {json.dumps(v)}"
            for k, v in {
                "run_id": run_id,
                "experiment": "M2_localization",
                "checkpoint": args.checkpoint,
                "n_trials_per_contrast": args.n_trials,
                "n_sites": len(sites),
                "contrast_mode": args.contrast_mode,
                "contrasts": contrasts,
                "components_filter": getattr(args, "components", None),
                "towers_filter": getattr(args, "towers", None),
                "seed": args.seed,
                "fdr": args.fdr,
                "device": args.device,
                "image_orientation": "rot180",
                "state_composition": list(STATE_KEYS),
                "platform": platform.platform(),
                "torch": torch.__version__,
            }.items()
        ),
        encoding="utf-8",
    )

    # site -> per-trial lists
    rec: dict[str, list[float]] = {s.name: [] for s in sites}
    cosp: dict[str, list[float]] = {s.name: [] for s in sites}
    disp: dict[str, list[float]] = {s.name: [] for s in sites}
    null_recoveries: list[float] = []
    baselines: list[TrialBaseline] = []
    skipped = 0

    t0 = time.time()
    n_done = 0

    for contrast in contrasts:
        dest = contrast["target"]
        anchor = anchors[dest]
        # Pool states from every task so both arms see the same distribution.
        from src.eval.composition import OBJECT_SOURCE_TASKS

        pooled = [t for ts in OBJECT_SOURCE_TASKS.values() for t in ts]
        states = collect_states(suite_dir, pooled, args.n_trials, args.seed)[: args.n_trials]
        wo, wd = contrast["working"]
        fo, fd = contrast["failing"]
        instr_w = instruction_for(wo, wd)
        instr_f = instruction_for(fo, fd)
        # 'longest'-padded checkpoints need the pair padded to a common length or
        # cross-run patches fail on a token-dim mismatch (see pair_pad_length). Our
        # checkpoints use max_length/48 so this is a no-op for them, but the control
        # contrasts pair instructions with different word counts, so keep the guard.
        pad_len = pair_pad_length(model.policy, [instr_w, instr_f])
        print(
            f"\ntarget '{dest}' | working: {wo} -> {wd} | failing: {fo} -> {fd} "
            f"({len(states)} states)"
        )

        for st in states:
            obs = {k: st[k] for k in ("agentview_rgb", "eye_in_hand_rgb")}
            obs |= {k: st[k] for k in STATE_KEYS}
            images = build_images(obs, keys, size)
            state_t = model.normalize_state(build_state(obs, sdim))
            ee = np.asarray(st["ee_pos"], dtype=np.float64)
            noise = model.make_noise(1)

            batch_w = make_batch(
                images, state_t, instr_w, model.policy, args.device,
                pad_to_length=pad_len,
            )
            batch_f = make_batch(
                images, state_t, instr_f, model.policy, args.device,
                pad_to_length=pad_len,
            )

            site_names = [s.name for s in sites]
            run_w = model.forward_with_cache(batch_w, sites=site_names, noise=noise)
            run_f = model.forward_with_cache(batch_f, sites=(), noise=noise)

            cos_w = direction_cosine(
                net_translation(model.unnormalize_action(run_w.action)), ee, anchor
            )
            cos_f = direction_cosine(
                net_translation(model.unnormalize_action(run_f.action)), ee, anchor
            )
            base = TrialBaseline(
                trial_id=f"{st['task']}__{st['demo']}__t{st['t']}__{dest}",
                cos_working=cos_w, cos_failing=cos_f,
                headroom=cos_w - cos_f, ee_pos=ee, anchor=anchor,
            )
            baselines.append(base)
            if not base.usable:
                skipped += 1
                continue

            for s in sites:
                patched = model.patch(
                    batch_f, patches={s.name: run_w.occurrences(s.name)}, noise=noise
                )
                d = net_translation(model.unnormalize_action(patched.action))
                c = direction_cosine(d, ee, anchor)
                rec[s.name].append(recovery_fraction(c, base))
                cosp[s.name].append(c)
                disp[s.name].append(float(np.linalg.norm(d)))

            # Position-shuffled null: same activation values, correspondence destroyed.
            # A site that only looks causal because patching perturbs anything at all
            # scores just as high here.
            for s in rng.choice(sites, size=min(args.null_sites, len(sites)), replace=False):
                occ = [t[:, torch.randperm(t.shape[1]), :] for t in run_w.occurrences(s.name)]
                patched = model.patch(batch_f, patches={s.name: occ}, noise=noise)
                c = direction_cosine(
                    net_translation(model.unnormalize_action(patched.action)), ee, anchor
                )
                r = recovery_fraction(c, base)
                if np.isfinite(r):
                    null_recoveries.append(r)

            n_done += 1
            if n_done == 1 or n_done % 5 == 0:
                per = (time.time() - t0) / n_done
                todo = len(CONTRASTS) * len(states) - n_done
                print(f"    trial {n_done}  ({per:.1f}s/trial, eta {per * todo / 60:.1f} min)",
                      flush=True)

    if not baselines:
        print("no trials", file=sys.stderr)
        return 1

    effects = [
        aggregate_site(
            s.name, s.tower, s.component, s.layer,
            rec[s.name], cosp[s.name], disp[s.name],
            resamples=args.resamples, seed=args.seed, min_trials=args.min_trials,
        )
        for s in sites
    ]
    effects = apply_fdr(effects, np.array(null_recoveries), fdr=args.fdr)
    summary = localization_summary(effects)
    summary |= {
        "n_trials_used": len(baselines) - skipped,
        "n_trials_skipped_no_headroom": skipped,
        "null_n": len(null_recoveries),
        "null_mean": float(np.mean(null_recoveries)) if null_recoveries else float("nan"),
        "mean_headroom": float(np.mean([b.headroom for b in baselines if b.usable]))
        if any(b.usable for b in baselines) else float("nan"),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "sites": [e.as_dict() for e in effects]}, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 74)
    print("M2 LOCALIZATION")
    print("=" * 74)
    print(f"trials used {summary['n_trials_used']}  (skipped {skipped}: working run did not "
          f"beat failing, so no gap to close)")
    print(f"mean headroom cos(working)-cos(failing): {summary['mean_headroom']:.3f}")
    print(f"null: n={summary['null_n']}  mean recovery {summary['null_mean']:.3f}")
    print(f"sites scored {summary['n_scored']}/{summary['n_sites']}  "
          f"degenerate {summary.get('n_degenerate', 0)}  "
          f"significant after BH {summary['n_significant_fdr']}")
    if summary.get("n_scored"):
        print(f"\nmax recovery {summary['max_recovery']:+.3f}")
        print(f"effect share in top 10% of sites: {summary['effect_share_in_top_10pct']:.3f} "
              f"-> {'NARROW' if summary['localization_narrow'] else 'diffuse'}")
        bt = "  ".join(f"{k}={v:+.3f}" for k, v in summary["by_tower"].items())
        bc = "  ".join(f"{k}={v:+.3f}" for k, v in summary["by_component"].items())
        print(f"\nby tower:     {bt}")
        print(f"by component: {bc}")
        print("\ntop sites by recovery:")
        for e in sorted(
            [x for x in effects if np.isfinite(x.recovery_mean) and not x.degenerate],
            key=lambda x: -x.recovery_mean,
        )[:12]:
            mark = "*" if e.significant_fdr else " "
            print(f"  {e.site:28s} {e.recovery_mean:+.3f} "
                  f"[{e.recovery_lo:+.3f},{e.recovery_hi:+.3f}]{mark} p={e.p_vs_null:.4f}")
        print("  * significant vs position-shuffled null after Benjamini-Hochberg")

    print(f"\nwritten -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
