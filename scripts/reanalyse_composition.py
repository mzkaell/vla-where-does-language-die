#!/usr/bin/env python
"""Re-analysis of the compositional result in response to review.

    python scripts/reanalyse_composition.py

Reads committed per-trial data only -- no model, no GPU. Produces the four numbers the
review said were missing or wrong, and writes them to results/reanalysis/metrics.json so
the paper can cite a run rather than a transcript.

1. SUBSTITUTION EXCESS. The raw "93-96% of errors land on a trained destination" is not
   interpretable alone. For `bowl -> the rack` every possible wrong answer is trained for
   the bowl, so that cell reads 100% by construction. We report observed minus chance.

2. NAMED-DESTINATION SENSITIVITY. The review's first blocking question: could a fixed
   object-conditioned prior explain the errors? A prior predicts the same distribution
   regardless of the command. This measures whether the choice moves with what was named.

3. RACK-EXCLUSION ROBUSTNESS. The rack is weak even when trained, so the headline gap
   might be destination difficulty rather than composition.

4. DEMONSTRATION-CLUSTERED CIs. States from one episode are not independent observations;
   unclustered intervals are narrower than the data earns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.composition import (  # noqa: E402
    demo_of,
    named_destination_sensitivity,
    substitution_excess,
)
from src.eval.stats import bootstrap_mean  # noqa: E402

RUNS = ["fs_finetune", "fs_scratch80k"]


def gap(rows, keep, resamples=10_000, seed=0):
    """Trained-minus-novel accuracy, clustered by demonstration."""
    tr = [r for r in rows if not r["novel"] and keep(r)]
    nv = [r for r in rows if r["novel"] and keep(r)]
    if len(tr) < 5 or len(nv) < 5:
        return None
    # Difference of two group means; resample demonstrations within each arm.
    a = bootstrap_mean(
        np.array([float(r["correct"]) for r in tr]), resamples=resamples, seed=seed,
        cluster=[demo_of(r["trial_id"]) for r in tr],
    )
    b = bootstrap_mean(
        np.array([float(r["correct"]) for r in nv]), resamples=resamples, seed=seed,
        cluster=[demo_of(r["trial_id"]) for r in nv],
    )
    rng = np.random.default_rng(seed)
    ta = np.array([float(r["correct"]) for r in tr])
    na = np.array([float(r["correct"]) for r in nv])
    tg = np.asarray([demo_of(r["trial_id"]) for r in tr])
    ng = np.asarray([demo_of(r["trial_id"]) for r in nv])

    def draw(arr, groups):
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]
        picked = rng.integers(0, len(members), size=len(members))
        return arr[np.concatenate([members[i] for i in picked])].mean()

    diffs = np.array([draw(ta, tg) - draw(na, ng) for _ in range(resamples)])
    return {
        "trained": {"mean": float(ta.mean()), "n": len(tr), "ci": [a.lo, a.hi]},
        "novel": {"mean": float(na.mean()), "n": len(nv), "ci": [b.lo, b.hi]},
        "gap": float(ta.mean() - na.mean()),
        "gap_ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
    }


def main() -> int:
    out: dict = {"note": "demonstration-clustered CIs throughout", "runs": {}}

    for run in RUNS:
        p = REPO_ROOT / "results" / run / "per_trial.jsonl"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

        sub = substitution_excess(rows)
        sens = named_destination_sensitivity(rows)
        all_d = gap(rows, lambda r: True)
        no_rack = gap(rows, lambda r: r["named_destination"] != "the rack")
        per_dest = {
            d: gap(rows, lambda r, d=d: r["named_destination"] == d)
            for d in ["the plate", "the stove", "the rack"]
        }

        out["runs"][run] = {
            "substitution": sub,
            "named_destination_sensitivity": sens,
            "gap_all_destinations": all_d,
            "gap_excluding_rack": no_rack,
            "gap_per_destination": per_dest,
            "n_demonstrations": len({demo_of(r["trial_id"]) for r in rows}),
        }

        print(f"===== {run} =====")
        print(f"  demonstrations (bootstrap clusters): {out['runs'][run]['n_demonstrations']}")
        e = sub["excess_over_chance"]
        print(f"\n  SUBSTITUTION  observed {sub['observed_rate']:.3f}  "
              f"chance {sub['chance_rate']:.3f}")
        print(f"                excess {e['value']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"
              f"  (n={sub['n_errors']} errors)")
        print("\n  NAMED-DESTINATION SENSITIVITY (does the choice track the command?)")
        for k, v in sens["chose_the_named_destination"].items():
            print(f"    {k:34s} chose it {v:.2f} of the time")
        print("\n  GAP (trained - novel), demonstration-clustered")
        for label, g in (("all destinations", all_d), ("excluding the rack", no_rack)):
            if g:
                print(f"    {label:20s} {g['trained']['mean']:.3f} -> {g['novel']['mean']:.3f}"
                      f"   gap {g['gap']:+.3f} [{g['gap_ci'][0]:+.3f}, {g['gap_ci'][1]:+.3f}]")
        for d, g in per_dest.items():
            if g:
                sig = "" if g["gap_ci"][0] > 0 else "   n.s."
                print(f"    {d:20s} {g['trained']['mean']:.3f} -> {g['novel']['mean']:.3f}"
                      f"   gap {g['gap']:+.3f} [{g['gap_ci'][0]:+.3f}, {g['gap_ci'][1]:+.3f}]{sig}")
        print()

    d = REPO_ROOT / "results" / "reanalysis"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
