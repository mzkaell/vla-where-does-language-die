#!/usr/bin/env python
"""Every headline number in the paper, recomputed from committed results.

    python scripts/audit_paper_numbers.py

The paper's own standard is that no number exists unless a run produced it. Enforcing that
by reading is unreliable: three separate errors survived repeated proofreading and were
found only when someone recomputed the value.

  * Table 1 said the compositional gap was +0.358. The appendix said +0.250. They were
    different estimators and the table used the confounded one (artifact 7).
  * The paper said every interval was a cluster bootstrap over episodes. The neutral and
    conflict intervals were not clustered.
  * The paper reported a patching replication of +0.046 and MPS/CPU divergences of 0.21
    and 0.014. None of the three could be reproduced from any committed run.

So this recomputes each claim from results/ and diffs it against what the paper says. Each
entry names the claim, the value in the paper, and a function returning the value the data
gives. Exits non-zero on any mismatch, so it can gate a submission.

Add an entry whenever a number enters the paper. An entry that cannot be written -- because
no committed run produces the number -- is itself the finding.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.stats import bootstrap_mean  # noqa: E402

TOL = 0.0015  # values are quoted to three decimals


def _metrics(run: str) -> dict:
    return json.loads((REPO_ROOT / "results" / run / "metrics.json").read_text())


def _trials(run: str) -> list[dict]:
    p = REPO_ROOT / "results" / run / "per_trial.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _pairs(run: str) -> list[dict]:
    p = REPO_ROOT / "results" / run / "per_pair.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _episode_of(pair_id: str) -> str:
    m = re.search(r"__(demo_\d+)__t\d+$", pair_id)
    parts = pair_id.split("__")
    return parts[-3] + "/" + m.group(1) if m else pair_id


def _ifr(run: str) -> tuple[float, float, float]:
    """Directional IFR with the episode-clustered bootstrap the paper claims to use."""
    rows = _pairs(run)
    x = np.array([float(r["followed_instruction"]) for r in rows])
    est = bootstrap_mean(
        x, resamples=10_000, seed=0, cluster=[_episode_of(r["pair_id"]) for r in rows]
    )
    return est.value, est.lo, est.hi


def _matched_gap(run: str, drop: set[str] = frozenset()) -> float:
    """Trained minus novel, over destinations occurring in BOTH arms (artifact 7)."""
    rows = _trials(run)
    shared = {r["named_destination"] for r in rows if r["novel"]} & {
        r["named_destination"] for r in rows if not r["novel"]
    }
    keep = shared - set(drop)
    tr = [r["correct"] for r in rows if not r["novel"] and r["named_destination"] in keep]
    nv = [r["correct"] for r in rows if r["novel"] and r["named_destination"] in keep]
    return float(np.mean(tr) - np.mean(nv))


def _recovery(run: str) -> dict[str, float]:
    return {
        s["site"]: s["recovery"]["value"]
        for s in _metrics(run)["sites"]
        if "recovery" in s and s["recovery"]["value"] == s["recovery"]["value"]
    }


def _sweep_diff(novel: str, control: str) -> float:
    a, b = _recovery(novel), _recovery(control)
    return float(np.mean([a[k] - b[k] for k in sorted(set(a) & set(b))]))


def _bh_survivors() -> float:
    runs = ["loc_finetune", "locctl_finetune", "loc_full_mps", "locctl_anant_contrasts"]
    return float(
        sum(
            1
            for r in runs
            for s in _metrics(r)["sites"]
            if s.get("significant_bh") or s.get("survives_bh")
        )
    )


def _expert_sites(run: str, pred: Callable[[float], bool]) -> float:
    return float(
        sum(
            1
            for s in _metrics(run)["sites"]
            if s["site"].startswith("expert.") and pred(s["acc_novel"])
        )
    )


def _backend_max_diff(component: str) -> float:
    a, m = _recovery("loc_finetune"), _recovery("loc_full_mps")
    return max(
        abs(a[k] - m[k]) for k in sorted(set(a) & set(m)) if k.split(".")[-1] == component
    )


# claim, value as printed in the paper, recomputation
CHECKS: list[tuple[str, float, Callable[[], float]]] = [
    # -- Table 1 and abstract: the three regimes -----------------------------------
    ("neutral IFR, ckpt A",            0.745, lambda: _ifr("m0_v2_k1000dai")[0]),
    ("neutral IFR lo (clustered)",     0.688, lambda: _ifr("m0_v2_k1000dai")[1]),
    ("neutral IFR hi (clustered)",     0.801, lambda: _ifr("m0_v2_k1000dai")[2]),
    ("conflict IFR, ckpt A",           0.946, lambda: _ifr("conflict_finetune")[0]),
    ("conflict IFR lo (clustered)",    0.916, lambda: _ifr("conflict_finetune")[1]),
    ("conflict IFR hi (clustered)",    0.974, lambda: _ifr("conflict_finetune")[2]),
    ("conflict IFR, ckpt B",           0.929, lambda: _ifr("conflict_scratch80k")[0]),
    # -- compositional gap, matched to shared destinations -------------------------
    ("compositional gap, ckpt A",      0.250, lambda: _matched_gap("fs_finetune")),
    ("compositional gap, ckpt B",      0.193, lambda: _matched_gap("fs_scratch80k")),
    ("gap excl. rack, ckpt A",         0.215, lambda: _matched_gap("fs_finetune", {"the rack"})),
    ("gap excl. rack, ckpt B",         0.160, lambda: _matched_gap("fs_scratch80k", {"the rack"})),
    ("confounded gap quoted in art.7", 0.358, lambda: _confounded("fs_finetune")),
    ("paraphrase gap, ckpt A",         0.227, lambda: _matched_gap("para_finetune")),
    ("paraphrase gap, ckpt B",         0.150, lambda: _matched_gap("para_scratch80k")),
    # -- substitution and command dependence ---------------------------------------
    ("substitution excess, ckpt A",    0.111,
     lambda: _sub("fs_finetune")),
    ("substitution excess, ckpt B",    0.098,
     lambda: _sub("fs_scratch80k")),
    ("bottle told plate, ckpt A",      0.450,
         lambda: _named("fs_finetune", "wine bottle told 'the plate'")),
    ("bottle told stove, ckpt A",      0.560,
         lambda: _named("fs_finetune", "wine bottle told 'the stove'")),
    ("bottle told plate, ckpt B",      0.340,
         lambda: _named("fs_scratch80k", "wine bottle told 'the plate'")),
    ("bottle told stove, ckpt B",      0.670,
         lambda: _named("fs_scratch80k", "wine bottle told 'the stove'")),
    ("bowl to rack, ckpt A",           0.010,
         lambda: _named("fs_finetune", "bowl told 'the rack'")),
    # -- the two negative results ---------------------------------------------------
    ("sweep novel-control (CUDA)",    -0.056,
         lambda: _sweep_diff("loc_finetune", "locctl_finetune")),
    ("sweep novel-control (MPS)",     -0.041,
         lambda: _sweep_diff("loc_full_mps", "locctl_anant_contrasts")),
    ("sites surviving BH, all runs",   0.0,   _bh_survivors),
    ("expert sites >0.5, ckpt A",     38.0,
         lambda: _expert_sites("probe_big_grouped", lambda a: a > 0.5)),
    ("expert sites >0.5, ckpt B",      2.0,
         lambda: _expert_sites("probe_big_grouped_scratch80k", lambda a: a > 0.5)),
    ("expert sites <0.10, ckpt B",    29.0,
         lambda: _expert_sites("probe_big_grouped_scratch80k", lambda a: a < 0.10)),
    # -- backend divergence ----------------------------------------------------------
    ("CUDA/MPS max diff, resid_post",  0.136, lambda: _backend_max_diff("resid_post")),
    ("CUDA/MPS max diff, attn_out",    0.078, lambda: _backend_max_diff("attn_out")),
    ("CUDA/MPS max diff, mlp_out",     0.044, lambda: _backend_max_diff("mlp_out")),
    # -- geometry null ----------------------------------------------------------------
    ("geometry-only acc, ckpt A",      0.546,
     lambda: _metrics("geometry_null")["runs"]["fs_finetune"]["geometry_only_accuracy"]),
    ("geometry floor, ckpt A",         0.496,
     lambda: _metrics("geometry_null")["runs"]["fs_finetune"]["majority_class_floor"]),
]


def _confounded(run: str) -> float:
    rows = _trials(run)
    tr = [r["correct"] for r in rows if not r["novel"]]
    nv = [r["correct"] for r in rows if r["novel"]]
    return float(np.mean(tr) - np.mean(nv))


def _sub(run: str) -> float:
    return _metrics("reanalysis")["runs"][run]["substitution"]["excess_over_chance"]["value"]


def _named(run: str, key: str) -> float:
    return _metrics("reanalysis")["runs"][run]["named_destination_sensitivity"][
        "chose_the_named_destination"
    ][key]


def main() -> int:
    bad = 0
    print(f"{'claim':38s}{'paper':>10s}{'data':>10s}   status")
    print("-" * 74)
    for name, printed, fn in CHECKS:
        try:
            got = fn()
        except Exception as exc:  # a missing run is a finding, not a crash
            print(f"{name:38s}{printed:>10.3f}{'--':>10s}   CANNOT COMPUTE: {exc}")
            bad += 1
            continue
        ok = abs(got - printed) <= TOL
        bad += not ok
        print(f"{name:38s}{printed:>10.3f}{got:>10.3f}   {'ok' if ok else '<<< MISMATCH'}")
    print("-" * 74)
    print(f"{len(CHECKS) - bad}/{len(CHECKS)} agree with the paper")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
