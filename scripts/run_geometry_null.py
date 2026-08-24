#!/usr/bin/env python
"""Geometry-only null: how much of the policy's destination choice is just geometry?

    python scripts/run_geometry_null.py

The review's remaining methodological objection. Our readout scores which destination
anchor the predicted motion heads toward, and in a fixed scene destination identity is
correlated with geometry, so a policy conditioning only on where things are could
reproduce part of the pattern without grounding anything.

This fits a classifier that predicts **the model's own chosen destination** from
non-linguistic features alone: end-effector position, distance and unit direction to each
anchor, and object identity. It never sees the instruction. Two numbers follow.

1. How well geometry alone predicts the model's choice. High accuracy means much of the
   behaviour is explained without reference to language.
2. Whether geometry can reproduce the *command dependence*. It cannot, by construction --
   the classifier has no command input, so its prediction is identical across the
   instructions given at one state. Quantifying that gap is the point: it is the part of
   our result a geometric account cannot reach, and the paper currently asserts this
   without measuring it.

No model forward passes: features come from the stored LIBERO states, labels from the
committed per-trial records. Runs in minutes on CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_composition import task_endpoints  # noqa: E402
from src.eval.composition import build_anchors, demo_of  # noqa: E402

RUNS = ["fs_finetune", "fs_scratch80k"]
SUITE = REPO_ROOT / "data" / "libero" / "libero_goal"


def ee_lookup(tasks: set[str]) -> dict[tuple[str, str, int], np.ndarray]:
    """(task, demo, timestep) -> end-effector position, read straight from LIBERO."""
    out: dict[tuple[str, str, int], np.ndarray] = {}
    for task in tasks:
        p = SUITE / f"{task}_demo.hdf5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as h:
            for demo in h["data"]:
                ee = h["data"][demo]["obs"]["ee_pos"][:]
                for t in range(ee.shape[0]):
                    out[(task, demo, t)] = ee[t]
    return out


def featurise(ee: np.ndarray, obj: str, anchors: dict[str, np.ndarray]) -> np.ndarray:
    """Non-linguistic features only. No component of this depends on the instruction."""
    feats = [ee]
    for _, a in sorted(anchors.items()):
        d = a - ee
        n = float(np.linalg.norm(d))
        feats.append([n])
        feats.append(d / n if n > 1e-9 else np.zeros(3))
    feats.append([1.0 if obj == "bowl" else 0.0])
    return np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in feats])


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    anchors = build_anchors(task_endpoints(SUITE))
    dests = sorted(anchors)
    out: dict = {"destinations": dests, "runs": {}}

    for run in RUNS:
        p = REPO_ROOT / "results" / run / "per_trial.jsonl"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

        tasks = {r["trial_id"].split("__")[0] for r in rows}
        ee = ee_lookup(tasks)

        X, y_model, groups, kept = [], [], [], []
        for r in rows:
            task, demo, tstr = r["trial_id"].split("__")[:3]
            key = (task, demo, int(tstr.lstrip("t")))
            if key not in ee:
                continue
            X.append(featurise(ee[key], r["object"], anchors))
            y_model.append(dests.index(r["chosen_destination"]))
            groups.append(demo_of(r["trial_id"]))
            kept.append(r)
        X = np.stack(X)
        y_model = np.array(y_model)
        groups = np.asarray(groups)

        # Split by episode, as everywhere else in this paper.
        rng = np.random.default_rng(0)
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        train_g = set(uniq[: int(0.7 * len(uniq))])
        tr = np.array([i for i, g in enumerate(groups) if g in train_g])
        te = np.array([i for i, g in enumerate(groups) if g not in train_g])

        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=0)
        clf.fit(sc.transform(X[tr]), y_model[tr])
        acc = float(clf.score(sc.transform(X[te]), y_model[te]))

        # Majority-class floor: predicting the single most common destination.
        counts = np.bincount(y_model[te], minlength=len(dests))
        floor = float(counts.max() / counts.sum())

        # The decisive quantity. At one state the policy is given several instructions and
        # selects different destinations; a geometry-only model gives that state ONE
        # prediction. So the fraction of states where the policy's choice actually varies
        # with the command is an upper bound on what geometry can ever explain.
        by_state: dict[tuple, set] = {}
        for r in kept:
            skey = tuple(r["trial_id"].split("__")[:3])
            by_state.setdefault(skey, set()).add(r["chosen_destination"])
        varies = sum(1 for v in by_state.values() if len(v) > 1) / len(by_state)

        out["runs"][run] = {
            "n_trials": int(len(y_model)),
            "n_states": len(by_state),
            "geometry_only_accuracy": acc,
            "majority_class_floor": floor,
            "states_where_choice_varies_with_command": varies,
            "n_features": int(X.shape[1]),
        }
        print(f"===== {run} =====")
        print(f"  trials {len(y_model)}, states {len(by_state)}, features {X.shape[1]}")
        print(f"  geometry-only prediction of the model's choice : {acc:.3f}")
        print(f"  majority-class floor                           : {floor:.3f}")
        print(f"  states where the choice varies with the command: {varies:.3f}")
        print("  (geometry gives one prediction per state, so it cannot account for")
        print("   any of that variation)\n")

    d = REPO_ROOT / "results" / "geometry_null"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
