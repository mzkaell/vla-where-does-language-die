#!/usr/bin/env python
"""Fit XGBoost destination probes from cached NumPy activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.interp.probe import fit_probe_nonlinear


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.input)
    y = data["y"]
    tr = data["tr"]
    te = data["te"]
    nv = data["nv"]
    site_names = [str(s) for s in data["site_names"]]

    out: dict[str, dict[str, float]] = {}
    for i, site in enumerate(site_names):
        x = data[site]
        y_shuf = y[tr].copy()
        np.random.default_rng(args.seed + i).shuffle(y_shuf)
        out[site] = {
            "acc_trained": fit_probe_nonlinear(x[tr], y[tr], x[te], y[te], seed=args.seed),
            "acc_novel": fit_probe_nonlinear(x[tr], y[tr], x[nv], y[nv], seed=args.seed),
            "acc_shuffled": fit_probe_nonlinear(x[tr], y_shuf, x[te], y[te], seed=args.seed),
        }
        if (i + 1) % 10 == 0 or i + 1 == len(site_names):
            print(f"    non-linear probe: {i + 1}/{len(site_names)} sites", flush=True)

    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
