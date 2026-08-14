"""M4 groundwork -- turn an M2 metrics.json into the layer x component recovery map.

    python scripts/plot_localization.py results/loc_full_mps

Writes two figures next to the metrics file:
  localization_heatmap.png   layer x component grid, one panel per tower.
                             Cell value = mean recovery; significant-after-BH
                             cells get a dot. Grey = degenerate (headroom too
                             small to score, per the sensitivity trap rule).
  localization_depth.png     recovery vs layer, one line per component, one
                             panel per tower, bootstrap CI band.

Both figures state n (trials) and the null size in the caption strip, because a
map without its evidence budget invites over-reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMPONENT_ORDER = ["resid_pre", "attn_out", "mlp_out", "resid_post", "final_norm"]
TOWER_ORDER = ["vlm", "expert"]


def load(results_dir: Path) -> dict:
    metrics = results_dir / "metrics.json"
    if not metrics.exists():
        sys.exit(f"no metrics.json in {results_dir}")
    return json.loads(metrics.read_text())


def site_table(data: dict):
    """-> (towers, components, layers, grid dicts) present in this run."""
    sites = data["sites"]
    towers = [t for t in TOWER_ORDER if any(s["tower"] == t for s in sites)]
    comps = [
        c for c in COMPONENT_ORDER
        if any(s["component"] == c and s["layer"] is not None for s in sites)
    ]
    layers = sorted({s["layer"] for s in sites if s["layer"] is not None})
    return sites, towers, comps, layers


def plot_heatmap(data: dict, out: Path):
    sites, towers, comps, layers = site_table(data)
    n_trials = data["summary"].get("n_trials_used", "?")
    fig, axes = plt.subplots(
        1, len(towers), figsize=(1.8 + 0.55 * len(layers) * len(towers), 1.2 + 0.5 * len(comps)),
        squeeze=False,
    )
    # degenerate sites' recoveries are documented as meaningless and can be huge
    # (headroom floor 0.05 admits |recovery| ~ 40); one such value would flatten
    # every real cell to near-white, so they set neither the scale nor a cell
    vals = [
        abs(s["recovery"]["value"])
        for s in sites
        if not s["degenerate"] and (s["layer"] is not None or s["component"] == "final_norm")
    ]
    finite = [v for v in vals if np.isfinite(v)]
    vmax = max(0.1, max(finite)) if finite else 0.1
    has_fn = any(s["component"] == "final_norm" for s in sites)
    for ax, tower in zip(axes[0], towers, strict=False):
        rows = comps + (["final_norm"] if has_fn else [])
        cols = list(layers) + (["FN"] if has_fn else [])
        grid = np.full((len(rows), len(cols)), np.nan)
        degen = np.zeros_like(grid, dtype=bool)
        sig = np.zeros_like(grid, dtype=bool)
        for s in sites:
            if s["tower"] != tower:
                continue
            if s["component"] == "final_norm":
                # one site, no layer: render in its own row/column so it stays
                # visible -- it sits directly on the readout path and is a live
                # candidate for the M3 extract site
                i, j = len(comps), len(layers)
            elif s["layer"] is not None and s["component"] in comps:
                i, j = comps.index(s["component"]), layers.index(s["layer"])
            else:
                continue
            grid[i, j] = s["recovery"]["value"]
            degen[i, j] = s["degenerate"]
            sig[i, j] = s["significant_fdr"]
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(cols)):
                if degen[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="0.8"))
                elif sig[i, j]:
                    ax.plot(j, i, "k.", ms=5)
        ax.set_title(f"{tower} tower")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("layer")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="recovery (patched, toward named destination)")
    fig.suptitle(
        f"M2 causal localization — recovery by site  (n={n_trials} trials, "
        f"dot = significant after BH, grey = degenerate headroom)",
        fontsize=10,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_depth(data: dict, out: Path):
    sites, towers, comps, layers = site_table(data)
    n_trials = data["summary"].get("n_trials_used", "?")
    fig, axes = plt.subplots(1, len(towers), figsize=(5.2 * len(towers), 3.4), squeeze=False, sharey=True)
    for ax, tower in zip(axes[0], towers, strict=False):
        for comp in comps:
            xs, ys, los, his = [], [], [], []
            for s in sorted(
                (s for s in sites if s["tower"] == tower and s["component"] == comp and s["layer"] is not None),
                key=lambda s: s["layer"],
            ):
                xs.append(s["layer"])
                if s["degenerate"]:
                    # NaN breaks the line: a silent skip would bridge straight
                    # across the one layer where the number is meaningless
                    ys.append(float("nan")); los.append(float("nan")); his.append(float("nan"))
                else:
                    ys.append(s["recovery"]["value"])
                    los.append(s["recovery"]["lo"])
                    his.append(s["recovery"]["hi"])
            if not xs:
                continue
            (line,) = ax.plot(xs, ys, marker="o", ms=3, label=comp)
            ax.fill_between(xs, los, his, alpha=0.15, color=line.get_color())
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(f"{tower} tower")
        ax.set_xlabel("layer")
    axes[0][0].set_ylabel("recovery")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"M2 recovery vs depth  (n={n_trials} trials, band = bootstrap CI)", fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()
    data = load(args.results_dir)
    plot_heatmap(data, args.results_dir / "localization_heatmap.png")
    plot_depth(data, args.results_dir / "localization_depth.png")
    print(f"wrote -> {args.results_dir}/localization_heatmap.png")
    print(f"wrote -> {args.results_dir}/localization_depth.png")


if __name__ == "__main__":
    main()
