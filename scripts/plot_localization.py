"""M4 groundwork -- turn an M2 metrics.json into the layer x component recovery map.

    python scripts/plot_localization.py results/loc_full_mps

Writes two figures next to the metrics file:
  localization_heatmap.png   layer x component grid, one panel per tower.
                             Cell value = mean recovery; significant-after-BH
                             cells get a dot. Grey = degenerate (headroom too
                             small to score, per the sensitivity trap rule).

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
    comps = [c for c in COMPONENT_ORDER if any(s["component"] == c for s in sites)]
    layers = sorted({s["layer"] for s in sites if s["layer"] is not None})
    return sites, towers, comps, layers


def plot_heatmap(data: dict, out: Path):
    sites, towers, comps, layers = site_table(data)
    n_trials = data["summary"].get("n_trials_used", "?")
    fig, axes = plt.subplots(
        1, len(towers), figsize=(1.8 + 0.55 * len(layers) * len(towers), 1.2 + 0.5 * len(comps)),
        squeeze=False,
    )
    vmax = max(0.1, np.nanmax([abs(s["recovery"]["value"]) for s in sites if s["layer"] is not None]))
    for ax, tower in zip(axes[0], towers, strict=False):
        grid = np.full((len(comps), len(layers)), np.nan)
        degen = np.zeros_like(grid, dtype=bool)
        sig = np.zeros_like(grid, dtype=bool)
        for s in sites:
            if s["tower"] != tower or s["layer"] is None or s["component"] not in comps:
                continue
            i, j = comps.index(s["component"]), layers.index(s["layer"])
            grid[i, j] = s["recovery"]["value"]
            degen[i, j] = s["degenerate"]
            sig[i, j] = s["significant_fdr"]
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(comps)):
            for j in range(len(layers)):
                if degen[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="0.8"))
                elif sig[i, j]:
                    ax.plot(j, i, "k.", ms=5)
        ax.set_title(f"{tower} tower")
        ax.set_xticks(range(len(layers)), layers)
        ax.set_yticks(range(len(comps)), comps)
        ax.set_xlabel("layer")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="recovery (patched, toward named destination)")
    fig.suptitle(
        f"M2 causal localization — recovery by site  (n={n_trials} trials, "
        f"dot = significant after BH, grey = degenerate headroom)",
        fontsize=10,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()
    data = load(args.results_dir)
    plot_heatmap(data, args.results_dir / "localization_heatmap.png")
    print(f"wrote -> {args.results_dir}/localization_heatmap.png")


if __name__ == "__main__":
    main()
