"""M4 groundwork — turn an M3 metrics.json into the dose-response figure.

    python scripts/plot_transplant.py results/m3_vlm_L8_resid_post

One panel: recovery vs alpha with bootstrap CI bars, the 0.5 readout-verdict
threshold drawn as a line, and each point labelled with its verdict. Degenerate
doses render as open markers at y=0 with a "collapsed motion" note rather than
disappearing — a dose that kills commanded motion is a result, not a gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.interp.transplant import READOUT_THRESHOLD


def plot_dose_response(data: dict, out: Path) -> None:
    verdicts = data["verdicts"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for v in verdicts:
        a = v["alpha"]
        if v["degenerate"]:
            ax.plot(a, 0, "o", mfc="none", mec="0.4")
            ax.annotate("collapsed\nmotion", (a, 0), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color="0.4")
            continue
        ax.errorbar(
            a, v["recovery_mean"],
            yerr=[[v["recovery_mean"] - v["recovery_lo"]], [v["recovery_hi"] - v["recovery_mean"]]],
            fmt="o", capsize=3, color="C0",
        )
        ax.annotate(f"{v['verdict']} (n={v['n']})", (a, v["recovery_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.axhline(READOUT_THRESHOLD, color="C3", lw=0.9, ls="--")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("alpha (injected dose of the binding delta)")
    ax.set_ylabel("recovery of the working-vs-failing gap")
    first = verdicts[0]
    ax.set_title(
        f"M3 transplant: {first['extract_site']} → {first['inject_site']}\n"
        f"(dashed = readout-verdict threshold {READOUT_THRESHOLD})",
        fontsize=9,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()
    metrics = args.results_dir / "metrics.json"
    if not metrics.exists():
        sys.exit(f"no metrics.json in {args.results_dir}")
    data = json.loads(metrics.read_text())
    if not data.get("verdicts"):
        sys.exit("metrics.json has no verdicts")
    out = args.results_dir / "transplant_dose_response.png"
    plot_dose_response(data, out)
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
