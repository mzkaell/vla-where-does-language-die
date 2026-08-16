"""Publication figures, built only from committed runs in `results/`.

Two figures, each carrying an argument the prose cannot make on its own:

**Fig 2 — the localization null.** The sweep's layer profile next to the *control*
profile (both instructions trained, so there is no failure to recover). The control shows
the same rise toward the output, and in fact sits above the novel curve at every layer, so
the novel-specific difference is flat-to-negative. The figure IS the argument that the
sweep measured causal proximity rather than mechanism; a reader shown only the novel curve
would reasonably conclude the opposite.

**Fig 3 — the probe.** Destination decodability for novel pairings through the action
expert, against its own label-shuffled control. Absent early, peaks at layer 7, stays
above chance to the output while the action goes elsewhere.

Design notes
------------
* Two categorical hues only, validated colourblind-safe (blue/orange, worst-pair
  ΔE 24.7 protan against a target of 8). Gray is used *only* for non-series reference
  lines (chance, zero) -- it deliberately fails a categorical chroma floor, which is the
  point: it must recede.
* Identity never rests on colour alone: every series is directly labelled at its right
  end, and marker shape differs.
* One y-axis per panel. Recessive grid, no chartjunk, thin marks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

SERIES_A = "#2a78d6"  # blue   -- the condition of interest
SERIES_B = "#eb6834"  # orange -- its control
REFERENCE = "#8a8985"  # gray  -- chance / zero lines only, never a series
INK = "#0b0b0b"
INK_SOFT = "#52514e"


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK_SOFT,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _load(run: str) -> dict[str, Any] | None:
    p = REPO_ROOT / "results" / run / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _profile(data: dict[str, Any], tower: str, component: str) -> list[float | None]:
    S = {s["site"]: s for s in data["sites"]}
    out = []
    for L in range(16):
        s = S.get(f"{tower}.L{L}.{component}")
        v = s["recovery"]["value"] if s and "recovery" in s else None
        out.append(v if (v is not None and v == v) else None)
    return out


def fig_sweep(out_path: Path) -> bool:
    """Fig 2: the novel sweep and its control coincide, so the profile is an artifact."""
    import matplotlib.pyplot as plt

    nov, ctl = _load("loc_finetune"), _load("locctl_finetune")
    if not nov or not ctl:
        print("fig2: missing loc_finetune / locctl_finetune")
        return False

    n = _profile(nov, "expert", "resid_post")
    c = _profile(ctl, "expert", "resid_post")
    xs = list(range(16))
    diff = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(n, c, strict=False)
    ]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.grid(axis="y", zorder=0)
    ax.axhline(0, color=REFERENCE, lw=0.9, zorder=1)
    ax.plot(xs, n, color=SERIES_A, marker="o", ms=3.2, zorder=3)
    ax.plot(xs, c, color=SERIES_B, marker="s", ms=3.0, zorder=3)
    ax.plot(xs, diff, color=REFERENCE, ls=":", lw=1.3, zorder=2)

    # Direct labels: identity never rests on colour alone.
    ax.annotate("novel", (15, n[15]), xytext=(3, 3), textcoords="offset points",
                color=SERIES_A, fontsize=7, fontweight="bold")
    ax.annotate("control", (15, c[15]), xytext=(3, -9), textcoords="offset points",
                color=SERIES_B, fontsize=7, fontweight="bold")
    ax.annotate("difference", (10, diff[10]), xytext=(2, -12), textcoords="offset points",
                color=INK_SOFT, fontsize=6.5)

    ax.set_xlabel("action-expert layer")
    ax.set_ylabel("patching recovery (resid_post)")
    # Accurate rather than flattering. The control does not merely resemble the novel
    # profile -- it sits ABOVE it at every layer, so the rise toward the output cannot be
    # the failure we are trying to localize, and the novel-specific difference is if
    # anything negative. Claiming the curves "coincide" would overstate it.
    ax.set_title("Recovery rises toward the output even\nwith no failure to recover", loc="left")
    ax.set_xticks([0, 4, 8, 12, 15])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def fig_probe(out_path: Path) -> bool:
    """Fig 3: the destination is decodable inside the expert, and unused."""
    import matplotlib.pyplot as plt

    d = _load("probe_big_grouped")
    if not d:
        print("fig3: missing probe_big_grouped")
        return False
    S = {s["site"]: s for s in d["sites"]}
    xs = list(range(16))
    acc = [S[f"expert.L{L}.resid_post"]["acc_novel"] for L in xs]
    shuf = [S[f"expert.L{L}.resid_post"]["acc_shuffled"] for L in xs]
    chance = S["expert.L0.resid_post"]["chance"]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.grid(axis="y", zorder=0)
    ax.axhline(chance, color=REFERENCE, lw=0.9, ls="--", zorder=1)
    ax.annotate("chance", (11.5, chance), xytext=(0, -11), textcoords="offset points",
                color=INK_SOFT, fontsize=6.5)
    ax.plot(xs, acc, color=SERIES_A, marker="o", ms=3.2, zorder=3)
    ax.plot(xs, shuf, color=SERIES_B, marker="s", ms=3.0, zorder=3)

    ax.annotate("novel pairings", (8, acc[8]), xytext=(-6, 6), textcoords="offset points",
                color=SERIES_A, fontsize=7, fontweight="bold")
    ax.annotate("label-shuffled", (13, shuf[13]), xytext=(-2, -11), textcoords="offset points",
                color=SERIES_B, fontsize=7, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("action-expert layer")
    ax.set_ylabel("destination decoding accuracy")
    ax.set_title("The named destination reaches the\nexpert and is not acted on", loc="left")
    ax.set_xticks([0, 4, 8, 12, 15])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def make_all(out_dir: Path | None = None) -> int:
    _style()
    out = out_dir or (REPO_ROOT / "figures")
    out.mkdir(parents=True, exist_ok=True)
    ok = fig_sweep(out / "fig2_sweep.pdf")
    ok = fig_probe(out / "fig3_probe.pdf") and ok
    return 0 if ok else 1
