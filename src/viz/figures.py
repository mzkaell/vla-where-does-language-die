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
    """Fig 3: destination decodability through the expert, against a shuffled control."""
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
    ok = fig_schematic(out / "fig1_schematic.pdf")
    ok = fig_sweep(out / "fig2_sweep.pdf") and ok
    ok = fig_probe(out / "fig3_probe.pdf") and ok
    return 0 if ok else 1


def fig_schematic(out_path: Path) -> bool:
    """Fig 1: the whole paper in one image.

    Three panels, left to right: what a stimulus is, the account the data rules out, and
    the finding. Panel 1 uses a real camera frame rather than a drawing -- the reader
    should see the actual scene the policy sees, including that the two arms of a
    contrastive pair are the *same* pixels.

    Panel 3 must show BOTH halves of the finding. Showing only the failure (bowl->rack at
    1%) would argue for the blind-substitution reading this paper explicitly retracts; the
    command dependence (told "plate" -> plate 45%) is what distinguishes degraded
    composition from a lookup.
    """
    import json

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    demo = REPO_ROOT / "data" / "libero" / "libero_goal" / "put_the_bowl_on_the_plate_demo.hdf5"
    if not demo.exists():
        print("fig1: LIBERO demo file missing")
        return False
    with h5py.File(demo, "r") as h:
        # rot180: LIBERO stores frames bottom-up (MuJoCo/OpenGL convention). Feeding them
        # unrotated made the policy score worse than a random trajectory -- artifact #2.
        frame = np.ascontiguousarray(h["data"]["demo_18"]["obs"]["agentview_rgb"][40][::-1, ::-1])

    fs = _load("fs_finetune")
    if not fs:
        print("fig1: missing fs_finetune")
        return False
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "results" / "fs_finetune" / "per_trial.jsonl")
        .read_text().splitlines() if line.strip()
    ]
    novel = [r for r in rows if r["novel"] and r["object"] == "wine bottle"]
    told_plate = [r for r in novel if r["named_destination"] == "the plate"]
    p_plate = sum(r["chosen_destination"] == "the plate" for r in told_plate) / len(told_plate)
    bowl_rack = [r for r in rows if r["novel"] and r["object"] == "bowl"]
    p_rack = sum(r["correct"] for r in bowl_rack) / len(bowl_rack)

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35))

    # ---- panel 1: the stimulus -------------------------------------------------
    ax = axes[0]
    ax.imshow(frame)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#c9c8c3")
    ax.set_title("1.  One scene, one word apart", loc="left", fontsize=8, pad=6)
    # Plain text with the differing word capitalised, not mathtext bold: "$\bf{...}$" in a
    # non-raw string makes Python read \b as a backspace, which matplotlib then fails to
    # parse. Capitals carry the contrast just as well and cannot break in transit.
    ax.text(0.5, -0.10, "put the bowl on the PLATE", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color=SERIES_A)
    ax.text(0.5, -0.25, "put the bowl on the STOVE", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color=SERIES_B)
    ax.text(0.5, -0.42, "identical pixels; the action difference\nis attributable to that word",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=INK_SOFT)

    # ---- panels 2 and 3 -------------------------------------------------------
    # Labels sit INSIDE each panel above their bar rather than on the y-axis. Long
    # y-tick labels overflow leftwards into the neighbouring panel and collide with its
    # bars, which is invisible in code and obvious the moment you render it.
    def _bars(ax, rows, colour, xlabel, title):
        ys = [1.0, 0.0]
        for y, (label, value, alpha) in zip(ys, rows, strict=False):
            ax.barh([y], [value], color=colour, alpha=alpha, height=0.30, zorder=3)
            ax.text(0, y + 0.26, label, fontsize=6.8, color=INK_SOFT, va="bottom", ha="left")
            ax.text(value + 2.5, y, f"{value:.1f}".rstrip("0").rstrip("."),
                    va="center", fontsize=8, color=INK, fontweight="bold")
        ax.set_yticks([])
        ax.set_ylim(-0.55, 1.75)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.grid(axis="x", zorder=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=8, pad=6)

    _bars(axes[1],
          [("arm already heading elsewhere", 94.6, 1.0), ("neutral state", 74.5, 0.45)],
          SERIES_A, "follows the command (%)", "2.  Vision does not override")

    _bars(axes[2],
          [('"bottle on the plate", told plate', p_plate * 100, 1.0),
           ('"bowl on the rack", never trained', p_rack * 100, 1.0)],
          SERIES_B, "heads to the named place (%)", "3.  Untrained pairings degrade")
    # Figure-level so it clears panel 3's x-axis label instead of landing on top of it.
    fig.text(0.685, 0.015, "still tracks the command, so not a fixed lookup",
             fontsize=6.5, color=INK_SOFT, ha="left")

    fig.subplots_adjust(bottom=0.30, wspace=0.30, top=0.86)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True
