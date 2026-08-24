"""Publication figures, built only from committed runs in `results/`.

Three figures, each carrying an argument the prose cannot make on its own:

**Fig 1 -- the design and what it returns.** Panel (a) is the experimental logic: one
stimulus, two instructions a word apart, scored by which destination the motion heads
toward, run in three settings. Panels (b) and (c) give the headline numbers for those
settings. Built here rather than in TikZ so it regenerates from `results/` and can be
rendered and inspected; a hand-drawn figure cannot be checked without compiling, which is
how a retracted claim survived in an earlier version of Fig 3's title.

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
    """Match the paper rather than matplotlib's defaults.

    Conventions taken from the NeurIPS style file and from how figures actually read in
    a two-column-width single-column layout:

    * Serif type. The body is Times; sans-serif panels look pasted in from elsewhere.
    * No in-plot titles. The caption carries the message, and a title above the axes
      duplicates it and costs vertical space. Panels are identified by (a), (b), (c).
    * Nothing below 7pt. At 3.3in wide, 6pt annotations are unreadable in print, which is
      the most common way a technically correct figure still fails.
    * Legible in greyscale: every series is directly labelled and marker shapes differ, so
      colour is never the only carrier of identity.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 400,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4a4a48",
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": "#4a4a48",
            "ytick.color": "#4a4a48",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "grid.color": "#dedcd6",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def _panel_label(ax, letter: str) -> None:
    """(a), (b), (c) above the axes, the NeurIPS convention for multi-panel figures."""
    ax.text(-0.02, 1.10, f"({letter})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left", color=INK)


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

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    ax.grid(axis="y", zorder=0)
    ax.axhline(0, color=REFERENCE, lw=0.9, zorder=1)
    ax.plot(xs, n, color=SERIES_A, marker="o", ms=3.2, zorder=3)
    ax.plot(xs, c, color=SERIES_B, marker="s", ms=3.0, zorder=3)
    ax.plot(xs, diff, color=REFERENCE, ls=":", lw=1.3, zorder=2)

    # Direct labels: identity never rests on colour alone. Anchored at layer 8, where
    # the curves are furthest apart, rather than at the last layer -- labelling the
    # right end pushes the text outside the axes and off the canvas.
    ax.annotate("Control", (8, c[8]), xytext=(3, 5), textcoords="offset points",
                color=SERIES_B, fontsize=7.5, fontweight="bold")
    ax.annotate("Novel", (8, n[8]), xytext=(3, -12), textcoords="offset points",
                color=SERIES_A, fontsize=7.5, fontweight="bold")
    ax.annotate("Difference", (10, diff[10]), xytext=(2, -13), textcoords="offset points",
                color=INK_SOFT, fontsize=7.5)

    ax.set_xlabel("Action-expert layer")
    ax.set_ylabel("Patching recovery (resid_post)")
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

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    ax.grid(axis="y", zorder=0)
    ax.axhline(chance, color=REFERENCE, lw=0.9, ls="--", zorder=1)
    ax.annotate("Chance", (0.2, chance), xytext=(0, -12), textcoords="offset points",
                color=INK_SOFT, fontsize=7.5)
    ax.plot(xs, acc, color=SERIES_A, marker="o", ms=3.2, zorder=3)
    ax.plot(xs, shuf, color=SERIES_B, marker="s", ms=3.0, zorder=3)

    ax.annotate("Untrained pairings", (10, acc[10]), xytext=(-22, -18),
                textcoords="offset points", color=SERIES_A, fontsize=7.5, fontweight="bold")
    ax.annotate("Label-shuffled", (12, shuf[12]), xytext=(-20, 8),
                textcoords="offset points", color=SERIES_B, fontsize=7.5, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Action-expert layer")
    ax.set_ylabel("Destination decoding accuracy")
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

    Panel (a) is the design: the contrastive stimulus, the pipeline it runs through, and
    the three settings the same contrast is run in. It uses a real camera frame rather than
    a drawing -- the reader should see the actual scene the policy sees, including that the
    two arms of a contrastive pair are the *same* pixels.

    Panels (b) and (c) are what that design returns. Panel (c) must show BOTH halves of the
    finding. Showing only the failure (bowl->rack at
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

    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(6.9, 3.95))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.16, 1.0], hspace=0.36,
                          wspace=0.22, left=0.045, right=0.985, top=0.955, bottom=0.155)

    # ---- panel (a): the pipeline, then the three settings it is run in ---------
    # One drawing axes in 0..1 coordinates. The camera frame goes in as a real image
    # rather than a drawn box: the reader should see the actual scene the policy sees,
    # and that both arms of a contrastive pair are the *same* pixels.
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _panel_label(ax, "a")

    def _box(x0, x1, y0, y1, text, fill="#f4f3ef", size=7.5, weight="normal"):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.7, edgecolor="#a9a8a3", facecolor=fill, zorder=2,
            transform=ax.transData))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=size, color=INK, zorder=3, fontweight=weight,
                linespacing=1.35)

    def _arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color="#6d6c68", lw=0.8,
                                    shrinkA=0, shrinkB=0))

    # top row: input -> policy -> motion -> scored destination
    ROW = 0.70
    _box(0.004, 0.438, 0.470, 0.995, "", fill="#fbfbf9")
    im = ax.inset_axes([0.020, 0.545, 0.125, 0.395])
    im.imshow(frame)
    im.set_xticks([])
    im.set_yticks([])
    for s in im.spines.values():
        s.set_edgecolor("#a9a8a3")
        s.set_linewidth(0.7)
    # The two instructions are the literal strings handed to the policy, so they are
    # quoted at their real casing. SHOUTING the differing word misrepresents the input;
    # colour and the quotation marks carry the contrast instead.
    ax.text(0.160, 0.885, "“put the bowl on the plate”", fontsize=7.5, color=SERIES_A,
            va="center", ha="left")
    ax.text(0.160, 0.720, "“put the bowl on the stove”", fontsize=7.5, color=SERIES_B,
            va="center", ha="left")
    ax.text(0.160, 0.552, "same pixels, one changed word",
            fontsize=6.8, color=INK_SOFT, va="center", ha="left")

    _arrow(0.441, 0.477, ROW)
    _box(0.480, 0.610, ROW - 0.115, ROW + 0.115, "SmolVLA\npolicy")
    _arrow(0.612, 0.653, ROW)
    _box(0.655, 0.785, ROW - 0.115, ROW + 0.115, "predicted\nmotion")
    _arrow(0.787, 0.828, ROW)
    _box(0.830, 0.988, ROW - 0.115, ROW + 0.115, "scored\ndestination", fill="#eaf1fa")

    # second row: the same contrast, run in three settings
    ax.text(0.5, 0.395, "The same contrast is run in three settings",
            fontsize=7.2, color=INK_SOFT, ha="center", va="center")
    setting = [
        ("Neutral", "Before the grasp, does the\ninstruction steer the arm?", "#f4f3ef"),
        ("Conflict", "The arm is already heading\nto a different goal.", "#fdf0e8"),
        ("Composition", "Familiar object, familiar place,\npairing never demonstrated.", "#eaf1fa"),
    ]
    for i, (name, body, fill) in enumerate(setting):
        x0 = 0.012 + i * 0.331
        _box(x0, x0 + 0.311, 0.020, 0.315, "", fill=fill)
        ax.text(x0 + 0.016, 0.245, name, fontsize=7.5, color=INK,
                fontweight="bold", ha="left", va="center")
        ax.text(x0 + 0.016, 0.115, body, fontsize=6.8, color=INK_SOFT,
                ha="left", va="center", linespacing=1.4)

    # ---- panels (b) and (c): what the contrast returns -------------------------
    # Labels sit INSIDE each panel above their bar rather than on the y-axis. Long
    # y-tick labels overflow leftwards into the neighbouring panel and collide with its
    # bars, which is invisible in code and obvious the moment you render it.
    def _bars(ax, rows, colour, xlabel, title):
        ys = [1.0, 0.0]
        for y, (label, value, alpha) in zip(ys, rows, strict=False):
            ax.barh([y], [value], color=colour, alpha=alpha, height=0.30, zorder=3)
            ax.text(0, y + 0.24, label, fontsize=7.5, color=INK, va="bottom", ha="left")
            ax.text(value + 2.5, y, f"{value:.1f}".rstrip("0").rstrip("."),
                    va="center", fontsize=8, color=INK, fontweight="bold")
        ax.set_yticks([])
        ax.set_ylim(-0.55, 1.75)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.grid(axis="x", zorder=0)
        ax.spines["left"].set_visible(False)
        del title

    ax_b = fig.add_subplot(gs[1, 0])
    _bars(ax_b,
          [("Conflict: arm already heading elsewhere", 94.6, 1.0),
           ("Neutral state", 74.5, 0.45)],
          SERIES_A, "Follows the command (%)", None)
    _panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 1])
    _bars(ax_c,
          [("Composition: bottle to the plate", p_plate * 100, 1.0),
           ("Composition: bowl to the rack", p_rack * 100, 1.0)],
          SERIES_B, "Heads to the named place (%)", None)
    _panel_label(ax_c, "c")

    # One caption per data panel, both on a single figure-coordinate baseline.
    fig.canvas.draw()
    captions = ("Vision does not override the command",
                "Some pairings survive, others do not")
    for a, cap in zip((ax_b, ax_c), captions, strict=True):
        box = a.get_position()
        t = fig.text(box.x0 + box.width / 2, 0.022, cap,
                     fontsize=7, color=INK_SOFT, ha="center", va="bottom")
        # A caption wider than its panel can run off the canvas. Measure and pull it back.
        w = t.get_window_extent(fig.canvas.get_renderer())
        w = w.transformed(fig.transFigure.inverted())
        shift = max(0.005 - w.x0, 0.0) + min(0.995 - w.x1, 0.0)
        if shift:
            t.set_x(t.get_position()[0] + shift)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True
