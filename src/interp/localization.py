"""M2 — causal localization of the compositional binding failure.

The contrast our stimulus design hands us
-----------------------------------------
The behavioural result is that a *novel* object↔destination pairing fails while a *trained*
pairing using the same destination word succeeds. That gives a matched pair of runs:

    working:  "put the wine bottle on the rack"   (trained -> heads to the rack)
    failing:  "put the bowl on the rack"          (novel   -> heads to a memorised place)

Same destination word, same scene, same state. The only difference is the object the
destination has to bind to. So the question "where does the binding fail" becomes a clean
patching question: **at which site does injecting the working run's activation into the
failing run redirect the action toward the named destination?**

This is deliberately the same primitive M3 needs. M2 sweeps all sites to find where the
information lives; M3 asks whether injecting it at the VLM->expert interface *recovers*
behaviour, which separates encoding failure from readout failure.

Readout
-------
Binary accuracy is too coarse for a per-site sweep. We use the **cosine of the predicted net
displacement against the direction to the named destination**, which is continuous and
signed, and report the *recovery*:

    recovery(S) = cos(patched at S) - cos(failing, unpatched)

normalised by the headroom `cos(working) - cos(failing)` so 1.0 means "patching this site
fully restores the working run's behaviour" and 0.0 means "no effect".

Two hazards this module is built to avoid
-----------------------------------------
1. **Low-sensitivity sites masquerade as grounding sites.** Established in M0: where the
   instruction cannot move the action, its direction is estimated from noise and any
   direction-based score drifts toward chance. Every site therefore also reports the raw
   displacement magnitude, and sites whose patched displacement is negligible are flagged
   rather than scored.
2. **Many sites, so many chances to find something.** 130 sites means an uncorrected
   threshold manufactures hits. We compare against a position-shuffled null and apply
   Benjamini-Hochberg across sites (`src/eval/stats.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import Tensor

MIN_DISPLACEMENT = 1e-4


def direction_cosine(
    predicted_delta: np.ndarray, ee_pos: np.ndarray, anchor: np.ndarray
) -> float:
    """Cosine between the commanded net translation and the direction to `anchor`.

    Scale-free, so the controller's action scaling is irrelevant. Returns NaN when the
    commanded motion is too small to have a direction -- reporting NaN rather than 0 keeps
    "no motion" distinguishable from "motion orthogonal to the target".
    """
    norm = float(np.linalg.norm(predicted_delta))
    if norm < MIN_DISPLACEMENT:
        return float("nan")
    to_anchor = anchor - ee_pos
    d = float(np.linalg.norm(to_anchor))
    if d < 1e-9:
        return float("nan")
    return float(np.dot(predicted_delta / norm, to_anchor / d))


def net_translation(action: Tensor) -> np.ndarray:
    """Net (dx, dy, dz) commanded over an action chunk."""
    return action[0, :, :3].sum(dim=0).detach().cpu().numpy().astype(np.float64)


@dataclass
class SiteEffect:
    """Patching one site, aggregated over trials."""

    site: str
    tower: str
    component: str
    layer: int | None
    n: int
    recovery_mean: float
    recovery_lo: float
    recovery_hi: float
    cos_patched_mean: float
    displacement_mean: float
    degenerate: bool
    """True if patching collapsed the commanded motion -- the score is then meaningless."""

    p_vs_null: float = float("nan")
    significant_fdr: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "tower": self.tower,
            "component": self.component,
            "layer": self.layer,
            "n": self.n,
            "recovery": {
                "value": self.recovery_mean,
                "lo": self.recovery_lo,
                "hi": self.recovery_hi,
            },
            "cos_patched_mean": self.cos_patched_mean,
            "displacement_mean": self.displacement_mean,
            "degenerate": self.degenerate,
            "p_vs_null": self.p_vs_null,
            "significant_fdr": self.significant_fdr,
        }


@dataclass
class TrialBaseline:
    """The unpatched endpoints a trial's recovery is measured between."""

    trial_id: str
    cos_working: float
    cos_failing: float
    headroom: float
    ee_pos: np.ndarray
    anchor: np.ndarray

    @property
    def usable(self) -> bool:
        """Recovery is only defined when the working run actually beats the failing one.

        A trial where the two are already equivalent has no gap to close, so including it
        would add noise with no signal and shrink every site's apparent effect.
        """
        return (
            np.isfinite(self.cos_working)
            and np.isfinite(self.cos_failing)
            and self.headroom > 0.05
        )


def recovery_fraction(cos_patched: float, base: TrialBaseline) -> float:
    """How much of the working-vs-failing gap this patch closed. 1.0 = fully restored."""
    if not np.isfinite(cos_patched):
        return float("nan")
    return (cos_patched - base.cos_failing) / base.headroom


def aggregate_site(
    site: str,
    tower: str,
    component: str,
    layer: int | None,
    recoveries: list[float],
    cos_patched: list[float],
    displacements: list[float],
    resamples: int = 10_000,
    seed: int = 0,
    min_trials: int = 5,
) -> SiteEffect:
    from src.eval.stats import bootstrap_mean

    r = np.array([x for x in recoveries if np.isfinite(x)], dtype=np.float64)
    disp = np.array(displacements, dtype=np.float64)
    degenerate = bool(disp.size and np.median(disp) < MIN_DISPLACEMENT * 10)

    if r.size < min_trials:
        return SiteEffect(
            site, tower, component, layer, int(r.size),
            float("nan"), float("nan"), float("nan"),
            float(np.nanmean(cos_patched)) if cos_patched else float("nan"),
            float(np.median(disp)) if disp.size else float("nan"),
            degenerate,
        )

    est = bootstrap_mean(r, resamples=resamples, seed=seed)
    return SiteEffect(
        site=site,
        tower=tower,
        component=component,
        layer=layer,
        n=int(r.size),
        recovery_mean=est.value,
        recovery_lo=est.lo,
        recovery_hi=est.hi,
        cos_patched_mean=float(np.nanmean(cos_patched)),
        displacement_mean=float(np.median(disp)),
        degenerate=degenerate,
    )


def apply_fdr(effects: list[SiteEffect], null_recoveries: np.ndarray, fdr: float = 0.05):
    """Score each site against a position-shuffled null, then BH-correct across sites.

    The null is built by patching with activations whose *positions* have been shuffled:
    same distribution of values, no positional correspondence. A site that only appears
    causal because patching perturbs the run at all will score just as high under it.
    """
    from src.eval.stats import benjamini_hochberg

    if null_recoveries.size == 0:
        return effects

    live = [e for e in effects if np.isfinite(e.recovery_mean) and not e.degenerate]
    for e in live:
        # one-sided: how often does the null reach this recovery?
        e.p_vs_null = float(
            (np.sum(null_recoveries >= e.recovery_mean) + 1) / (null_recoveries.size + 1)
        )
    if live:
        rejected = benjamini_hochberg([e.p_vs_null for e in live], fdr=fdr)
        for e, rej in zip(live, rejected, strict=False):
            e.significant_fdr = bool(rej)
    return effects


def localization_summary(effects: list[SiteEffect]) -> dict[str, Any]:
    """Layer x component map plus the sharpness statistic from CLAUDE.md §8."""
    live = [e for e in effects if np.isfinite(e.recovery_mean) and not e.degenerate]
    if not live:
        # Every key the caller reads must exist even when nothing scored, so a run with too
        # few trials reports "0 sites scored" instead of dying in the summary printer.
        return {
            "n_sites": len(effects),
            "n_scored": 0,
            "n_degenerate": sum(1 for e in effects if e.degenerate),
            "n_significant_fdr": 0,
            "top_sites": [],
            "max_recovery": float("nan"),
            "effect_share_in_top_10pct": float("nan"),
            "localization_narrow": False,
            "by_tower": {},
            "by_component": {},
        }

    vals = np.array([max(e.recovery_mean, 0.0) for e in live])
    order = np.argsort(-vals)
    total = vals.sum()
    top_k = max(1, int(round(0.10 * len(live))))
    share_top10 = float(vals[order[:top_k]].sum() / total) if total > 0 else float("nan")

    sig = [e for e in live if e.significant_fdr]
    return {
        "n_sites": len(effects),
        "n_scored": len(live),
        "n_degenerate": sum(1 for e in effects if e.degenerate),
        "n_significant_fdr": len(sig),
        "top_sites": [e.site for e in sorted(live, key=lambda x: -x.recovery_mean)[:10]],
        "max_recovery": float(max(e.recovery_mean for e in live)),
        # "narrow" if >=70% of total effect sits in <=10% of sites (CLAUDE.md §8)
        "effect_share_in_top_10pct": share_top10,
        "localization_narrow": bool(share_top10 >= 0.70) if np.isfinite(share_top10) else False,
        "by_tower": {
            t: float(np.mean([e.recovery_mean for e in live if e.tower == t]))
            for t in sorted({e.tower for e in live})
        },
        "by_component": {
            c: float(np.mean([e.recovery_mean for e in live if e.component == c]))
            for c in sorted({e.component for e in live})
        },
    }
