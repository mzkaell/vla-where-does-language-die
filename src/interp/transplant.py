"""M3 — binding transplant: the encoding-vs-readout verdict (CLAUDE.md §7–8).

M2 asks *where* patching the whole activation redirects the failing run. M3 asks a
sharper question at the site M2 finds: is the object↔destination binding *absent* from
the stream feeding the action expert (encoding failure), or present but unread
(readout failure)?

The test: extract the binding as the working-minus-failing difference at the extraction
site, then inject it additively into the failing run — optionally at a *different* site
(extract from the VLM stream, inject at the expert input), optionally restricted to the
language-token positions, and at a controlled dose `alpha`.

At alpha=1, same-site, all-positions, this reduces exactly to M2's full patch — but only
for vlm.* sites, which fire once. Expert sites fire once per denoising step with state
feedback: the step-0 injection changes the trajectory, so at steps 1..9 the stream no
longer equals the failing run's and old + (w_k - f_k) != w_k. On expert sites the
alpha=1 number is a genuinely different quantity from M2's recovery, not a sanity check.

Verdict rule: CLAUDE.md §8 states "readout if recovery ≥50% of the gap" on the mean;
this module deliberately applies it to the bootstrap CI instead (whole interval above
0.5 → readout, whole interval below → not-readout, straddling → indeterminate). That is
stricter than §8 as written: a mean of 0.6 with CI [0.45, 0.75] is "readout" per the
prose and "indeterminate" here. The recovery itself is the one defined in
localization.py, so the two milestones stay directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from src.interp.localization import MIN_DISPLACEMENT, TrialBaseline, recovery_fraction

READOUT_THRESHOLD = 0.5


def binding_delta(act_working: Tensor, act_failing: Tensor) -> Tensor:
    """The transplanted quantity: what the working run has that the failing one lacks."""
    if act_working.shape != act_failing.shape:
        raise ValueError(
            f"cannot form a binding delta across shapes {tuple(act_working.shape)} and "
            f"{tuple(act_failing.shape)}; pad the pair to a common length first "
            f"(see pair_pad_length)"
        )
    return act_working - act_failing


def restrict_to_positions(delta: Tensor, positions: list[int]) -> Tensor:
    """Zero the delta everywhere except the given token positions (dim -2).

    Restricting to language-token positions is what separates "the binding moved the
    action" from "patching the whole prefix moved the action".
    """
    if not positions:
        raise ValueError("empty position list would zero the whole delta")
    out = delta.new_zeros(delta.shape)
    out[..., positions, :] = delta[..., positions, :]
    return out


def dosed_patch(deltas: list[Tensor], alpha: float = 1.0):
    """A patch callable for forward_with_cache: old + alpha * deltas[k] at firing k.

    Takes one delta per firing and refuses to recycle: silently re-injecting step-0's
    delta at steps 1..9 of an expert site is exactly the bug this signature prevents.
    """

    def _patch(old: Tensor, index: int) -> Tensor:
        if index >= len(deltas):
            raise IndexError(
                f"site fired {index + 1} times but only {len(deltas)} deltas cached"
            )
        d = deltas[index]
        if d.shape != old.shape:
            raise ValueError(
                f"delta shape {tuple(d.shape)} != activation shape {tuple(old.shape)}"
            )
        return old + alpha * d.to(dtype=old.dtype, device=old.device)

    return _patch


@dataclass
class TransplantVerdict:
    """The M3 outcome for one (extract site, inject site, alpha) configuration."""

    extract_site: str
    inject_site: str
    alpha: float
    n: int
    recovery_mean: float
    recovery_lo: float
    recovery_hi: float
    verdict: str  # "readout" | "not-readout" | "indeterminate"
    degenerate: bool = False  # injection collapsed motion; direction is noise
    n_dropped_nonfinite: int = 0  # trials whose recovery went NaN and left `n`


def judge(
    extract_site: str,
    inject_site: str,
    alpha: float,
    cos_patched: list[float],
    baselines: list[TrialBaseline],
    displacements: list[float],
    resamples: int = 10_000,
    seed: int = 0,
    min_trials: int = 5,
) -> TransplantVerdict:
    """Aggregate per-trial recoveries into a verdict with a bootstrap CI.

    "readout" needs the whole CI above the threshold, not just the mean — a verdict is
    a claim, and a claim whose interval straddles the rule is "indeterminate". Two
    guards mirror aggregate_site: the sensitivity trap (an injection that collapses
    commanded motion has no direction to score, so a degenerate median displacement
    forces "indeterminate"), and NaN recoveries are counted, not silently dropped —
    trials vanishing at high alpha is itself evidence the dose is off-distribution.
    """
    from src.eval.stats import bootstrap_mean

    triples = list(zip(cos_patched, baselines, displacements, strict=True))
    kept = [(c, b, d) for c, b, d in triples if b.usable]
    r = np.array([recovery_fraction(c, b) for c, b, _ in kept], dtype=np.float64)
    n_dropped = int(np.sum(~np.isfinite(r)))
    r = r[np.isfinite(r)]
    # degeneracy is judged on the SAME trials the CI is built from; mixing filtered
    # recoveries with unfiltered displacements lets normal motion on unusable trials
    # mask a collapsed-motion injection on the usable ones — a false-readout path
    disp = np.array([d for _, _, d in kept], dtype=np.float64)
    degenerate = bool(disp.size and np.median(disp) < MIN_DISPLACEMENT * 10)
    if degenerate or r.size < min_trials:
        return TransplantVerdict(
            extract_site, inject_site, alpha, int(r.size),
            float("nan"), float("nan"), float("nan"), "indeterminate",
            degenerate, n_dropped,
        )
    est = bootstrap_mean(r, resamples=resamples, seed=seed)
    if est.lo >= READOUT_THRESHOLD:
        verdict = "readout"
    elif est.hi < READOUT_THRESHOLD:
        verdict = "not-readout"
    else:
        verdict = "indeterminate"
    return TransplantVerdict(
        extract_site, inject_site, alpha, int(r.size), est.value, est.lo, est.hi, verdict,
        degenerate, n_dropped,
    )
