"""M3 — binding transplant: the encoding-vs-readout verdict (CLAUDE.md §7–8).

M2 asks *where* patching the whole activation redirects the failing run. M3 asks a
sharper question at the site M2 finds: is the object↔destination binding *absent* from
the stream feeding the action expert (encoding failure), or present but unread
(readout failure)?

The test: extract the binding as the working-minus-failing difference at the extraction
site, then inject it additively into the failing run — optionally at a *different* site
(extract from the VLM stream, inject at the expert input), optionally restricted to the
language-token positions, and at a controlled dose `alpha`. At alpha=1, same-site,
all-positions, this reduces exactly to M2's full patch; every departure from that
corner is what makes it a transplant rather than a re-run.

Verdict rule (CLAUDE.md §8): **readout** if mean recovery ≥ 0.5 of the working-minus-
failing gap. The rule is stated on the recovery defined in localization.py, so the two
milestones share one readout and are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from src.interp.localization import TrialBaseline, recovery_fraction

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


def additive_patch(delta: Tensor, alpha: float = 1.0):
    """A patch callable for forward_with_cache: old + alpha * delta, at every firing."""

    def _patch(old: Tensor, index: int) -> Tensor:
        if delta.shape != old.shape:
            raise ValueError(
                f"delta shape {tuple(delta.shape)} != activation shape {tuple(old.shape)}"
            )
        return old + alpha * delta.to(dtype=old.dtype, device=old.device)

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


def judge(
    extract_site: str,
    inject_site: str,
    alpha: float,
    cos_patched: list[float],
    baselines: list[TrialBaseline],
    resamples: int = 10_000,
    seed: int = 0,
    min_trials: int = 5,
) -> TransplantVerdict:
    """Aggregate per-trial recoveries into a verdict with a bootstrap CI.

    "readout" needs the whole CI above the threshold, not just the mean — a verdict is
    a claim, and a claim whose interval straddles the rule is "indeterminate".
    """
    from src.eval.stats import bootstrap_mean

    r = np.array(
        [recovery_fraction(c, b) for c, b in zip(cos_patched, baselines, strict=True) if b.usable],
        dtype=np.float64,
    )
    r = r[np.isfinite(r)]
    if r.size < min_trials:
        return TransplantVerdict(
            extract_site, inject_site, alpha, int(r.size),
            float("nan"), float("nan"), float("nan"), "indeterminate",
        )
    est = bootstrap_mean(r, resamples=resamples, seed=seed)
    if est.lo >= READOUT_THRESHOLD:
        verdict = "readout"
    elif est.hi < READOUT_THRESHOLD:
        verdict = "not-readout"
    else:
        verdict = "indeterminate"
    return TransplantVerdict(
        extract_site, inject_site, alpha, int(r.size), est.value, est.lo, est.hi, verdict
    )
