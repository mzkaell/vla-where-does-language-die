"""Instruction-following under contradiction (M0).

The behavioral effect the rest of the project localizes. Measured **offline** on the fixed
paired states in the stimulus set: no simulator, exactly reproducible, and cheap enough to
run over hundreds of pairs on CPU.

Two readouts, deliberately
--------------------------
CLAUDE.md's standing rule is that no headline claim rests on one technique. These two have
different failure modes:

`sensitivity`
    Paired action divergence ||a_A - a_B|| on identical pixels. Asks only "does the
    instruction change the action at all". Cannot be faked by a policy that ignores
    language -- such a policy scores exactly 0. But it is direction-blind: a policy that
    reacts to the words *incorrectly* still scores high.

`directional IFR`
    Of the two instructions, does the policy's action move *away* from the action the
    demonstration actually took, when commanded the non-source instruction? The state
    comes from a demo of one task, so that demo action is the visual-prior-consistent
    behaviour. Commanding the other instruction should push the action away from it.
    This one has direction, but depends on the demo action being a fair reference.

Agreement between them is the evidence. A high sensitivity with chance-level directional
IFR would mean the policy notices the words but does not ground them -- which is itself a
finding, and precisely the kind of thing a single metric would hide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.eval.stats import Estimate, bootstrap_mean, paired_bootstrap, permutation_test

CHANCE_LEVEL = 0.5


@dataclass
class PairOutcome:
    """Per-pair record. Everything needed to recompute the aggregates."""

    pair_id: str
    family: str
    divergence_ab: float
    """||a_A - a_B||: how much swapping the referent moved the action."""

    dist_a_to_demo: float
    dist_b_to_demo: float
    followed_instruction: bool
    """True if commanding the non-source instruction moved the action away from the demo."""

    source_is_a: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "family": self.family,
            "divergence_ab": self.divergence_ab,
            "dist_a_to_demo": self.dist_a_to_demo,
            "dist_b_to_demo": self.dist_b_to_demo,
            "followed_instruction": self.followed_instruction,
            "source_is_a": self.source_is_a,
        }


def _l2(a: Tensor, b: Tensor) -> float:
    return float(torch.linalg.vector_norm((a - b).reshape(-1).float()))


def score_pair(
    pair_id: str,
    family: str,
    action_a: Tensor,
    action_b: Tensor,
    demo_action: Tensor | None,
    source_is_a: bool,
) -> PairOutcome:
    """Score one pair from its two predicted action chunks.

    `demo_action` is the action the demonstration took at this state, broadcast over the
    chunk. It is the visual-prior-consistent reference. If unavailable the directional
    readout is undefined and reported as such rather than guessed.
    """
    divergence = _l2(action_a, action_b)

    if demo_action is None:
        return PairOutcome(
            pair_id=pair_id,
            family=family,
            divergence_ab=divergence,
            dist_a_to_demo=float("nan"),
            dist_b_to_demo=float("nan"),
            followed_instruction=False,
            source_is_a=source_is_a,
        )

    ref = demo_action.reshape(1, 1, -1).expand_as(action_a)
    d_a = _l2(action_a, ref)
    d_b = _l2(action_b, ref)

    # The source instruction is the one the demo was actually following. Commanding the
    # OTHER instruction should move the action further from the demonstrated action.
    if source_is_a:
        followed = d_b > d_a
    else:
        followed = d_a > d_b

    return PairOutcome(
        pair_id=pair_id,
        family=family,
        divergence_ab=divergence,
        dist_a_to_demo=d_a,
        dist_b_to_demo=d_b,
        followed_instruction=followed,
        source_is_a=source_is_a,
    )


@dataclass
class IFRResult:
    n: int
    sensitivity: Estimate
    directional_ifr: Estimate
    directional_p_vs_chance: float
    by_family: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "sensitivity": self.sensitivity.as_dict(),
            "directional_ifr": self.directional_ifr.as_dict(),
            "directional_p_vs_chance": self.directional_p_vs_chance,
            "chance_level": CHANCE_LEVEL,
            "by_family": self.by_family,
        }

    def summary(self) -> str:
        d = self.directional_ifr
        verdict = (
            "ABOVE chance"
            if d.lo > CHANCE_LEVEL
            else ("BELOW chance" if d.hi < CHANCE_LEVEL else "NOT distinguishable from chance")
        )
        return (
            f"n = {self.n} pairs\n"
            f"instruction sensitivity ||a_A - a_B||  : {self.sensitivity}\n"
            f"directional IFR                        : {self.directional_ifr}\n"
            f"  vs chance ({CHANCE_LEVEL}): {verdict}  (permutation p = "
            f"{self.directional_p_vs_chance:.4g})"
        )


def aggregate(
    outcomes: list[PairOutcome],
    resamples: int = 10_000,
    seed: int = 0,
) -> IFRResult:
    """Aggregate per-pair outcomes with paired bootstrap CIs."""
    if not outcomes:
        raise ValueError("no outcomes to aggregate")

    divergence = np.array([o.divergence_ab for o in outcomes], dtype=np.float64)
    followed = np.array([float(o.followed_instruction) for o in outcomes], dtype=np.float64)
    usable = np.array([np.isfinite(o.dist_a_to_demo) for o in outcomes], dtype=bool)

    sensitivity = bootstrap_mean(divergence, resamples=resamples, seed=seed)

    if usable.any():
        directional = bootstrap_mean(followed[usable], resamples=resamples, seed=seed)
        # Chance is 0.5; test the per-pair indicator against it.
        p = permutation_test(
            followed[usable],
            np.full(usable.sum(), CHANCE_LEVEL),
            resamples=resamples,
            seed=seed,
        )
    else:
        directional = Estimate(float("nan"), float("nan"), float("nan"), 0, resamples, 0.05)
        p = float("nan")

    by_family: dict[str, dict[str, Any]] = {}
    for fam in sorted({o.family for o in outcomes}):
        sel = [o for o in outcomes if o.family == fam]
        fam_div = np.array([o.divergence_ab for o in sel], dtype=np.float64)
        fam_fol = np.array(
            [float(o.followed_instruction) for o in sel if np.isfinite(o.dist_a_to_demo)],
            dtype=np.float64,
        )
        by_family[fam] = {
            "n": len(sel),
            "sensitivity": bootstrap_mean(fam_div, resamples=resamples, seed=seed).as_dict(),
            "directional_ifr": (
                bootstrap_mean(fam_fol, resamples=resamples, seed=seed).as_dict()
                if fam_fol.size
                else None
            ),
        }

    return IFRResult(
        n=len(outcomes),
        sensitivity=sensitivity,
        directional_ifr=directional,
        directional_p_vs_chance=p,
        by_family=by_family,
    )


def sensitivity_vs_control(
    paired_divergence: np.ndarray,
    control_divergence: np.ndarray,
    resamples: int = 10_000,
    seed: int = 0,
) -> Estimate:
    """Instruction effect above a same-instruction control.

    The control re-runs the SAME instruction twice. Under fixed noise that is exactly 0,
    so a non-zero control would reveal nondeterminism leaking into the measurement.
    """
    return paired_bootstrap(paired_divergence, control_divergence, resamples=resamples, seed=seed)
