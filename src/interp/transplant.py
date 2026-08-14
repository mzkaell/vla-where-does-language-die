"""M3 — binding transplant: encoding failure or readout failure?

The question
------------
M2 says *where* the object↔destination binding lives. It does not say why the novel pairing
fails. Two possibilities with opposite implications:

**Encoding failure** — the correct binding is never written into the stream that feeds the
action expert. Nothing downstream can read what was never there, so injecting it should not
help unless you inject the binding itself.

**Readout failure** — the binding *is* present in the backbone but the expert fails to use
it. Then supplying it at the expert's input should restore the behaviour.

CLAUDE.md §3 predicts readout-dominated failure at the VLM→expert interface. §8 sets the
threshold: **verdict is readout if the transplant recovers ≥50% of the working-minus-failing
gap.**

Why a direction rather than a whole activation
----------------------------------------------
M2 patches one run's entire activation into another. That is the right primitive for
localization but a blunt intervention: it carries everything about the working run, not just
the binding, and pushes the residual stream off-distribution.

The transplant instead estimates a **binding direction** — the difference of means between
working and failing runs at the target site, averaged over many trials, so trial-specific
content cancels and what survives is the component that systematically distinguishes a bound
pairing from an unbound one. That direction is then added to the failing run:

    patched = failing_activation + alpha * d

`alpha` is swept. A genuine mechanism should show graded recovery rather than an all-or-
nothing jump at one magnitude, and should not require alpha far outside the scale of the
activations themselves.

Controls, because a positive result here is the paper's headline
---------------------------------------------------------------
* **Random-direction control.** A norm-matched random direction must NOT recover. Without
  it, "adding a big vector changes the action" is indistinguishable from a binding transplant.
* **Held-out estimation.** The direction is estimated on one set of trials and applied to a
  different set, so recovery cannot be memorisation of the trials that defined it.
* **Off-target site control.** Injecting at a site M2 found irrelevant should not recover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

READOUT_THRESHOLD = 0.50  # CLAUDE.md §8


def binding_direction(working: list[Tensor], failing: list[Tensor]) -> Tensor:
    """Difference-of-means direction separating bound from unbound pairings.

    Averaging over trials cancels trial-specific content; what survives is the component
    that systematically differs between a demonstrated pairing and a novel one.
    """
    if not working or not failing:
        raise ValueError("need both working and failing activations to estimate a direction")
    w = torch.stack([t.float() for t in working]).mean(dim=0)
    f = torch.stack([t.float() for t in failing]).mean(dim=0)
    return w - f


def random_direction_like(d: Tensor, seed: int = 0) -> Tensor:
    """Norm-matched random direction — the control that makes a positive result meaningful."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(d.shape, generator=g, dtype=torch.float32).to(d.device)
    return r * (d.norm() / r.norm().clamp_min(1e-8))


def inject(activation: Tensor, direction: Tensor, alpha: float) -> Tensor:
    """activation + alpha * direction, preserving dtype (mixed precision in this model)."""
    return (activation.float() + alpha * direction.float()).to(activation.dtype)


@dataclass
class TransplantPoint:
    """Recovery at one injection strength."""

    alpha: float
    n: int
    recovery: float
    recovery_lo: float
    recovery_hi: float
    control_recovery: float
    """Norm-matched random direction at the same alpha. Should stay near zero."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "n": self.n,
            "recovery": {"value": self.recovery, "lo": self.recovery_lo, "hi": self.recovery_hi},
            "control_recovery": self.control_recovery,
        }


def verdict(points: list[TransplantPoint]) -> dict[str, Any]:
    """Encoding vs readout, per the CLAUDE.md §8 threshold.

    Reports the best alpha by *lower confidence bound*, not point estimate, so a noisy
    single alpha cannot win the sweep. Also reports the margin over the random-direction
    control, since recovery that the control matches is not evidence of a binding transplant.
    """
    scored = [p for p in points if np.isfinite(p.recovery)]
    if not scored:
        return {"verdict": "UNDETERMINED", "reason": "no finite recovery estimates"}

    best = max(scored, key=lambda p: p.recovery_lo)
    margin = best.recovery - best.control_recovery
    control_clean = best.control_recovery < 0.20

    if best.recovery_lo >= READOUT_THRESHOLD and control_clean:
        v = "READOUT"
        reason = (
            f"transplant recovers {best.recovery:.2f} (CI lower bound {best.recovery_lo:.2f}) "
            f"at alpha={best.alpha:g}, above the {READOUT_THRESHOLD:.2f} threshold, while a "
            f"norm-matched random direction recovers only {best.control_recovery:.2f}. The "
            "binding is present upstream and the expert fails to read it."
        )
    elif not control_clean:
        v = "UNDETERMINED"
        reason = (
            f"the random-direction control also recovers {best.control_recovery:.2f}, so the "
            "intervention is not specific to the binding direction. Any apparent recovery "
            "here is perturbation, not transplant."
        )
    else:
        v = "ENCODING"
        reason = (
            f"best transplant recovers only {best.recovery:.2f} (CI lower bound "
            f"{best.recovery_lo:.2f}), below the {READOUT_THRESHOLD:.2f} threshold. Supplying "
            "the binding at this site does not restore behaviour, consistent with the binding "
            "never being formed rather than being formed and unread."
        )

    return {
        "verdict": v,
        "reason": reason,
        "best_alpha": best.alpha,
        "best_recovery": best.recovery,
        "best_recovery_lo": best.recovery_lo,
        "control_recovery": best.control_recovery,
        "margin_over_control": margin,
        "threshold": READOUT_THRESHOLD,
    }
