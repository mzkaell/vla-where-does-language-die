"""Layer-wise probing: is the named destination *encoded* at all?

Why probing rather than more patching
-------------------------------------
The causal sweep failed for a structural reason, not a tuning one. Patching an early site
propagates through everything downstream while a late one changes almost nothing, so
recovery tracks distance-from-output. The control confirmed it: an identical sweep on
pairings the model handles correctly reproduced the same profile, and novel-minus-control
came out at −0.06.

Probing has a different failure mode, which is exactly why it is worth running. It asks a
*correlational* question — can a linear readout recover the named destination from the
residual stream at layer L? — and that question is unaffected by how much computation sits
downstream. The two methods disagreeing would be informative; the two agreeing is the
"two techniques with different failure modes" the project's standing rule demands.

What the answer means
---------------------
Train a probe on **trained** pairings (where behaviour is correct) and test on **novel**
pairings (where it is not):

* probe decodes the destination on novel pairings ⇒ the information is present in the
  stream and the action does not follow it ⇒ **readout failure**
* probe cannot decode it ⇒ the instruction never got bound into the representation ⇒
  **encoding failure**

That is the RQ2 verdict the transplant was supposed to deliver, obtained without the
object-identity confound that invalidated it.

Controls
--------
* **Label-shuffled baseline.** A probe trained on shuffled labels must sit at chance. With
  a 4-way task and a few hundred high-dimensional examples, an unregularised probe can fit
  noise; this is what catches it.
* **Train on trained pairings only, test on novel.** A probe trained on both would learn
  the novel condition directly and could not distinguish "encoded" from "learnable by the
  probe".
* **Report chance explicitly** (1/n_destinations), never implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ProbeResult:
    """Decoding accuracy at one site."""

    site: str
    layer: int | None
    tower: str
    n_train: int
    n_test_trained: int
    n_test_novel: int
    acc_trained: float
    """Held-out accuracy on TRAINED pairings -- an upper bound and a sanity check."""

    acc_novel: float
    """Accuracy on NOVEL pairings. The number the verdict turns on."""

    acc_shuffled: float
    """Label-shuffled control. Must sit near chance or the probe is fitting noise."""

    chance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "layer": self.layer,
            "tower": self.tower,
            "n_train": self.n_train,
            "n_test_trained": self.n_test_trained,
            "n_test_novel": self.n_test_novel,
            "acc_trained": self.acc_trained,
            "acc_novel": self.acc_novel,
            "acc_shuffled": self.acc_shuffled,
            "chance": self.chance,
        }


def pool_activation(act: np.ndarray) -> np.ndarray:
    """(seq, hidden) -> (hidden,) by mean over positions.

    Deliberately position-agnostic: this asks whether the destination is decodable
    *anywhere* in the representation, which is the weakest claim that still answers the
    encoding question. Position-resolved probing is a refinement, not the headline.
    """
    a = np.asarray(act, dtype=np.float64)
    return a.mean(axis=0) if a.ndim == 2 else a.reshape(-1)


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 0,
    max_iter: int = 2000,
) -> float:
    """Multinomial logistic regression accuracy. Regularised, standardised."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_train)) < 2 or x_test.shape[0] == 0:
        return float("nan")
    scaler = StandardScaler().fit(x_train)
    # No multi_class kwarg: removed in scikit-learn 1.7+, and multinomial is the default
    # for the lbfgs solver anyway.
    clf = LogisticRegression(max_iter=max_iter, C=0.1, random_state=seed)
    clf.fit(scaler.transform(x_train), y_train)
    return float(clf.score(scaler.transform(x_test), y_test))


def verdict(results: list[ProbeResult], margin: float = 0.10) -> dict[str, Any]:
    """Encoding vs readout, from the best-decoding site.

    Requires the shuffled control to be near chance at that site; a probe that fits noise
    can report high accuracy on anything, which would invert the verdict.
    """
    live = [r for r in results if np.isfinite(r.acc_novel)]
    if not live:
        return {"verdict": "UNDETERMINED", "reason": "no site produced a finite probe accuracy"}

    best = max(live, key=lambda r: r.acc_novel)
    chance = best.chance
    control_ok = abs(best.acc_shuffled - chance) < margin

    if not control_ok:
        return {
            "verdict": "UNDETERMINED",
            "reason": (
                f"at {best.site} the label-shuffled probe reaches {best.acc_shuffled:.2f} "
                f"against chance {chance:.2f}; the probe is fitting noise, so its accuracy "
                "on real labels cannot be interpreted. Collect more states, or regularise "
                "harder, before reading anything off this run."
            ),
            "best_site": best.site,
            "acc_shuffled": best.acc_shuffled,
            "chance": chance,
        }

    # Compare against the SHUFFLED baseline, not just chance. The shuffled probe is the
    # empirical null for this feature dimensionality and sample size, and with
    # high-dimensional activations it can sit well above the analytic chance level.
    # Testing only against chance would call noise a readout failure.
    if best.acc_novel >= max(chance, best.acc_shuffled) + margin:
        v, reason = "READOUT", (
            f"the named destination is decodable from {best.site} on NOVEL pairings at "
            f"{best.acc_novel:.2f} (chance {chance:.2f}, shuffled control "
            f"{best.acc_shuffled:.2f}). The instruction is represented; the action does not "
            "follow it. This is a readout failure."
        )
    else:
        v, reason = "ENCODING", (
            f"the best site ({best.site}) decodes novel pairings at {best.acc_novel:.2f} "
            f"against chance {chance:.2f}, while trained pairings reach "
            f"{best.acc_trained:.2f}. The destination is representable in principle but is "
            "not encoded for novel pairings. This is an encoding failure."
        )

    return {
        "verdict": v,
        "reason": reason,
        "best_site": best.site,
        "acc_novel": best.acc_novel,
        "acc_trained": best.acc_trained,
        "acc_shuffled": best.acc_shuffled,
        "chance": chance,
        "margin": margin,
    }
