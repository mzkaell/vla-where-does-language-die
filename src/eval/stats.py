"""Statistics for paired designs (CLAUDE.md §8).

Every effect in this project is *paired*: the same visual state is evaluated under two
instructions, or the same state is evaluated clean vs patched. Pairing is the whole point
-- between-state variance in VLA action outputs dwarfs the instruction effect, so an
unpaired test would need orders of magnitude more samples to see anything.

Consequently the resampling here is over **pairs, not observations**. Resampling the two
arms independently would destroy the pairing and inflate the interval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    value: float
    lo: float
    hi: float
    n: int
    resamples: int
    alpha: float

    @property
    def excludes_zero(self) -> bool:
        return (self.lo > 0.0) or (self.hi < 0.0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"excludes_zero": self.excludes_zero}

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.lo:.4f}, {self.hi:.4f}] (n={self.n})"


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def paired_bootstrap(
    a: ArrayLike,
    b: ArrayLike,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int | np.random.Generator | None = 0,
) -> Estimate:
    """Bootstrap CI for the paired mean difference ``mean(a - b)``.

    Resamples pair indices with replacement, so `a[i]` and `b[i]` always move together.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired arrays must have equal shape, got {a_arr.shape} vs {b_arr.shape}")
    return bootstrap_mean(a_arr - b_arr, resamples=resamples, alpha=alpha, seed=seed)


def bootstrap_mean(
    x: ArrayLike,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int | np.random.Generator | None = 0,
    cluster: ArrayLike | None = None,
) -> Estimate:
    """Percentile bootstrap CI for the mean of `x` (already-paired differences).

    `cluster` gives each observation a group label and switches to a **cluster
    bootstrap**: whole groups are resampled with replacement rather than individual
    observations.

    This matters here and is easy to get wrong. Our trials are drawn from demonstration
    episodes, several states per episode, and states from one episode are strongly
    correlated -- same scene, same object placement, adjacent timesteps. Treating them as
    independent understates the variance, sometimes badly, so an interval computed without
    clustering is narrower than the data earns. Pass the demonstration id.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = arr.size
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not np.isfinite(arr).all():
        raise ValueError("sample contains NaN or inf; refusing to report a CI over it")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")

    rng = _rng(seed)

    if cluster is not None:
        groups = np.asarray(cluster).ravel()
        if groups.size != n:
            raise ValueError(f"cluster has {groups.size} labels for {n} observations")
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]
        means = np.empty(resamples, dtype=np.float64)
        for r in range(resamples):
            picked = rng.integers(0, len(members), size=len(members))
            means[r] = arr[np.concatenate([members[i] for i in picked])].mean()
        lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
        return Estimate(
            value=float(arr.mean()), lo=float(lo), hi=float(hi),
            n=n, resamples=resamples, alpha=alpha,
        )

    idx = rng.integers(0, n, size=(resamples, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Estimate(
        value=float(arr.mean()),
        lo=float(lo),
        hi=float(hi),
        n=n,
        resamples=resamples,
        alpha=alpha,
    )


def permutation_test(
    a: ArrayLike,
    b: ArrayLike,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | np.random.Generator | None = 0,
) -> float:
    """Two-sided p-value for a paired difference under label exchange.

    The null is that the two arms are exchangeable *within* each pair, so the test flips
    the sign of each pair's difference independently -- the paired analogue of shuffling
    labels. Returns an (r+1)/(n+1) corrected p-value, which is never 0.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired arrays must have equal shape")

    diff = a_arr - b_arr
    observed = abs(diff.mean())

    rng = _rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(resamples, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (resamples + 1))


def benjamini_hochberg(pvalues: ArrayLike, fdr: float = 0.05) -> NDArray[np.bool_]:
    """Benjamini-Hochberg step-up. Returns a boolean mask of rejected hypotheses.

    Used across the many candidate sites in the M2 sweep, where an uncorrected threshold
    would manufacture "significant" sites out of a few hundred independent tests.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    order = np.argsort(p)
    ranked = p[order]
    thresholds = fdr * np.arange(1, n + 1) / n
    passing = np.nonzero(ranked <= thresholds)[0]

    rejected = np.zeros(n, dtype=bool)
    if passing.size:
        # step-up: reject everything at or below the largest passing rank
        rejected[order[: passing[-1] + 1]] = True
    return rejected
