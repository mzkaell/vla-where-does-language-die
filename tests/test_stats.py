"""Metric math tests. Silent bugs here corrupt every reported number (CLAUDE.md §11).

These check statistical *properties* against known-answer cases, not just that the code
runs. A bootstrap that returns plausible-looking intervals while ignoring the pairing is
the exact failure this file exists to catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.stats import (
    benjamini_hochberg,
    bootstrap_mean,
    paired_bootstrap,
    permutation_test,
)


class TestBootstrap:
    def test_recovers_known_mean(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 1.0, size=2000)
        est = bootstrap_mean(x, resamples=2000, seed=1)
        assert est.value == pytest.approx(x.mean())
        assert est.lo < 5.0 < est.hi

    def test_ci_brackets_point_estimate(self):
        est = bootstrap_mean(np.arange(100.0), resamples=1000, seed=1)
        assert est.lo <= est.value <= est.hi

    def test_ci_narrows_with_more_data(self):
        rng = np.random.default_rng(0)
        small = bootstrap_mean(rng.normal(0, 1, 50), resamples=2000, seed=1)
        large = bootstrap_mean(rng.normal(0, 1, 5000), resamples=2000, seed=1)
        assert (large.hi - large.lo) < (small.hi - small.lo)

    def test_deterministic_under_seed(self):
        x = np.random.default_rng(3).normal(size=200)
        a = bootstrap_mean(x, resamples=500, seed=42)
        b = bootstrap_mean(x, resamples=500, seed=42)
        assert (a.value, a.lo, a.hi) == (b.value, b.lo, b.hi)

    def test_nominal_coverage(self):
        """A 95% interval must cover the truth about 95% of the time.

        The strongest available check that the interval means what it claims.
        """
        rng = np.random.default_rng(7)
        covered = 0
        trials = 200
        for i in range(trials):
            sample = rng.normal(1.0, 1.0, size=120)
            est = bootstrap_mean(sample, resamples=600, seed=i)
            covered += est.lo <= 1.0 <= est.hi
        assert 0.88 <= covered / trials <= 0.99, f"coverage {covered / trials:.3f} off nominal"

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            bootstrap_mean([1.0, np.nan, 3.0])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_mean([])


class TestPairedBootstrap:
    def test_zero_difference_ci_contains_zero(self):
        x = np.random.default_rng(0).normal(size=300)
        est = paired_bootstrap(x, x, resamples=1000, seed=1)
        assert est.value == 0.0
        assert not est.excludes_zero

    def test_detects_constant_offset(self):
        x = np.random.default_rng(0).normal(size=300)
        est = paired_bootstrap(x + 2.0, x, resamples=2000, seed=1)
        assert est.value == pytest.approx(2.0)
        assert est.excludes_zero

    def test_pairing_beats_unpaired_precision(self):
        """The reason we pair at all.

        With large between-item variance and a small constant effect, the paired interval
        must be dramatically tighter. If someone "fixes" the resampler to draw the two
        arms independently, this test fails.
        """
        rng = np.random.default_rng(5)
        base = rng.normal(0.0, 50.0, size=400)  # huge between-state variance
        a = base + 1.0  # small consistent effect
        b = base

        paired = paired_bootstrap(a, b, resamples=2000, seed=1)
        unpaired_width = 2 * 1.96 * np.sqrt(a.var() / a.size + b.var() / b.size)

        assert paired.excludes_zero
        assert (paired.hi - paired.lo) < unpaired_width / 10

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            paired_bootstrap([1.0, 2.0], [1.0])


class TestPermutation:
    def test_null_is_not_significant(self):
        x = np.random.default_rng(0).normal(size=200)
        assert permutation_test(x, x.copy(), resamples=1000, seed=1) > 0.05

    def test_strong_effect_is_significant(self):
        x = np.random.default_rng(0).normal(size=200)
        assert permutation_test(x + 3.0, x, resamples=1000, seed=1) < 0.01

    def test_pvalue_never_zero(self):
        """(r+1)/(n+1) correction: p=0 would be a lie about resolution."""
        x = np.random.default_rng(0).normal(size=100)
        p = permutation_test(x + 500.0, x, resamples=100, seed=1)
        assert p > 0.0

    def test_pvalue_in_unit_interval(self):
        rng = np.random.default_rng(2)
        for _ in range(10):
            p = permutation_test(rng.normal(size=50), rng.normal(size=50), resamples=200, seed=3)
            assert 0.0 < p <= 1.0


class TestBenjaminiHochberg:
    def test_all_null_rejects_almost_nothing(self):
        p = np.random.default_rng(0).uniform(size=500)
        assert benjamini_hochberg(p, fdr=0.05).sum() <= 5

    def test_strong_signals_rejected(self):
        p = np.concatenate([np.full(10, 1e-8), np.random.default_rng(0).uniform(size=200)])
        rejected = benjamini_hochberg(p, fdr=0.05)
        assert rejected[:10].all()

    def test_is_step_up_not_naive_threshold(self):
        """BH rejects everything below the largest passing rank.

        A naive `p <= fdr*rank/n` elementwise mask would miss the middle p-value here.
        This is the classic step-up bug.
        """
        p = np.array([0.001, 0.04, 0.005])  # sorted: .001, .005, .04
        # thresholds at n=3, fdr=.05: .0167, .0333, .05 -> .04 <= .05 passes at rank 3,
        # so all three must be rejected even though .005 > .0167 fails elementwise...
        rejected = benjamini_hochberg(p, fdr=0.05)
        assert rejected.all(), "step-up property violated"

    def test_more_conservative_than_uncorrected(self):
        p = np.random.default_rng(1).uniform(size=300)
        assert benjamini_hochberg(p, fdr=0.05).sum() <= (p < 0.05).sum()

    def test_empty_input(self):
        assert benjamini_hochberg([]).shape == (0,)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            benjamini_hochberg([0.5, 1.5])
