"""M3 transplant math. Same spirit as test_patching: positive controls plus the
guards that stop them from passing vacuously."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.interp.localization import TrialBaseline
from src.interp.transplant import (
    dosed_patch,
    binding_delta,
    judge,
    restrict_to_positions,
)


def _base(cw=1.0, cf=0.0, i=0):
    return TrialBaseline(
        trial_id=f"t{i}", cos_working=cw, cos_failing=cf, headroom=cw - cf,
        ee_pos=np.zeros(3), anchor=np.ones(3),
    )


def test_delta_plus_failing_reconstructs_working():
    w, f = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    patched = dosed_patch([binding_delta(w, f)], alpha=1.0)(f, 0)
    assert torch.allclose(patched, w)  # alpha=1 same-site == M2 (vlm sites only)


def test_dosed_patch_refuses_to_recycle_deltas():
    d = [torch.zeros(1, 5, 8)]
    with pytest.raises(IndexError, match="only 1 deltas cached"):
        dosed_patch(d)(torch.zeros(1, 5, 8), 1)


def test_delta_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="pad the pair"):
        binding_delta(torch.zeros(1, 6, 8), torch.zeros(1, 5, 8))


def test_position_restriction_touches_only_named_positions():
    d = torch.ones(1, 5, 8)
    r = restrict_to_positions(d, [0, 3])
    assert r[0, [0, 3]].sum() == 16 and r[0, [1, 2, 4]].sum() == 0
    with pytest.raises(ValueError, match="empty position list"):
        restrict_to_positions(d, [])


DISP_OK = [1.0] * 8  # comfortably above the degeneracy floor


def test_full_recovery_reads_readout():
    baselines = [_base(i=i) for i in range(8)]
    v = judge("vlm.L8.resid_post", "expert.L0.resid_pre", 1.0, [1.0] * 8, baselines, DISP_OK)
    assert v.verdict == "readout" and v.n == 8


def test_degenerate_displacement_forces_indeterminate():
    baselines = [_base(i=i) for i in range(8)]
    v = judge("s", "s", 1.0, [1.0] * 8, baselines, [1e-5] * 8)
    assert v.verdict == "indeterminate" and v.degenerate


def test_nonfinite_recoveries_are_counted():
    baselines = [_base(i=i) for i in range(8)]
    v = judge("s", "s", 1.0, [1.0] * 6 + [float("nan")] * 2, baselines, DISP_OK)
    assert v.n == 6 and v.n_dropped_nonfinite == 2


def test_no_recovery_reads_not_readout():
    baselines = [_base(i=i) for i in range(8)]
    v = judge("s", "s", 1.0, [0.0] * 8, baselines, DISP_OK)
    assert v.verdict == "not-readout"


def test_straddling_ci_is_indeterminate():
    # recoveries alternating around the threshold -> CI straddles 0.5
    baselines = [_base(i=i) for i in range(8)]
    v = judge("s", "s", 1.0, [0.1, 0.9] * 4, baselines, DISP_OK)
    assert v.verdict == "indeterminate"


def test_too_few_trials_never_claims():
    baselines = [_base(i=i) for i in range(3)]
    v = judge("s", "s", 1.0, [1.0] * 3, baselines, [1.0] * 3)
    assert v.verdict == "indeterminate" and np.isnan(v.recovery_mean)


def test_unusable_trials_are_excluded():
    # headroom below the usability floor must not count toward n
    baselines = [_base(i=i) for i in range(5)] + [_base(cw=0.01, cf=0.0, i=9)]
    v = judge("s", "s", 1.0, [1.0] * 6, baselines, [1.0] * 6)
    assert v.n == 5
