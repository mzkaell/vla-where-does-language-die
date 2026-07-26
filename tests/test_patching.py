"""Patching correctness for SmolVLA.

This is the file that decides whether every downstream number is trustworthy. The
localization map in M2 is built entirely out of "patch site S, measure the action
change", so a hook that silently fails to fire, or fires but does not propagate,
produces a clean-looking heatmap of zeros that means nothing.

The suite is deliberately adversarial about that:

- `test_instrumented_forward_matches_stock` guards the re-implemented two-tower loop
  against upstream drift in lerobot.
- `test_patch_identity_*` are the positive controls: substituting a site's own cached
  value must be a no-op.
- `test_patch_negative_control_*` are the guards that stop the positive controls from
  passing vacuously. An identity test alone passes just fine when the patch never
  applied at all.
"""

from __future__ import annotations

import pytest
import torch

from src.models.base import action_divergence

pytestmark = pytest.mark.slow

# One site per tower. The expert tower is where the central hypothesis lives (the VLM ->
# action-expert interface), the VLM tower is the encoding side.
VLM_SITE = "vlm.L8.resid_post"
EXPERT_SITE = "expert.L8.resid_post"


# ------------------------------------------------------------------ determinism


def test_predict_action_is_deterministic(model, batch):
    """Flow matching integrates from sampled noise; fixed noise must pin the output.

    If this fails, no patching result is interpretable -- run-to-run noise would be
    confounded with the effect of the patch.
    """
    a = model.predict_action(batch)
    b = model.predict_action(batch)
    assert torch.equal(a, b), "same seed produced different actions"


def test_different_noise_changes_action(model, batch):
    """Negative control for the above: the seed must actually be doing something."""
    a = model.predict_action(batch, seed=0)
    b = model.predict_action(batch, seed=99)
    assert not torch.equal(a, b)


def test_action_shape(model, batch):
    action = model.predict_action(batch)
    assert action.ndim == 3
    assert action.shape[0] == 1
    assert action.shape[1] == model.config.chunk_size
    assert action.shape[2] == model.config.action_feature.shape[0]


# --------------------------------------------------- instrumented == stock forward


def test_instrumented_forward_matches_stock(model, batch):
    """The re-implemented two-tower loop must reproduce upstream exactly.

    `src/models/smolvla.py` re-implements `SmolVLMWithExpertModel.forward` because that
    method inlines the decoder layer and never calls it as a module, so forward hooks
    cannot see the residual stream. Re-implementation buys correct taps at the cost of a
    copy that can drift from lerobot. This test is what makes that trade safe: if lerobot
    changes the loop, this fails rather than silently returning wrong activations.
    """
    noise = model.make_noise(1)

    if hasattr(model.policy, "reset"):
        model.policy.reset()
    stock = model.policy.predict_action_chunk(dict(batch), noise=noise)

    instrumented = model.forward_with_cache(batch, sites=(), noise=noise).action

    assert torch.equal(stock, instrumented), (
        "instrumented forward diverged from stock lerobot forward; "
        "the re-implementation in src/models/smolvla.py is out of date"
    )


# ------------------------------------------------------------------ capture


def test_capture_residual_stream(model, batch):
    """Requirement (a): capture residual-stream activations at a named site."""
    res = model.forward_with_cache(batch, sites=[VLM_SITE, EXPERT_SITE])

    vlm = res.occurrences(VLM_SITE)
    expert = res.occurrences(EXPERT_SITE)

    # The VLM prefix runs once to fill the KV cache; the expert runs once per
    # denoising step. Getting this backwards is the most likely modelling error.
    assert len(vlm) == 1, f"expected 1 VLM prefix pass, got {len(vlm)}"
    assert len(expert) == model.config.num_steps, (
        f"expected {model.config.num_steps} expert passes (one per denoising step), "
        f"got {len(expert)}"
    )

    assert vlm[0].ndim == 3  # (batch, seq, hidden)
    assert torch.isfinite(vlm[0]).all()
    assert torch.isfinite(expert[0]).all()


def test_capture_does_not_change_output(model, batch):
    """Observing must not perturb. Capture is clone-on-read for exactly this reason."""
    noise = model.make_noise(1)
    clean = model.forward_with_cache(batch, sites=(), noise=noise).action
    observed = model.forward_with_cache(
        batch, sites=[VLM_SITE, EXPERT_SITE], noise=noise
    ).action
    assert torch.equal(clean, observed)


def test_all_sites_fire(model, batch):
    """Every advertised site must actually exist in the computation graph.

    A site that never fires would show zero causal effect in M2 and be misread as
    "this component does not matter".
    """
    names = model.site_names()
    res = model.forward_with_cache(batch, sites=names)
    never_fired = [n for n in names if not res.cache.get(n)]
    assert not never_fired, f"advertised sites that never fired: {never_fired}"


# ------------------------------------------------------- patching correctness


@pytest.mark.parametrize("site", [VLM_SITE, EXPERT_SITE])
def test_patch_identity(model, batch, site):
    """THE sanity check: patching a site with its own cached value changes nothing.

    Requirement (b) plus the correctness proof. This catches wrong shapes, off-by-one
    occurrence indexing, taps applied after the value is consumed, and dtype drift.
    """
    noise = model.make_noise(1)

    clean = model.forward_with_cache(batch, sites=[site], noise=noise)
    cached = clean.occurrences(site)

    repatched = model.patch(batch, patches={site: cached}, noise=noise)

    assert torch.equal(clean.action, repatched.action), (
        f"self-patch at {site} changed the action; patching is not correct"
    )


@pytest.mark.parametrize("site", [VLM_SITE, EXPERT_SITE])
def test_patch_negative_control(model, batch, site):
    """Guards the identity test against passing vacuously.

    If `patch` were a silent no-op, `test_patch_identity` would pass perfectly. This
    asserts that substituting a *different* value does move the action.
    """
    noise = model.make_noise(1)

    clean = model.forward_with_cache(batch, sites=[site], noise=noise)
    cached = clean.occurrences(site)
    perturbed = [t + 1.0 for t in cached]

    patched = model.patch(batch, patches={site: perturbed}, noise=noise)

    assert not torch.equal(clean.action, patched.action), (
        f"patching {site} with a different value had no effect; the patch is not applied"
    )
    div = action_divergence(clean.action, patched.action)
    assert torch.isfinite(div).all() and div.item() > 0


def test_patch_across_instructions_changes_action(model, batch, alt_batch):
    """The M2 primitive end-to-end: run on instruction A with B's activation patched in.

    Same scene, different instruction. Patching the VLM residual stream from the B run
    into the A run must move the action, otherwise there is no instruction signal at this
    site to localize.
    """
    noise = model.make_noise(1)

    run_a = model.forward_with_cache(batch, sites=[VLM_SITE], noise=noise)
    run_b = model.forward_with_cache(alt_batch, sites=[VLM_SITE], noise=noise)

    assert not torch.equal(run_a.action, run_b.action), (
        "the two instructions produced identical actions, so this pair carries no "
        "instruction signal and cannot support a localization claim"
    )

    patched = model.patch(batch, patches={VLM_SITE: run_b.occurrences(VLM_SITE)}, noise=noise)

    moved = action_divergence(run_a.action, patched.action).item()
    assert moved > 0, "patching B's activation into A did not change A's action"


def test_patch_single_occurrence_broadcasts(model, batch):
    """A single tensor patch must apply to every invocation of a multi-call site."""
    noise = model.make_noise(1)
    clean = model.forward_with_cache(batch, sites=[EXPERT_SITE], noise=noise)
    one = clean.occurrences(EXPERT_SITE)[0]

    # Broadcasting occurrence 0 to all 10 denoising steps is NOT the identity, since
    # steps 1..9 legitimately differ -- so this must change the action.
    patched = model.patch(batch, patches={EXPERT_SITE: one}, noise=noise)
    assert not torch.equal(clean.action, patched.action)


# ------------------------------------------------------------------ error paths


def test_unknown_site_raises(model, batch):
    with pytest.raises(KeyError, match="unknown site"):
        model.forward_with_cache(batch, sites=["vlm.L999.resid_post"])


def test_single_raises_on_multi_occurrence_site(model, batch):
    """`single()` must refuse to hide the multi-call nature of expert sites."""
    res = model.forward_with_cache(batch, sites=[EXPERT_SITE])
    with pytest.raises(ValueError, match="fired"):
        res.single(EXPERT_SITE)


def test_wrong_patch_shape_raises(model, batch):
    noise = model.make_noise(1)
    with pytest.raises(ValueError, match="shape"):
        model.patch(batch, patches={VLM_SITE: torch.zeros(1, 3, 5)}, noise=noise)
