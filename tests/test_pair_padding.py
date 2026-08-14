"""pair_pad_length: cross-run patch alignment on 'longest'-padded checkpoints.

The scratch_80k sweep died on a (1, 137, 960) vs (1, 136, 960) patch-shape mismatch
because that checkpoint pads language to 'longest' and the two instructions in a
contrastive pair tokenize to different lengths. These tests pin the fix's two load-
bearing properties without loading weights: 'max_length' checkpoints are untouched
(None), and a 'longest' pair comes back with one common length that make_batch
actually honours.
"""

from __future__ import annotations

import pytest
import torch

from src.models.smolvla import make_batch, pair_pad_length

PAIR = ["put the wine bottle on the rack", "put the bowl on the rack"]


class _Cfg:
    vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    tokenizer_max_length = 48
    pad_language_to = "longest"
    image_features: list = []  # make_batch only touches this for images


class _Policy:
    config = _Cfg()


@pytest.fixture(scope="module")
def policy():
    return _Policy()


def test_max_length_checkpoints_are_untouched(policy):
    policy.config.pad_language_to = "max_length"
    try:
        assert pair_pad_length(policy, PAIR) is None
    finally:
        policy.config.pad_language_to = "longest"


def test_longest_pair_gets_one_common_length(policy):
    n = pair_pad_length(policy, PAIR)
    assert isinstance(n, int)
    # the pair's instructions tokenize to different lengths; the common length
    # must cover the longer one and stay within the checkpoint's cap
    assert 0 < n <= policy.config.tokenizer_max_length


def test_make_batch_honours_pad_to_length(policy):
    n = pair_pad_length(policy, PAIR)
    state = torch.zeros(1, 8)
    shapes = set()
    for instr in PAIR:
        batch = make_batch({}, state, instr, policy, pad_to_length=n)
        shapes.add(batch["observation.language.tokens"].shape[1])
    assert shapes == {n}


def test_native_longest_still_mismatches(policy):
    """The guard that stops the tests above from passing vacuously: without the
    override, this pair really does produce unequal token lengths."""
    state = torch.zeros(1, 8)
    lengths = {
        make_batch({}, state, instr, policy)["observation.language.tokens"].shape[1]
        for instr in PAIR
    }
    assert len(lengths) == 2
