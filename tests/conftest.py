"""Shared fixtures.

The real-weights fixtures are session-scoped: loading SmolVLA costs ~20s on CPU and
every test in the patching suite needs the same frozen model.
"""

from __future__ import annotations

import pytest
import torch

CHECKPOINT = "lerobot/smolvla_base"


# The two fixture instructions tokenize to different lengths (12 vs 13 with the
# SmolVLM2 tokenizer). Cross-run patching tests must pad the pair to a common
# length exactly like the runners do, or they pass only on 'max_length'-padded
# checkpoints and reproduce the production shape-mismatch on 'longest' ones.
PAIR_INSTRUCTIONS = [
    "pick up the black bowl and put it on the plate",
    "pick up the red mug and put it on the plate",
]

@pytest.fixture(scope="session")
def model():
    """Real SmolVLA-450M on CPU. Skips (does not fail) if weights are unavailable."""
    pytest.importorskip("lerobot")
    from src.models.smolvla import SmolVLA

    try:
        return SmolVLA.load(CHECKPOINT, device="cpu")
    except Exception as exc:  # network down, cache missing, etc.
        pytest.skip(f"could not load {CHECKPOINT}: {exc}")


@pytest.fixture(scope="session")
def batch(model):
    """A fixed, deterministic input batch. Content is arbitrary but stable."""
    from src.models.smolvla import make_batch, pair_pad_length

    cfg = model.config
    g = torch.Generator().manual_seed(1234)

    images = {
        key: torch.rand((1, *feat.shape), generator=g)
        for key, feat in cfg.image_features.items()
    }
    state = torch.rand((1, cfg.robot_state_feature.shape[0]), generator=g)

    return make_batch(
        images=images,
        state=state,
        instruction=PAIR_INSTRUCTIONS[0],
        policy=model.policy,
        pad_to_length=pair_pad_length(model.policy, PAIR_INSTRUCTIONS),
    )


@pytest.fixture(scope="session")
def alt_batch(model):
    """Same scene, different instruction -- the minimal contrastive manipulation."""
    from src.models.smolvla import make_batch, pair_pad_length

    cfg = model.config
    g = torch.Generator().manual_seed(1234)  # same seed => identical pixels and state

    images = {
        key: torch.rand((1, *feat.shape), generator=g)
        for key, feat in cfg.image_features.items()
    }
    state = torch.rand((1, cfg.robot_state_feature.shape[0]), generator=g)

    return make_batch(
        images=images,
        state=state,
        instruction=PAIR_INSTRUCTIONS[1],
        policy=model.policy,
        pad_to_length=pair_pad_length(model.policy, PAIR_INSTRUCTIONS),
    )
