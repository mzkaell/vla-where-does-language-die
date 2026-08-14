"""M3 — binding transplant (CLAUDE.md §7–8): the encoding-vs-readout verdict.

    # after M2 names the site, e.g.:
    python scripts/run_transplant.py --checkpoint k1000dai/smolvla_libero_finetune \
        --extract-site vlm.L8.resid_post --alphas 0.25,0.5,1.0 --device mps

Same contrasts, states, and recovery readout as run_localization.py — the two
milestones must be directly comparable. What M3 adds over M2's full patch: the
injection is the working-minus-failing *delta*, at a controlled dose, optionally
restricted to a token-position slice (--positions a:b). alpha=1, same-site, all
positions reduces exactly to M2, which doubles as this script's sanity check.

Cross-tower injection (extract vlm.*, inject expert.*) needs a projection between
streams of different shapes and is deliberately NOT implemented yet; the shapes
refuse loudly rather than broadcasting into nonsense.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.reproduce_ifr import STATE_KEYS, build_images, build_state  # noqa: E402
from scripts.run_composition import collect_states, task_endpoints  # noqa: E402
from scripts.run_localization import CONTRASTS  # noqa: E402
from src.eval.composition import build_anchors, instruction_for  # noqa: E402
from src.interp.localization import (  # noqa: E402
    TrialBaseline,
    direction_cosine,
    net_translation,
)
from src.interp.transplant import binding_delta, judge, restrict_to_positions  # noqa: E402


def dosed_patch(deltas: list[torch.Tensor], alpha: float):
    """Patch callable for a site that fires once per delta (expert sites fire per
    denoising step). Refuses to recycle a delta across extra firings."""

    def _patch(old: torch.Tensor, index: int) -> torch.Tensor:
        if index >= len(deltas):
            raise IndexError(f"site fired {index + 1} times but only {len(deltas)} deltas cached")
        d = deltas[index]
        if d.shape != old.shape:
            raise ValueError(f"delta shape {tuple(d.shape)} != activation {tuple(old.shape)}")
        return old + alpha * d.to(dtype=old.dtype, device=old.device)

    return _patch
