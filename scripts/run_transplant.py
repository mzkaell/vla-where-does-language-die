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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--data-root", default="data/libero")
    ap.add_argument("--extract-site", required=True)
    ap.add_argument("--inject-site", default=None, help="defaults to the extract site")
    ap.add_argument("--alphas", default="0.25,0.5,1.0")
    ap.add_argument("--positions", default=None, help="token slice a:b to restrict the delta to")
    ap.add_argument("--n-trials", type=int, default=10, help="states per contrast")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-trials", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    inject = args.inject_site or args.extract_site
    alphas = [float(a) for a in args.alphas.split(",")]
    pos = None
    if args.positions:
        a, b = args.positions.split(":")
        pos = list(range(int(a), int(b)))

    run_id = args.run_id or f"m3_{args.extract_site.replace('.', '_')}"
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_dir = Path(args.data_root) / args.suite
    np.random.seed(args.seed)

    anchors = build_anchors(task_endpoints(suite_dir))

    from src.models.smolvla import SmolVLA, make_batch, pair_pad_length

    print(f"run_id  : {run_id}\ncheckpoint: {args.checkpoint}")
    print(f"extract : {args.extract_site}  inject: {inject}  alphas: {alphas}  pos: {args.positions}")
    model = SmolVLA.load(args.checkpoint, device=args.device)
    cfg = model.config
    keys = list(cfg.image_features)
    size = cfg.image_features[keys[0]].shape[-1]
    sdim = cfg.robot_state_feature.shape[0]

    (out_dir / "config.yaml").write_text(
        "\n".join(
            f"{k}: {json.dumps(v)}"
            for k, v in {
                "run_id": run_id, "experiment": "M3_transplant",
                "checkpoint": args.checkpoint, "extract_site": args.extract_site,
                "inject_site": inject, "alphas": alphas, "positions": args.positions,
                "n_trials_per_contrast": args.n_trials, "contrasts": CONTRASTS,
                "seed": args.seed, "device": args.device,
                "image_orientation": "rot180", "state_composition": list(STATE_KEYS),
                "platform": platform.platform(), "torch": torch.__version__,
            }.items()
        ),
        encoding="utf-8",
    )

    # TODO: trial loop lands in the next commit; running now writes config only.
    return 0


if __name__ == "__main__":
    sys.exit(main())
