#!/usr/bin/env python
"""M0 -- instruction-following under contradiction on SmolVLA + LIBERO-Goal.

    python scripts/reproduce_ifr.py --model smolvla --suite libero_goal

Runs entirely offline on the fixed paired states in the stimulus set. No simulator.

IMPORTANT -- checkpoint choice
------------------------------
`lerobot/smolvla_base` is a PRETRAINED BASE checkpoint, not a LIBERO policy. Its config
declares a 6-dim state and 6-dim action, while LIBERO is 8-dim state / 7-dim action, and
it has never seen these tasks. Measuring "does the policy follow the instruction" on a
policy that cannot do the task at all measures nothing.

So M0 needs a LIBERO-finetuned SmolVLA. Several exist publicly but none is official; pass
one explicitly with --checkpoint. The chosen checkpoint is recorded in the run config, and
its provenance is a caveat that belongs in the paper.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_pairs import load_observation, load_pairs  # noqa: E402
from src.eval.ifr import aggregate, score_pair  # noqa: E402

# LIBERO -> LeRobot observation mapping. LIBERO stores a 7-dim joint vector, a 2-dim
# gripper vector and the end-effector pose; LIBERO-finetuned VLA checkpoints conventionally
# take an 8-dim state of [ee_pos(3), ee_ori(3), gripper(2)].
STATE_KEYS = ("ee_pos", "ee_ori", "gripper_states")


def build_state(obs: dict[str, np.ndarray], dim: int) -> torch.Tensor:
    vec = np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in STATE_KEYS])
    if vec.size < dim:
        vec = np.pad(vec, (0, dim - vec.size))
    return torch.from_numpy(vec[:dim]).unsqueeze(0)


def build_images(obs: dict[str, np.ndarray], image_keys: list[str], size: int) -> dict:
    """Map LIBERO's two cameras onto whatever image features the checkpoint declares."""
    import torch.nn.functional as F

    sources = {
        "wrist": obs["eye_in_hand_rgb"],
        "agent": obs["agentview_rgb"],
    }

    out = {}
    for key in image_keys:
        raw = sources["wrist"] if "wrist" in key.lower() else sources["agent"]
        t = torch.from_numpy(np.asarray(raw, dtype=np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)  # HWC -> 1CHW
        if t.shape[-1] != size:
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        out[key] = t
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="LIBERO-finetuned SmolVLA repo id. REQUIRED: smolvla_base cannot do LIBERO.",
    )
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N pairs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--allow-base-checkpoint",
        action="store_true",
        help="run against smolvla_base anyway (produces a meaningless IFR; for plumbing only)",
    )
    args = ap.parse_args()

    if args.model != "smolvla":
        print(f"only smolvla is implemented (got {args.model!r})", file=sys.stderr)
        return 2

    if args.checkpoint is None and not args.allow_base_checkpoint:
        print(
            "ERROR: --checkpoint is required.\n\n"
            "lerobot/smolvla_base is a pretrained base model, not a LIBERO policy: it\n"
            "declares a 6-dim state and 6-dim action against LIBERO's 8/7, and has never\n"
            "seen these tasks. An IFR measured on it would not be the behavioral effect\n"
            "this project localizes -- it would be noise from an incompetent policy.\n\n"
            "Pass a LIBERO-finetuned SmolVLA, e.g.\n"
            "  --checkpoint k1000dai/smolvla_libero_finetune\n"
            "  --checkpoint bicmol/smolvla-libero\n\n"
            "Or --allow-base-checkpoint to exercise the plumbing only.",
            file=sys.stderr,
        )
        return 2

    checkpoint = args.checkpoint or "lerobot/smolvla_base"
    pairs_path = args.pairs or REPO_ROOT / "stimuli" / f"{args.suite}_pairs_v1.jsonl"
    run_id = args.run_id or f"ifr_{args.suite}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPO_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"run_id     : {run_id}")
    print(f"checkpoint : {checkpoint}")
    print(f"pairs      : {pairs_path}")

    pairs = load_pairs(pairs_path)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"n pairs    : {len(pairs)}")

    from src.models.smolvla import SmolVLA, make_batch

    print("loading model ...")
    model = SmolVLA.load(checkpoint, device=args.device)
    cfg = model.config
    image_keys = list(cfg.image_features)
    state_dim = cfg.robot_state_feature.shape[0]
    img_size = cfg.image_features[image_keys[0]].shape[-1]
    print(f"image keys : {image_keys}")
    print(f"state dim  : {state_dim} | action dim: {cfg.action_feature.shape[0]}")
    print(f"norm stats : {'recovered from checkpoint' if model.has_norm_stats else 'NONE'}")
    if not model.has_norm_stats:
        print(
            "       WARNING: no normalization buffers found. Inputs stay raw and outputs\n"
            "       stay in the policy's own units, so the directional readout compares\n"
            "       different scales and is NOT interpretable. Sensitivity still is.",
            file=sys.stderr,
        )

    # Dump the resolved config BEFORE the run (CLAUDE.md §11) so a crashed run is still
    # attributable.
    resolved = {
        "run_id": run_id,
        "milestone": "M0",
        "args": {k: str(v) for k, v in vars(args).items()},
        "checkpoint": checkpoint,
        "checkpoint_is_base_model": checkpoint == "lerobot/smolvla_base",
        "pairs_file": str(pairs_path),
        "n_pairs": len(pairs),
        "seed": args.seed,
        "resamples": args.resamples,
        "device": args.device,
        "image_keys": image_keys,
        "state_dim": int(state_dim),
        "state_composition": list(STATE_KEYS),
        "normalization_recovered": model.has_norm_stats,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    (out_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {json.dumps(v)}" for k, v in resolved.items()), encoding="utf-8"
    )

    outcomes = []
    t0 = time.time()
    for i, pair in enumerate(pairs):
        obs = load_observation(pair, args.data_root)
        images = build_images(obs, image_keys, img_size)
        state = build_state(obs, state_dim)

        state = model.normalize_state(state)
        batch_a = make_batch(images, state, pair["instruction_a"], model.policy, args.device)
        batch_b = make_batch(images, state, pair["instruction_b"], model.policy, args.device)

        # Same noise on both arms: the ONLY difference between them is the instruction.
        noise = model.make_noise(1)
        # Unnormalize so predictions and the demonstration action share raw LIBERO units;
        # the directional readout compares distances between them.
        action_a = model.unnormalize_action(model.predict_action(batch_a, noise=noise))
        action_b = model.unnormalize_action(model.predict_action(batch_b, noise=noise))

        demo_action = torch.from_numpy(
            np.asarray(obs["action"][: cfg.action_feature.shape[0]], dtype=np.float32)
        )

        source_is_a = pair["source_task"] in pair["instruction_a"].replace(" ", "_")
        outcomes.append(
            score_pair(
                pair_id=pair["pair_id"],
                family=pair["family"],
                action_a=action_a,
                action_b=action_b,
                demo_action=demo_action,
                source_is_a=source_is_a,
            )
        )

        if (i + 1) % 10 == 0 or i == len(pairs) - 1:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(pairs) - i - 1) / max(rate, 1e-9)
            print(
                f"  {i + 1}/{len(pairs)}  ({rate:.2f} pair/s, eta {eta / 60:.1f} min)",
                flush=True,
            )

    result = aggregate(outcomes, resamples=args.resamples, seed=args.seed)

    (out_dir / "metrics.json").write_text(
        json.dumps(result.as_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "per_pair.jsonl").write_text(
        "\n".join(json.dumps(o.as_dict()) for o in outcomes) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 68)
    print("M0 -- instruction-following under contradiction")
    print("=" * 68)
    print(result.summary())
    print("\nby family:")
    for fam, stats in result.by_family.items():
        s = stats["sensitivity"]
        d = stats["directional_ifr"]
        print(f"  {fam:18s} n={stats['n']:4d}  sensitivity={s['value']:.4f}", end="")
        if d:
            print(f"  IFR={d['value']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]")
        else:
            print()

    if resolved["checkpoint_is_base_model"]:
        print(
            "\nWARNING: this ran on the BASE checkpoint, which has not been trained on\n"
            "LIBERO. These numbers exercise the plumbing and must NOT be reported as M0."
        )

    print(f"\nwritten -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
