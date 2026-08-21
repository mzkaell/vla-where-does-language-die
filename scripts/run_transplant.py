"""M3 — binding transplant (CLAUDE.md §7–8): the encoding-vs-readout verdict.

    # after M2 names the site, e.g.:
    python scripts/run_transplant.py --checkpoint k1000dai/smolvla_libero_finetune \
        --extract-site vlm.L8.resid_post --alphas 0.25,0.5,1.0 --device mps

Same contrasts, states, and recovery readout as run_localization.py — the two
milestones must be directly comparable. What M3 adds over M2's full patch: the
injection is the working-minus-failing *delta*, at a controlled dose, optionally
restricted to a token-position slice (--positions a:b). alpha=1, same-site, all
positions reduces exactly to M2 for vlm.* sites (which fire once) and doubles as the
sanity check there; on expert sites state feedback across denoising steps makes the
alpha=1 number a different quantity from M2's recovery — do not compare them.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_ifr import STATE_KEYS, build_images, build_state  # noqa: E402
from scripts.run_composition import collect_states, task_endpoints  # noqa: E402
from scripts.run_localization import CONTRASTS  # noqa: E402
from src.eval.composition import build_anchors, instruction_for  # noqa: E402
from src.interp.localization import (  # noqa: E402
    TrialBaseline,
    direction_cosine,
    net_translation,
)
from src.interp.transplant import (  # noqa: E402
    binding_delta,
    dosed_patch,
    judge,
    restrict_to_positions,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--extract-site", required=True)
    ap.add_argument("--inject-site", default=None, help="defaults to the extract site")
    ap.add_argument("--alphas", default="0.25,0.5,1.0")
    ap.add_argument(
        "--positions", default=None, help="token slice a:b, or 'lang' for the language block"
    )
    ap.add_argument("--n-trials", type=int, default=10, help="states per contrast")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-trials", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    inject = args.inject_site or args.extract_site
    if args.positions == "lang" and not args.extract_site.startswith("vlm."):
        # expert activations are action tokens; prefix arithmetic applied to them
        # returns plausible-looking garbage positions, not an error
        sys.exit("--positions lang is only defined for vlm.* extract sites")
    alphas = [float(a) for a in args.alphas.split(",")]
    if len(set(alphas)) != len(alphas):
        # duplicates double-append every trial for that alpha: pseudo-replication
        # that narrows the bootstrap CI and can flip 'indeterminate' to 'readout'
        sys.exit(f"duplicate alphas in {alphas}")
    pos = None
    if args.positions and args.positions != "lang":
        a, b = args.positions.split(":")
        pos = list(range(int(a), int(b)))
        if not pos:
            sys.exit(f"--positions {args.positions} selects nothing")

    run_id = args.run_id or f"m3_{args.extract_site.replace('.', '_')}"
    out_dir = REPO_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_dir = args.data_root / args.suite
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    anchors = build_anchors(task_endpoints(suite_dir))

    from src.models.smolvla import SmolVLA, make_batch, pair_pad_length

    print(f"run_id  : {run_id}\ncheckpoint: {args.checkpoint}")
    print(
        f"extract : {args.extract_site}  inject: {inject}  alphas: {alphas}  pos: {args.positions}"
    )
    model = SmolVLA.load(args.checkpoint, device=args.device)
    cfg = model.config
    keys = list(cfg.image_features)
    size = cfg.image_features[keys[0]].shape[-1]
    sdim = cfg.robot_state_feature.shape[0]

    (out_dir / "config.yaml").write_text(
        "\n".join(
            f"{k}: {json.dumps(v)}"
            for k, v in {
                "run_id": run_id,
                "experiment": "M3_transplant",
                "checkpoint": args.checkpoint,
                "extract_site": args.extract_site,
                "inject_site": inject,
                "alphas": alphas,
                "positions": args.positions,
                "n_trials_per_contrast": args.n_trials,
                "contrasts": CONTRASTS,
                "suite": args.suite,
                "data_root": str(args.data_root),
                "min_trials": args.min_trials,
                "seed": args.seed,
                "device": args.device,
                "image_orientation": "rot180",
                "state_composition": list(STATE_KEYS),
                "platform": platform.platform(),
                "torch": torch.__version__,
            }.items()
        ),
        encoding="utf-8",
    )

    cosp: dict[float, list[float]] = {a: [] for a in alphas}
    disp_per_alpha: dict[float, list[float]] = {a: [] for a in alphas}
    baselines: list[TrialBaseline] = []  # one per scored trial, shared by every alpha
    skipped = 0
    t0, n_done = time.time(), 0

    from src.eval.composition import OBJECT_SOURCE_TASKS

    # contrast-invariant: same pooled tasks and seed every time, so collect once
    pooled = [t for ts in OBJECT_SOURCE_TASKS.values() for t in ts]
    states = collect_states(suite_dir, pooled, args.n_trials, args.seed)[: args.n_trials]

    for contrast in CONTRASTS:
        dest = contrast["destination"]
        anchor = anchors[dest]
        instr_w = instruction_for(contrast["working_object"], dest)
        instr_f = instruction_for(contrast["failing_object"], dest)
        pad_len = pair_pad_length(model.policy, [instr_w, instr_f])
        print(f"\ncontrast: '{dest}'  ({len(states)} states)", flush=True)

        for st in states:
            obs = {k: st[k] for k in ("agentview_rgb", "eye_in_hand_rgb")}
            obs |= {k: st[k] for k in STATE_KEYS}
            images = build_images(obs, keys, size)
            state_t = model.normalize_state(build_state(obs, sdim))
            ee = np.asarray(st["ee_pos"], dtype=np.float64)
            noise = model.make_noise(1)

            batch_w = make_batch(
                images, state_t, instr_w, model.policy, args.device, pad_to_length=pad_len
            )
            batch_f = make_batch(
                images, state_t, instr_f, model.policy, args.device, pad_to_length=pad_len
            )

            run_w = model.forward_with_cache(batch_w, sites=[args.extract_site], noise=noise)
            run_f = model.forward_with_cache(batch_f, sites=[args.extract_site], noise=noise)

            cos_w = direction_cosine(
                net_translation(model.unnormalize_action(run_w.action)), ee, anchor
            )
            cos_f = direction_cosine(
                net_translation(model.unnormalize_action(run_f.action)), ee, anchor
            )
            base = TrialBaseline(
                trial_id=f"{st['task']}__{st['demo']}__t{st['t']}__{dest}",
                cos_working=cos_w,
                cos_failing=cos_f,
                headroom=cos_w - cos_f,
                ee_pos=ee,
                anchor=anchor,
            )
            if not base.usable:
                skipped += 1
                continue

            deltas = [
                binding_delta(w, f)
                for w, f in zip(
                    run_w.occurrences(args.extract_site),
                    run_f.occurrences(args.extract_site),
                    strict=True,
                )
            ]
            if pos is not None and pos[-1] >= deltas[0].shape[1]:
                # on 'longest'-padded checkpoints the prefix length varies per
                # contrast, so a fixed numeric slice can walk off the end (or
                # silently name different tokens); die here, not after hours
                sys.exit(
                    f"--positions {args.positions} exceeds this contrast's "
                    f"{deltas[0].shape[1]}-token activation; on variable-pad "
                    f"checkpoints use --positions lang"
                )
            if args.positions == "lang":
                from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

                from src.models.smolvla import language_token_positions

                n_lang = batch_f[OBS_LANGUAGE_TOKENS].shape[1]
                pos = language_token_positions(model.policy, deltas[0].shape[1], n_lang)
            if pos is not None:
                deltas = [restrict_to_positions(d, pos) for d in deltas]

            for alpha in alphas:
                patched = model.patch(
                    batch_f, patches={inject: dosed_patch(deltas, alpha)}, noise=noise
                )
                d = net_translation(model.unnormalize_action(patched.action))
                cosp[alpha].append(direction_cosine(d, ee, anchor))
                disp_per_alpha[alpha].append(float(np.linalg.norm(d)))
            baselines.append(base)

            n_done += 1
            if n_done == 1 or n_done % 5 == 0:
                per = (time.time() - t0) / n_done
                todo = len(CONTRASTS) * len(states) - n_done
                print(
                    f"    trial {n_done}  ({per:.1f}s/trial, eta {per * todo / 60:.1f} min)",
                    flush=True,
                )

    verdicts = [
        judge(
            args.extract_site,
            inject,
            a,
            cosp[a],
            baselines,
            disp_per_alpha[a],
            seed=args.seed,
            min_trials=args.min_trials,
        )
        for a in alphas
    ]

    print("\n" + "=" * 60 + "\nM3 TRANSPLANT\n" + "=" * 60)
    print(f"trials used {n_done}  skipped {skipped} (no headroom)")
    for v in verdicts:
        flags = " DEGENERATE" if v.degenerate else ""
        flags += f" dropped={v.n_dropped_nonfinite}" if v.n_dropped_nonfinite else ""
        print(
            f"  alpha={v.alpha:<5} recovery {v.recovery_mean:+.3f} "
            f"[{v.recovery_lo:+.3f},{v.recovery_hi:+.3f}]  n={v.n}  -> {v.verdict}{flags}"
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "verdicts": [asdict(v) for v in verdicts],
                "n_trials_used": n_done,
                "n_skipped": skipped,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
