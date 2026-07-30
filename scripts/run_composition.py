#!/usr/bin/env python
"""Compositional grounding test: familiar words, novel pairings.

    python scripts/run_composition.py --checkpoint k1000dai/smolvla_libero_finetune

Reads the METHOD docstring in src/eval/composition.py before interpreting anything. The
short version: M0 only ever commanded the 10 tasks the policy was trained on, which a
sentence-matching policy passes without composing word meanings. This asks for
combinations that are executable in the scene but were never demonstrated.

The trained-composition arm is the control. If it does not score well above chance, the
readout is broken and the novel-composition number means nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_ifr import STATE_KEYS, build_images, build_state  # noqa: E402
from src.data.build_pairs import post_commitment_timesteps  # noqa: E402
from src.eval.composition import (  # noqa: E402
    DESTINATIONS,
    OBJECT_SOURCE_TASKS,
    aggregate,
    build_anchors,
    instruction_for,
    score_direction,
)


def task_endpoints(suite_dir: Path, n_demos: int = 50) -> dict[str, np.ndarray]:
    """Mean final end-effector position per task -- the location of its destination.

    Keyed by the task's natural-language instruction (read from the file, not inferred
    from the filename) so it joins against `DESTINATIONS` without a naming convention
    having to be maintained in two places.
    """
    out = {}
    for f in sorted(glob.glob(str(suite_dir / "*.hdf5"))):
        with h5py.File(f, "r") as h:
            d = h["data"]
            instruction = json.loads(d.attrs["problem_info"])["language_instruction"].strip()
            ends = [d[k]["obs"]["ee_pos"][-1] for k in list(d.keys())[:n_demos]]
        out[instruction] = np.mean(ends, axis=0)
    return out


def collect_states(suite_dir: Path, tasks: list[str], per_task: int, seed: int):
    """Post-grasp states: the object is held, so motion is destination-directed."""
    rng = np.random.default_rng(seed)
    states = []
    for task in tasks:
        path = suite_dir / f"{task}_demo.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as h:
            d = h["data"]
            demos = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
            for di in rng.permutation(len(demos))[: max(1, per_task // 2)]:
                name = demos[int(di)]
                for t, _prog in post_commitment_timesteps(d[name], 2, stride=1):
                    obs = d[name]["obs"]
                    states.append(
                        {
                            "task": task,
                            "demo": name,
                            "t": int(t),
                            "agentview_rgb": obs["agentview_rgb"][t],
                            "eye_in_hand_rgb": obs["eye_in_hand_rgb"][t],
                            "ee_pos": obs["ee_pos"][t],
                            "ee_ori": obs["ee_ori"][t],
                            "gripper_states": obs["gripper_states"][t],
                        }
                    )
                    if len(states) >= per_task * len(tasks):
                        break
    return states[: per_task * len(tasks)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--per-task", type=int, default=20, help="states per source task")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    suite_dir = args.data_root / args.suite
    run_id = args.run_id or f"composition_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPO_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    anchors = build_anchors(task_endpoints(suite_dir))
    print(f"run_id     : {run_id}")
    print(f"checkpoint : {args.checkpoint}")
    print("\ndestination anchors (mean final EE position of the demos ending there):")
    for d, a in anchors.items():
        print(f"  {d:22s} [{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}]")
    seps = [
        np.linalg.norm(anchors[a] - anchors[b])
        for i, a in enumerate(anchors)
        for b in list(anchors)[i + 1 :]
    ]
    print(f"  min separation between anchors: {min(seps):.3f} m")

    from src.models.smolvla import SmolVLA, make_batch

    print("\nloading model ...")
    model = SmolVLA.load(args.checkpoint, device=args.device)
    cfg = model.config
    keys = list(cfg.image_features)
    size = cfg.image_features[keys[0]].shape[-1]
    sdim = cfg.robot_state_feature.shape[0]
    print(f"norm stats : {'recovered' if model.has_norm_stats else 'NONE'}")

    (out_dir / "config.yaml").write_text(
        "\n".join(
            f"{k}: {json.dumps(v)}"
            for k, v in {
                "run_id": run_id,
                "experiment": "compositional_grounding",
                "checkpoint": args.checkpoint,
                "per_task": args.per_task,
                "seed": args.seed,
                "destinations": list(DESTINATIONS),
                "anchors": {k: [float(x) for x in v] for k, v in anchors.items()},
                "state_composition": list(STATE_KEYS),
                "image_orientation": "rot180",
                "platform": platform.platform(),
            }.items()
        ),
        encoding="utf-8",
    )

    outcomes = []
    skipped = 0
    t0 = time.time()
    for obj, tasks in OBJECT_SOURCE_TASKS.items():
        states = collect_states(suite_dir, tasks, args.per_task, args.seed)
        print(f"\n{obj}: {len(states)} post-grasp states from {len(tasks)} tasks")
        for si, st in enumerate(states):
            obs = {k: st[k] for k in ("agentview_rgb", "eye_in_hand_rgb")}
            obs |= {k: st[k] for k in STATE_KEYS}
            images = build_images(obs, keys, size)
            state_t = model.normalize_state(build_state(obs, sdim))

            for dest in anchors:
                instruction = instruction_for(obj, dest)
                batch = make_batch(images, state_t, instruction, model.policy, args.device)
                act = model.unnormalize_action(
                    model.predict_action(batch, noise=model.make_noise(1))
                )
                # Net commanded translation over the chunk: dx, dy, dz are dims 0..2.
                delta = act[0, :, :3].sum(dim=0).cpu().numpy().astype(np.float64)

                res = score_direction(
                    trial_id=f"{st['task']}__{st['demo']}__t{st['t']}__{obj}__{dest}",
                    obj=obj,
                    named_destination=dest,
                    ee_pos=np.asarray(st["ee_pos"], dtype=np.float64),
                    predicted_delta=delta,
                    anchors=anchors,
                )
                if res is None:
                    skipped += 1
                else:
                    outcomes.append(res)
            if (si + 1) % 10 == 0:
                rate = len(outcomes) / max(time.time() - t0, 1e-9)
                print(f"    {si + 1}/{len(states)} states  ({rate:.2f} trials/s)", flush=True)

    if not outcomes:
        print("no scoreable trials", file=sys.stderr)
        return 1

    summary = aggregate(outcomes, resamples=args.resamples, seed=args.seed)
    summary["skipped_zero_motion"] = skipped
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "per_trial.jsonl").write_text(
        "\n".join(json.dumps(o.as_dict()) for o in outcomes) + "\n", encoding="utf-8"
    )

    chance = summary["chance_level"]
    print("\n" + "=" * 70)
    print("COMPOSITIONAL GROUNDING")
    print("=" * 70)
    print(f"chance = {chance:.3f} ({summary['n_destinations']} destinations)")
    if skipped:
        print(f"skipped {skipped} trials with no commanded motion")

    for label in ("trained", "novel"):
        b = summary[label]
        if not b:
            print(f"\n{label}: too few trials")
            continue
        a = b["accuracy"]
        flag = "ABOVE chance" if a["lo"] > chance else "at chance"
        print(
            f"\n{label:>8} compositions  n={b['n']:<4} accuracy "
            f"{a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]  {flag}"
        )
        print(f"{'':>8} margin over best competitor: {b['margin']['value']:+.4f}")

    if summary.get("within_destination"):
        print("\nPRIMARY: within-destination (same destination word, trained vs novel object)")
        print(f"  {'destination':<22}{'trained':>18}{'novel':>18}{'gap':>8}")
        for dest, b in summary["within_destination"].items():
            t, nv = b["trained"], b["novel"]
            print(
                f"  {dest:<22}{t['accuracy']:>10.3f} (n={t['n']:<3}){nv['accuracy']:>10.3f} "
                f"(n={nv['n']:<3}){b['gap']:>+8.3f}"
            )
        print("  holds the destination prior fixed; only the object and trained/novel differ")

    print("\nchosen-destination distribution (a flat-ish spread means no runaway prior):")
    for label, sel in (
        ("trained", [o for o in outcomes if not o.novel]),
        ("novel", [o for o in outcomes if o.novel]),
    ):
        if not sel:
            continue
        counts: dict[str, int] = {}
        for o in sel:
            counts[o.chosen_destination] = counts.get(o.chosen_destination, 0) + 1
        tot = len(sel)
        spread = ", ".join(
            f"{k}={v / tot:.2f}" for k, v in sorted(counts.items(), key=lambda x: -x[1])
        )
        print(f"  {label:>8}: {spread}")

    ctrl = summary["trained"]
    if ctrl and ctrl["accuracy"]["lo"] <= chance:
        print(
            "\nCONTROL FAILED: trained compositions are at chance, so the direction readout\n"
            "is not measuring destination selection. The novel-composition number above is\n"
            "NOT interpretable.",
            file=sys.stderr,
        )
    elif summary["novel"] and ctrl:
        gap = ctrl["accuracy"]["value"] - summary["novel"]["accuracy"]["value"]
        print(f"\ntrained - novel gap: {gap:+.3f}")
        if summary["novel_error_destinations"]:
            print("where novel commands went when wrong:")
            for k, v in sorted(
                summary["novel_error_destinations"].items(), key=lambda x: -x[1]
            ):
                print(f"    {v:>4}  {k}")

    print(f"\nwritten -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
