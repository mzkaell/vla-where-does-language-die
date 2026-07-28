#!/usr/bin/env python
"""Smoke test: load SmolVLA and run one LIBERO-Goal episode headless.

    python scripts/smoke_test.py

Two independent checks, because they have different dependencies and different failure
modes. The model half runs anywhere; the simulator half is Linux-only.

  [1] MODEL   load the checkpoint, build one observation from the released LIBERO
              demonstrations, predict an action chunk, print its shape.
  [2] SIM     step one LIBERO-Goal episode under MUJOCO_GL=osmesa and report success/fail.

Check [2] cannot pass on native Windows. robosuite officially supports Linux and macOS,
and MuJoCo headless rendering on Windows was requested upstream and closed as "not
planned" (google-deepmind/mujoco#2164). The `egl`->`wgl` edit that circulates online needs
an on-screen-capable context and is not true headless. Use WSL2 or a Linux host.

Check [1] alone is enough for M0-M3, which run offline on fixed stored states.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OK, FAIL, SKIP = "  OK  ", " FAIL ", " SKIP "


def check_model(checkpoint: str, data_root: Path, device: str) -> bool:
    print("\n[1] MODEL")
    try:
        import numpy as np
        import torch

        from src.data.build_pairs import load_observation, load_pairs
        from src.models.smolvla import SmolVLA, make_batch
        from scripts.reproduce_ifr import build_images, build_state
    except Exception as exc:
        print(f"[{FAIL}] imports: {exc}")
        return False

    try:
        print(f"       loading {checkpoint} ...")
        model = SmolVLA.load(checkpoint, device=device)
        cfg = model.config
        n_params = sum(p.numel() for p in model.policy.parameters())
        print(f"[{OK}] loaded, {n_params / 1e6:.1f}M params on {device}")
        print(f"       VLM layers={model.num_layers}  denoising steps={cfg.num_steps}")
        print(f"       patchable sites: {len(model.sites())}")
    except Exception as exc:
        print(f"[{FAIL}] load: {exc}")
        return False

    pairs_file = REPO_ROOT / "stimuli" / "libero_goal_pairs_v1.jsonl"
    if not pairs_file.exists():
        print(f"[{SKIP}] no stimulus file; run scripts/build_pairs.py first")
        return True

    try:
        pair = load_pairs(pairs_file)[0]
        obs = load_observation(pair, data_root)
        image_keys = list(cfg.image_features)
        size = cfg.image_features[image_keys[0]].shape[-1]

        batch = make_batch(
            build_images(obs, image_keys, size),
            build_state(obs, cfg.robot_state_feature.shape[0]),
            pair["instruction_a"],
            model.policy,
            device,
        )
        action = model.predict_action(batch)
        print(f"[{OK}] instruction: {pair['instruction_a']!r}")
        print(f"[{OK}] predicted action chunk shape: {tuple(action.shape)}")
        assert torch.isfinite(action).all(), "action contains NaN/inf"

        # Determinism is a prerequisite for every patching result.
        again = model.predict_action(batch)
        print(f"[{OK}] deterministic across repeat calls: {bool(torch.equal(action, again))}")

        # And the instruction must actually matter.
        batch_b = make_batch(
            build_images(obs, image_keys, size),
            build_state(obs, cfg.robot_state_feature.shape[0]),
            pair["instruction_b"],
            model.policy,
            device,
        )
        noise = model.make_noise(1)
        d = float(
            torch.linalg.vector_norm(
                (
                    model.predict_action(batch, noise=noise)
                    - model.predict_action(batch_b, noise=noise)
                ).reshape(-1)
            )
        )
        print(f"[{OK}] action divergence between the two instructions: {d:.5f}")
    except Exception as exc:
        print(f"[{FAIL}] forward pass: {exc}")
        return False
    return True


def check_sim(task_suite: str) -> bool:
    print("\n[2] SIMULATOR")

    if platform.system() == "Windows":
        print(f"[{SKIP}] native Windows cannot run LIBERO headless.")
        print("       robosuite supports Linux/macOS only, and MUJOCO_GL=osmesa on Windows")
        print("       was closed upstream as 'not planned' (google-deepmind/mujoco#2164).")
        print("       Fix: run this under WSL2 or on a Linux host (`make setup-sim`).")
        print("       This does NOT block M0-M3, which run offline on stored states.")
        return True

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:
        print(f"[{SKIP}] LIBERO not installed in this env ({exc}).")
        print("       Fix: make setup-sim   (a SEPARATE venv -- LIBERO pins")
        print("       transformers==4.21.1 and numpy==1.22.4, which conflict with LeRobot)")
        return True

    try:
        suite = benchmark.get_benchmark_dict()[task_suite]()
        task = suite.get_task(0)
        env = OffScreenRenderEnv(
            bddl_file_name=str(task.bddl_file),
            camera_heights=128,
            camera_widths=128,
        )
        env.reset()
        for _ in range(5):
            obs, reward, done, info = env.step([0.0] * 7)
        env.close()
        print(f"[{OK}] stepped 5 frames of {task.name!r} headless (MUJOCO_GL=osmesa)")
        print(f"[{OK}] agentview_rgb shape: {obs['agentview_image'].shape}")
        return True
    except Exception as exc:
        print(f"[{FAIL}] simulator: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print(f"SMOKE TEST  ({platform.system()} {platform.machine()})")
    print("=" * 68)

    ok = check_model(args.checkpoint, args.data_root, args.device)
    if not args.skip_sim:
        ok = check_sim(args.suite) and ok

    print("\n" + "=" * 68)
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
