# Where Does Language Die?

Causal, component-level localization of instruction-grounding failure in Vision-Language-Action
models. VLAs map a camera image + a language instruction to robot actions, but frequently ignore
the instruction and act on visual priors. Prior work shows *that* this happens; we find *where*
inside the model the instruction stops influencing the action, and *why*.

> **[`CLAUDE.md`](CLAUDE.md) is the source of truth** for the research questions, method, metrics,
> repo layout, conventions, and scope. Read it first. This README covers setup only.

Primary model: **SmolVLA-450M** (`lerobot/smolvla_base`). Primary environment: **LIBERO-Goal**.
We train nothing — all analysis is forward passes with hooks on frozen public checkpoints.

## Status

| Milestone | What | State |
|---|---|---|
| M0 | Instruction-following rate under contradiction | in progress |
| M1 | Contrastive-pair generator + released stimulus set | in progress |
| M2 | Causal localization map (layer × component) | not started |
| M3 | Binding transplant → encoding-vs-readout verdict | not started |
| M4 | Figures + workshop draft | not started |

No number appears in this repo unless it was produced by a run whose resolved config is committed
alongside it. Unrun experiments emit `TODO`/`NaN`, never plausible placeholders (CLAUDE.md §11).

## Quickstart (new clone)

Needs [uv](https://docs.astral.sh/uv/) and ~5 GB free (3.3 GB data + ~1.8 GB weights).
No GPU, no admin rights, no simulator required — everything below runs on CPU.

```bash
git clone https://github.com/mzkaell/vla-where-does-language-die.git
cd vla-where-does-language-die

make setup        # .venv on Python 3.12 + deps            (~3 min)
make test-fast    # 36 tests that need no weights/data      (~5 s)
make data         # LIBERO-Goal demos, 3.3 GB               (~10 min)
make smoke        # loads SmolVLA, predicts on a real state (~1 min)
make test         # full suite incl. patching correctness   (~10 min)
```

If you do not have `uv`:
`curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or
`irm https://astral.sh/uv/install.ps1 | iex` (Windows). It needs no admin rights and
installs its own Python, so your system Python is untouched.

Then reproduce the artifacts:

```bash
make pairs                                          # rebuild the 200-pair stimulus set
make ifr CKPT=k1000dai/smolvla_libero_finetune      # M0, ~26 min on 8 CPU cores
```

`make ifr` refuses to run without `CKPT`, on purpose — see *Checkpoints* below.

### Writing your own experiment

The model interface is in [src/models/base.py](src/models/base.py). A patching experiment
is three calls:

```python
from src.models.smolvla import SmolVLA

model = SmolVLA.load("k1000dai/smolvla_libero_finetune")

# 130 sites: {vlm,expert}.L{0..15}.{resid_pre,attn_out,mlp_out,resid_post} + final norms
run_b = model.forward_with_cache(batch_b, sites=["vlm.L8.resid_post"])
patched = model.patch(batch_a, patches={"vlm.L8.resid_post": run_b.occurrences("vlm.L8.resid_post")})
```

Two things to know before you trust a result:

- **Always pass the same `noise` to both arms.** SmolVLA is a flow-matching policy that
  integrates from sampled noise. `model.make_noise(batch_size)` gives a seeded tensor.
  Without it, run-to-run variance is confounded with the effect you are measuring.
- **Sites fire more than once.** The VLM prefix runs once; the action expert runs once per
  denoising step (10). `cache[site]` is a list, one tensor per invocation. `single()`
  raises on multi-occurrence sites rather than silently returning the first.

## Two environments, and why

The analysis stack and the simulator stack **cannot coexist in one environment**: LIBERO pins
`transformers==4.21.1` and `numpy==1.22.4`, while SmolVLA via LeRobot needs modern `transformers`
and `numpy>=1.24`. They are kept in separate venvs that communicate only through serialized
state/observation files on disk — never in-process.

This split is cheap because **most of the science does not need a live simulator**. M0–M3 operate
on *fixed paired states*, and LIBERO ships rendered HDF5 demonstrations, so pair construction and
all patching run offline. The simulator is needed only for the smoke test and for closed-loop
confirmation of the final shortlist.

### Analysis environment (any OS, CPU or CUDA)

```bash
make setup          # uv venv --python 3.12 + pip install -e ".[vla,interp,dev]"
make test
```

Python ≥3.12 is forced by LeRobot 0.6.0's own `requires-python`.

### Simulator environment (Linux only)

```bash
make setup-sim      # separate .venv-sim, Python 3.11
export MUJOCO_GL=osmesa
make smoke
```

**This does not work on native Windows.** robosuite officially supports Linux and macOS only, and
MuJoCo headless rendering on Windows (`MUJOCO_GL=osmesa`) was requested upstream and
[closed as "not planned"](https://github.com/google-deepmind/mujoco/issues/2164). The
`egl`→`wgl` edit to `robosuite/utils/binding_utils.py` that circulates online requires an
on-screen-capable context and is not true headless rendering. On Windows, use WSL2 or a Linux host.

## Pipeline

```bash
make smoke                      # load SmolVLA, run 1 headless LIBERO-Goal episode
make pairs   SUITE=libero_goal N=200   # M1: build + validate contrastive pairs
make ifr     MODEL=smolvla             # M0: instruction-following rate + bootstrap CI
make localization                      # M2 -- not built yet (fails loudly)
make transplant                        # M3 -- not built yet
make figures                           # M4 -- not built yet
```

Unbuilt targets exit non-zero on purpose. A stub that silently succeeds would let a phase look
finished when it is not.

## Layout

```
configs/      yaml: models, tasks, experiments (logged with every run)
src/models/   base.py + smolvla.py -- common interface; no model specifics leak out
src/data/     contrastive pair generation from LIBERO  (schema: src/data/README.md)
src/interp/   patching, path_patching, action_lens, sae, binding_transplant
src/eval/     ifr, metrics, stats (bootstrap, permutation, FDR)
src/viz/      heatmaps, redirection plots
scripts/      thin CLI entrypoints (call src/)
experiments/  runners; each writes its resolved config + results
results/      cached activations + metrics (large blobs gitignored, configs tracked)
stimuli/      released contrastive-pair sets (tracked -- these are release artifacts)
tests/        unit tests for interp + metrics
paper/        workshop draft + final figures
```

## Checkpoints

`lerobot/smolvla_base` is a **pretrained base model, not a LIBERO policy**. It declares a
6-dim state and 6-dim action against LIBERO's 8 and 7, and has never seen these tasks. An
instruction-following rate measured on it would describe an incompetent policy rather than
instruction grounding, so `scripts/reproduce_ifr.py` refuses to use it without
`--allow-base-checkpoint` (which exists only to exercise plumbing).

M0 therefore needs a LIBERO-finetuned SmolVLA. Several are public; none is official, and
**which one we use is a reproducibility claim in the paper**, so it is pinned in each run's
`results/<run_id>/config.yaml`. Known-good schemas (2 cameras, 8-dim state, 7-dim action):

| Checkpoint | Notes |
|---|---|
| `k1000dai/smolvla_libero_finetune` | used for the current pilot |
| `msv6/smolvla_meta_libero` | same schema, untested here |
| `bicmol/smolvla-libero` | channels-last shapes in config |

These older checkpoints carry normalization buffers that LeRobot 0.6.0's loader drops
(it logs `Unexpected key(s) ... normalize_inputs.buffer_observation_state`). The policy
would then silently consume raw proprioception and emit actions in normalized units.
`src.models.smolvla.load_norm_stats` recovers them; `model.has_norm_stats` tells you
whether it worked. Without them the divergence readout still holds (both arms are affected
equally) but any comparison against a demonstration action does not.

## Standing rules

- **No headline claim on a single technique.** Every headline claim needs ≥2 techniques with
  different failure modes agreeing. Knockout can miss information that reroutes; patching can push
  activations off-distribution.
- **Determinism.** Seeds are set and logged; every run dumps its fully-resolved config to
  `results/<run_id>/config.yaml`.
- **Cache aggressively**, keyed by `(model, input hash, site)`.
- **Small-first.** Validate any new analysis on a few tasks and a couple of layers before sweeping.

## Hardware note

CLAUDE.md budgets a single CUDA GPU. SmolVLA-450M forward passes are viable on CPU, which is
enough for M0/M1 and the patching unit tests; the M2 sweep over (layer × position × component)
across thousands of paired states is not, and needs a GPU.
