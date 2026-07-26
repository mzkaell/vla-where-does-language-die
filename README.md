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
