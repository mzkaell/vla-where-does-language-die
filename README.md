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
| M0 | Instruction-following rate under contradiction | pilot run, see below |
| M1 | Contrastive-pair generator + released stimulus set | 200 pairs released |
| M2 | Causal localization map (layer × component) | not started |
| M3 | Binding transplant → encoding-vs-readout verdict | not started |
| M4 | Figures + workshop draft | not started |

No number appears in this repo unless it was produced by a run whose resolved config is committed
alongside it. Unrun experiments emit `TODO`/`NaN`, never plausible placeholders (CLAUDE.md §11).

### M0 pilot — `results/m0_pilot_k1000dai/`

`k1000dai/smolvla_libero_finetune`, 200 LIBERO-Goal pairs, paired bootstrap (10k resamples).

| Readout | All pairs | Destination swap (n=160) | Object swap (n=40) |
|---|---|---|---|
| Instruction sensitivity ‖a_A − a_B‖ | 7.98 [7.61, 8.35] | 8.29 [7.85, 8.73] | 6.72 [6.30, 7.21] |
| Directional IFR (chance = 0.5) | 0.885 [0.84, 0.93] | **0.994 [0.98, 1.00]** | **0.45 [0.30, 0.60]** |

The aggregate is dominated by the 160 destination pairs and should not be quoted alone. The
result is the **dissociation**: the policy tracks a swapped *destination* almost perfectly, but
a swapped *object* leaves it at chance — while still changing its action substantially. It
reacts to the object word without grounding it.

**Caveat, unresolved.** The directional readout compares a 50-step predicted chunk against a
*single* demonstration action broadcast across all 50 steps, which is a weak reference. Both
families are scored the same way, so the dissociation is unlikely to be an artifact of it, but
the absolute values should not be trusted until the reference is the demonstration's true
next-50-step chunk. Fixing that is the next task, along with more object-swap pairs (n=40 is
thin, and its interval spans chance).

## Setup for teammates

**No GPU, no admin rights, and no simulator are required.** Everything below runs on a
laptop CPU. Budget ~5 GB of disk (3.3 GB of LIBERO demos + ~1.8 GB of model weights) and
about 15 minutes, most of it downloading.

### 1. Install uv

The only prerequisite. It installs its own Python, so your system Python is untouched, and
it needs no administrator rights.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
```
```powershell
irm https://astral.sh/uv/install.ps1 | iex           # Windows
```

Restart your shell afterwards so `uv` is on `PATH`.

### 2. Clone and set up

**Windows does not ship `make`,** so use these commands directly. They are exactly what the
Makefile runs, and they work on every platform.

```bash
git clone https://github.com/mzkaell/vla-where-does-language-die.git
cd vla-where-does-language-die

uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[vla,dev]"     # Windows
uv pip install --python .venv/bin/python           -e ".[vla,dev]"   # macOS / Linux
```

Everything after this uses that interpreter. Substitute `.venv/bin/python` on macOS/Linux:

```bash
.venv/Scripts/python.exe -m pytest -m "not slow" -q   # 36 tests, no weights/data  (~5 s)
.venv/Scripts/python.exe scripts/download_data.py     # LIBERO demos, 3.3 GB      (~10 min)
.venv/Scripts/python.exe scripts/smoke_test.py        # loads SmolVLA, predicts    (~1 min)
.venv/Scripts/python.exe -m pytest -q                 # full suite                (~10 min)
```

On macOS/Linux the `make` targets are equivalent and shorter: `make setup`, `make test-fast`,
`make data`, `make smoke`, `make test`. Run `make help` for the full list.

### 3. Confirm it worked

`smoke_test.py` should print a predicted action chunk shape and confirm determinism. The
simulator half will report **SKIP** on Windows — that is expected and blocks nothing (see
*Two environments* below).

### 4. Reproduce the artifacts

```bash
.venv/Scripts/python.exe scripts/build_pairs.py --suite libero_goal --n 200
.venv/Scripts/python.exe scripts/reproduce_ifr.py --checkpoint k1000dai/smolvla_libero_finetune
```

The second refuses to run without `--checkpoint`, on purpose — see *Checkpoints* below.
It takes ~30 min for 200 pairs on 8 CPU cores; add `--limit 20` for a quick check first.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `make: command not found` | Windows has no `make`. Use the direct commands above. |
| `No solution found when resolving dependencies` | You installed the `sae` or `nnsight` extra. They are mutually incompatible and unused; install `.[vla,dev]`. |
| `No module named pytest` | The install step failed silently. Re-run it without `-q` and read the error. |
| `Device 'cuda' is not available. Switching to 'cpu'` | Harmless. Everything in Phase 1 is CPU-only. |
| `Unexpected key(s) ... normalize_inputs` | Expected. We recover those buffers ourselves; see *Checkpoints*. |
| `no LIBERO .hdf5 files under ...` | Run `scripts/download_data.py`. Check with `--check`. |
| Simulator check says SKIP on Windows | Expected. MuJoCo headless is Linux-only; M0–M3 do not need it. |

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
