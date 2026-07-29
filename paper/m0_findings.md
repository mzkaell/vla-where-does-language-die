# M0: SmolVLA follows *where*, not *what*

*Interim. Assumes you know VLAs and activation patching; assumes nothing about this
project. Numbers live in `results/`, with each run's resolved config beside them.*

---

## The finding in one line

**Swap the destination in an instruction and the policy adapts almost perfectly. Swap the
object and it performs at chance — while still changing its action.**

## Setup

SmolVLA-450M (LIBERO-finetuned), evaluated offline on LIBERO-Goal. We use LIBERO-Goal
because all ten of its tasks share one kitchen scene with a fixed object set, so vision
underdetermines the target and the instruction has to be read.

The stimulus is a **minimal contrastive pair**: one observation, two instructions differing
in exactly one referent.

```
observation:    demo_18, frame 0          <- byte-identical across arms
instruction A:  put the bowl on the plate
instruction B:  put the bowl on the stove
```

Since the pixels are identical, any change in the predicted action is attributable to the
swapped span. Two families: **destination** swaps (`plate`→`stove`) and **object** swaps
(`bowl`→`wine bottle`). States are drawn pre-grasp, so nothing has been manipulated yet and
both instructions remain achievable. Both arms run with identical flow-matching noise.

Two readouts, chosen to fail differently:

- **Sensitivity** `‖a_A − a_B‖` — does the instruction change the action *at all*? A
  language-ignoring policy scores 0, so it can't be faked. But it's direction-blind.
- **Directional IFR** — does it change *correctly*? The state comes from a demonstration of
  one of the two tasks; commanding the other should push the prediction away from that
  demonstrated trajectory. Chance = 0.5.

## Results (n=200, paired bootstrap, 10k resamples)

| | Destination (n=160) | Object (n=40) |
|---|---|---|
| Sensitivity | 8.29 [7.85, 8.73] | 6.72 [6.30, 7.21] |
| Directional IFR | **0.994 [0.98, 1.00]** | **0.45 [0.30, 0.60]** |

Aggregate IFR is 0.885 [0.84, 0.93], but it's dominated by the 160 destination pairs and
misleads on its own. The result is the split.

## Why it matters

The object condition is **not** the model ignoring language — sensitivity is high there.
It registers the lexical change and then moves in a way unrelated to the new referent.
That's a *binding* failure, not an attention failure, and it's exactly what this project
set out to localize.

It also arrives in a more tractable form than a single failure rate: a **within-model
dissociation**. Same network, same scene, same measurement — one referent type works, the
other doesn't. M2 therefore gets a built-in control. Instead of asking "where does
instruction information live," we can trace the working destination pathway against the
failing object pathway through the same 130 tap points and find where they diverge.

The direction is consistent with the binding-ID literature: a destination may be
recoverable as a fairly direct goal-location signal, whereas resolving which of several
visible objects a noun denotes needs an object↔word binding — the operation CLAUDE.md §3
predicts the action expert fails to read.

## What we don't know yet

1. **Rerun in flight with a corrected reference.** The n=200 numbers above scored a 50-step
   prediction against a *single* demo action repeated 50 times. Fixed to use the demo's
   true next-50-step chunk; results pending. Both families were scored identically, so the
   dissociation should survive, but treat the absolute values as provisional.
2. **The object arm rests on one pairing.** LIBERO-Goal contains exactly one object swap
   (`bowl`↔`wine bottle`). Rebuilding on all ten tasks raised it to n=80, but that adds
   states, not referent diversity. Real breadth needs another suite.
3. **One checkpoint, unofficial.** No official LIBERO-finetuned SmolVLA exists. A second
   checkpoint is running now; if the dissociation doesn't reproduce, it's a property of one
   upload rather than of SmolVLA.
4. **Offline, not closed-loop.** These are action predictions on stored states, so "correct"
   means resembling the demonstrated trajectory, not task success.

Passing already: the **same-instruction control** returns exactly 0.0 divergence, ruling out
nondeterminism inflating any of these numbers.

## What's next

**Now (CPU, in progress)** — rerun both checkpoints with the corrected reference on 400
pairs.

**Then (needs a GPU)** — M2 localization. Infrastructure is built and tested: 130 tap points
over `{vlm, expert} × L0–L15 × {resid_pre, attn_out, mlp_out, resid_post}`, patching
verified by self-patch identity plus perturbed negative controls. Run activation patching
separately per family and compare pathways; patch language-token positions independently of
vision positions (the prefix is `[image][language][state]`, so this is clean) to separate
*never encoded* from *encoded but not read*; permutation null plus BH FDR across sites.
130 × 400 × 2 is not viable at the ~11 s/pair we see on CPU.

**After** — M3 binding transplant, targeting the object condition since that's where binding
demonstrably fails and the destination condition supplies a working comparison. Verdict is
*readout failure* if injecting the binding direction into the expert's input recovers ≥50%
of the gap.

Per the standing rule, no headline claim ships until a second technique with different
failure modes agrees.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[vla,dev]"
.venv/bin/python scripts/download_data.py --all
.venv/bin/python scripts/build_pairs.py --suite libero_goal --n 400
.venv/bin/python scripts/reproduce_ifr.py --checkpoint k1000dai/smolvla_libero_finetune
```

Windows paths and troubleshooting are in the README. 57 tests cover patching correctness,
the statistics, and the stimulus validity gates.
