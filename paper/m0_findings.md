# M0: status — directional result retracted, rerun in progress

*Interim. Assumes you know VLAs and activation patching; assumes nothing about this
project. Numbers live in `results/`, with each run's resolved config beside them.*

---

> ## ⚠️ Retraction
>
> **An earlier version of this document reported a destination-vs-object dissociation
> (destination IFR 0.99, object 0.45). That result is withdrawn — it came from a bug in
> the stimulus generator, not from the model.**
>
> Every pair drew its state from the **A-side task only**; the generator consumed task A's
> states fully before reaching task B, and callers never took enough pairs to get there.
> That turns the directional readout into a test of instruction asymmetry rather than of
> grounding, because the question "does commanding the *other* instruction move the action
> away from the demonstration" was only ever asked in one direction.
>
> The symptom that exposed it: after fixing a separate reference bug, the two checkpoints
> returned **opposite** and strongly significant scores on identical stimuli — 0.73 above
> chance vs 0.33 below chance, p ≈ 1e-4. A real model property does not invert between
> checkpoints while staying that significant; a one-sided design interacting with each
> policy's own bias does.
>
> The stimulus set is now counterbalanced 40/40 across all five pairings and both
> checkpoints are rerunning. **All directional IFR numbers below this box are from the
> broken set and should not be cited.** Sensitivity is unaffected: it is symmetric in the
> two arms and never references the demonstration.

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

## What survives

**Instruction sensitivity**, which never touches the demonstration and is symmetric across
the two arms, so neither bug reaches it. Stable across every run and both checkpoints:

| | k1000dai | msv6 |
|---|---|---|
| Sensitivity ‖a_A − a_B‖ (n=400) | 8.10 [7.82, 8.38] | 8.08 [7.90, 8.26] |

Swapping one referent moves the predicted action substantially, and reliably. That
establishes the necessary precondition for the project — the instruction does reach the
action — but says nothing about whether the model grounds it *correctly*. Direction is
exactly what the retracted readout was supposed to supply.

**The same-instruction control passes at exactly 0.0.** Running one arm twice under fixed
noise gives bit-identical actions, so no nondeterminism inflates any divergence here.

**The infrastructure.** 130 tap points across
`{vlm, expert} × L0–L15 × {resid_pre, attn_out, mlp_out, resid_post}`, with patching
verified by self-patch identity plus perturbed negative controls, and an equivalence test
pinning the instrumented forward bitwise against stock LeRobot.

## What we don't know yet

1. **Whether any directional effect exists at all.** Pending the counterbalanced rerun on
   both checkpoints.
2. **The object arm rests on one pairing.** LIBERO-Goal contains exactly one object swap
   (`bowl`↔`wine bottle`). Rebuilding on all ten tasks raised it to n=80, but that adds
   states, not referent diversity. Real breadth needs another suite.
3. **Both checkpoints are unofficial.** No official LIBERO-finetuned SmolVLA exists.
4. **Offline, not closed-loop.** These are action predictions on stored states, so "correct"
   means resembling the demonstrated trajectory, not task success.

## Lesson recorded

Three separate stimulus-validity bugs have now been caught by inspecting generator output
rather than by the gates themselves: two non-minimal pair types, and this counterbalancing
failure. In each case the code looked right and the docstring asserted the property that
was missing. The counterbalancing bug is the instructive one — it produced *clean,
significant, plausible* numbers, and was only exposed because a second checkpoint
disagreed in the opposite direction.

Practical consequence for M2: run the replication checkpoint **before** interpreting any
localization map, not after. A one-sided design produces heatmaps that look just as
convincing as real ones.

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
