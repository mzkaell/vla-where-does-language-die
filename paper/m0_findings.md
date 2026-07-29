# M0: object reference is not grounded; destination grounding is weak and checkpoint-dependent

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
> checkpoints have been rerun. The corrected results are below; the retracted magnitudes
> (0.99 / 0.45) do not survive. Sensitivity was never affected: it is symmetric in the two
> arms and never references the demonstration.

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

## Results (400 counterbalanced pairs, paired bootstrap, 10k resamples)

| | k1000dai | msv6 |
|---|---|---|
| Sensitivity ‖a_A − a_B‖ | 7.87 [7.62, 8.13] | 8.08 [7.90, 8.27] |
| IFR, **destination** (n=320) | 0.647 [0.594, 0.700] — above chance, p≈1e-4 | 0.472 [0.419, 0.525] — at chance, p=0.34 |
| IFR, **object** (n=80) | 0.487 [0.375, 0.588] — at chance, p=0.91 | 0.500 [0.388, 0.613] — at chance, p=1.0 |
| destination − object | **+0.159 [+0.037, +0.281]** significant | −0.028 [−0.150, +0.097] n.s. |
| Same-instruction control | 0.0 — PASS | 0.0 — PASS |

**Object swaps pooled across both checkpoints: 0.494 [0.419, 0.569], n=160, p=0.94.**

### What replicates

**Object reference is not grounded, in either checkpoint.** Both sit on 0.5, and pooled
they give 0.494 — as close to chance as this design can measure. Meanwhile sensitivity in
the object condition is high (6.55 and 9.11): swapping `bowl` → `wine bottle` clearly moves
the action, just not toward the named object. This is the one result that survives both
bugs and both checkpoints.

### What does not replicate

**Destination grounding.** k1000dai shows a real but modest effect (0.647, with a
significant +0.159 advantage over object swaps); msv6 shows nothing (0.472, at chance). The
destination-vs-object dissociation therefore holds in one checkpoint and is absent in the
other — a far weaker claim than the retracted 0.99-vs-0.45.

The likeliest explanation is checkpoint quality rather than architecture: msv6 is at chance
on *everything*, which is what a policy that never learned to condition on language would
look like. That would make it uninformative rather than contradictory. But it is a
hypothesis. Distinguishing "does not ground language" from "our metric cannot see its
grounding" needs a competence check we have not run — closed-loop success, or whether its
predictions track the demonstration at all under the *correct* instruction.

### Still standing

**The same-instruction control passes at exactly 0.0** on both runs: identical inputs give
bit-identical actions, so no nondeterminism inflates any divergence above.

**The infrastructure.** 130 tap points across
`{vlm, expert} × L0–L15 × {resid_pre, attn_out, mlp_out, resid_post}`, with patching
verified by self-patch identity plus perturbed negative controls, and an equivalence test
pinning the instrumented forward bitwise against stock LeRobot.

## What we don't know yet

1. **Is msv6 a competent policy?** It is at chance on everything. Until we check whether it
   can do LIBERO at all, its disagreement with k1000dai cannot be interpreted. This is the
   single highest-value next check and it is cheap.
2. **The object arm rests on one pairing.** LIBERO-Goal contains exactly one object swap
   (`bowl`↔`wine bottle`). Rebuilding on all ten tasks raised it to n=80, but that adds
   states, not referent diversity. The headline result therefore generalizes over *states*,
   not over *referents* — a third checkpoint would not fix this, but LIBERO-Object would.
3. **Both checkpoints are unofficial.** No official LIBERO-finetuned SmolVLA exists.
4. **Offline, not closed-loop.** These are action predictions on stored states, so "correct"
   means resembling the demonstrated trajectory, not task success.
5. **Chance-level object IFR is a null result.** It is consistent with "no grounding", but
   also with the metric being underpowered for object swaps at n=80. The destination
   condition in k1000dai shows the metric *can* detect an effect at this n, which is
   reassuring but not conclusive.

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
