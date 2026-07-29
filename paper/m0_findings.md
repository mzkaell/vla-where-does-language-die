# M0 does not reproduce: SmolVLA grounds both referent types

**Bottom line: on the one checkpoint that can actually do the task, instruction following is
substantially above chance for both referent types — object swaps *better* than destination
swaps. The instruction-grounding failure this project set out to localize does not appear in
this design. Per CLAUDE.md §4 that changes the plan.**

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

## Competence gate comes first

A directional score means nothing unless the policy can do the task. Ratio is
median ‖prediction − demonstration‖ over the distance between two *unrelated*
demonstrations, so <1 means the prediction tracks this specific trajectory
(`scripts/check_competence.py`).

| | ratio | verdict |
|---|---|---|
| k1000dai | **0.614** | competent — predictions track the demonstration |
| msv6 | **1.998** | worse than an unrelated trajectory — **excluded** |

Rotating the frames 180° moved k1000dai from 0.981 to 0.614 but made msv6 *worse*
(1.771 → 1.998), so msv6 either expects different preprocessing or is simply broken. Its
uniformly at-chance scores are therefore uninformative, not a contradicting replication.
**We have one usable checkpoint, so the standing rule is not yet satisfied.**

## Results — k1000dai only (400 counterbalanced pairs, 10k resamples)

| | destination (n=320) | object (n=80) | all (n=400) |
|---|---|---|---|
| Sensitivity ‖a_A − a_B‖ | 6.18 | 5.24 | 5.99 [5.76, 6.23] |
| Directional IFR | 0.725 [0.675, 0.775] | **0.825 [0.738, 0.900]** | 0.745 [0.703, 0.788] |
| vs chance | above, p≈1e-4 | above | above, p≈1e-4 |
| wrong-direction rate | 27.5% | 17.5% | 25.5% |

destination − object = **−0.100 [−0.191, −0.003]**, significant. Same-instruction control
passes at exactly 0.0.

msv6, for the record (uninterpretable, incompetent): IFR 0.480 [0.433, 0.528], at chance on
everything.

### The premise does not reproduce

**SmolVLA follows the instruction here.** 0.745 overall, well above chance, and objects are
grounded *better* than destinations — the reverse of the retracted dissociation, which was
an artefact of upside-down images plus a one-sided design.

There is no "language dies" effect in this setup to localize. RQ1 asks where instruction
identity *stops* influencing the action; in these stimuli it never stops.

### Why: this design never created a contradiction

Reviewing the design against CLAUDE.md §1, the gap is clear. The project is about
instruction following **under contradiction** — where the visual prior and the instruction
*disagree*. These stimuli never construct that disagreement:

- LIBERO-Goal deliberately holds the scene fixed across tasks, so no visual prior favours
  either instruction.
- States are drawn **pre-grasp**, precisely so both instructions remain achievable — which
  also guarantees nothing in the image contradicts either one.

So M0 measured instruction *sensitivity and correctness* on neutral states, not grounding
under conflict. A ~75% success rate on a neutral, unconflicted swap is roughly what a
working policy should give. The null is a property of the stimuli, not evidence that VLAs
ground language robustly.

### Still standing

**The same-instruction control passes at exactly 0.0** on both runs: identical inputs give
bit-identical actions, so no nondeterminism inflates any divergence above.

**The infrastructure.** 130 tap points across
`{vlm, expert} × L0–L15 × {resid_pre, attn_out, mlp_out, resid_post}`, with patching
verified by self-patch identity plus perturbed negative controls, and an equivalence test
pinning the instrumented forward bitwise against stock LeRobot.

## The decision this forces

M0 was the gate: establish the behavioural effect before localizing it. It did not
establish one, so **M2/M3 cannot start** — patching would localize an effect that isn't
there. Three ways forward, cheapest first.

**A. Build a real contradiction condition (CPU, ~half a day).** Keep everything and change
the states: draw from **post-commitment** timesteps, where the demonstration is already
executing task A, then command B. Now the visual context genuinely conflicts with the
instruction. This directly tests CLAUDE.md's premise, reuses the whole pipeline, and needs
no GPU. It also has a natural difficulty knob (how far into the trajectory), so grounding
can be measured as a function of conflict strength. *Recommended.*

Note this trades away one validity gate: post-grasp states are no longer states where both
instructions are equally achievable. That is the point — but it means "correct" has to be
redefined, since the demonstration is no longer a neutral reference.

**B. Change suite.** LIBERO-Object/-Spatial vary the scene with the task, so the visual
prior does favour one reading. Cheap, but loses the fixed-scene control that made
LIBERO-Goal attractive, and reintroduces the visual confound the design was built to avoid.

**C. Change model.** OpenVLA-7B has *official* LIBERO checkpoints — no provenance caveat,
no competence gamble — plus discrete action tokens giving a genuine action lens instead of
SmolVLA's probe surrogate. But it needs the 40GB A100 in CLAUDE.md §12 and abandons the
single-consumer-GPU premise.

## Other open items

1. **One usable checkpoint.** msv6 is excluded on competence, so nothing replicates yet. A
   third public checkpoint should be screened with `check_competence.py` *before* any
   further analysis — that check costs no forward passes.
2. **The object arm rests on one pairing.** LIBERO-Goal contains exactly one object swap
   (`bowl`↔`wine bottle`). n=80 generalizes over *states*, not *referents*; a third
   checkpoint would not fix that, but LIBERO-Object would.
3. **Competence is adequate, not comfortable.** 0.614 means the prediction is meaningfully
   closer to the right trajectory than to a random one, but this is still an unofficial
   community checkpoint.
4. **Offline, not closed-loop.** "Correct" means resembling the demonstrated trajectory, not
   task success.

## Lessons recorded

Five bugs surfaced during M0, and the dangerous ones were not the crashes — they were the
three that produced clean, significant, plausible numbers:

1. **Directional reference** compared a 50-step prediction against one action repeated 50
   times.
2. **Counterbalancing** — 100% of states came from the A-side task, turning the readout into
   a test of instruction asymmetry. Produced a striking dissociation that was pure artefact.
3. **Image orientation** — LIBERO stores frames under MuJoCo's OpenGL convention, so the
   policy saw upside-down scenes and scored no better than predicting a random trajectory.

Each was caught only by a check that *could disagree with the result*: a second checkpoint,
a balance audit, a competence baseline. None was caught by reading the code, which looked
correct, or the docstrings, which asserted the very properties that were missing.

**Consequence for M2:** a layer × component heatmap offers far fewer opportunities to
visibly contradict itself than a single scalar did. The competence gate and a replication
checkpoint have to run *before* any map is interpreted, and the permutation null over
shuffled positions matters more than the headline effect.

## Held, pending the decision above

M2 localization and M3 binding transplant are **not started and should not start** until a
behavioural effect exists to localize.

The infrastructure is built and tested and will apply unchanged to whichever condition
replaces this one: 130 tap points over
`{vlm, expert} × L0–L15 × {resid_pre, attn_out, mlp_out, resid_post}`, patching verified by
self-patch identity plus perturbed negative controls, positions separable into
`[image][language][state]` so language-token patching can be isolated from vision, and
paired bootstrap / permutation / BH-FDR machinery in `src/eval/stats.py`.

Cost when it does run: 130 sites × 400 pairs × 2 arms is not viable at the ~5–10 s/pair
observed on CPU, so M2 needs a GPU.

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
