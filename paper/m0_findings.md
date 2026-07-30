# M0: the grounding failure is compositional

*Assumes you know VLAs and activation patching; assumes nothing about this project. Every
number below comes from a run in `results/` with its resolved config committed beside it.*

---

## Bottom line

**SmolVLA follows instructions it was trained on (75–95%) and fails almost completely on
novel combinations of the same familiar words. When it fails, 96–97% of the time it heads to
a destination that object *was* trained with — it substitutes a memorised pairing rather than
wandering.** This replicates on both competent checkpoints and survives a
destination-prior control.

That is an object↔destination **binding** failure: the object word is grounded (behaviour
stays object-appropriate) while the destination word is not bound to it outside memorised
pairs. It is the effect CLAUDE.md §3 predicts, and it gives M2 a target *with a matched
control condition* — same model, same scene, same destination word, one pairing working and
one not.

Two earlier conditions were null. They are reported in full below, because the reason they
were null is what identified the compositional design, and because "the effect is specifically
compositional, not general instruction-blindness" is a claim those nulls support.

| condition | what it varied | result |
|---|---|---|
| neutral states | destination or object word, pre-grasp | **null** — 0.745 IFR, follows instruction |
| visual conflict | same, but mid-trajectory toward a goal | **null** — 0.93–0.95, follows instruction |
| **compositional** | trained vs never-demonstrated pairings | **effect** — pooled gap +0.35 to +0.41 |

---

## Setup

SmolVLA-450M (LIBERO-finetuned), evaluated offline on LIBERO-Goal. All ten LIBERO-Goal tasks
share one kitchen scene containing the same seven objects (verified from the MuJoCo models),
so vision underdetermines the target and the instruction must be read.

Offline throughout: predictions on fixed stored states from the released demonstrations. No
simulator, exactly reproducible, and cheap enough to run hundreds of trials on a laptop CPU.

## Competence gate

**This runs before anything else.** A grounding score is uninterpretable unless the policy can
do the task — a policy producing noise scores at chance, which is indistinguishable from a
policy that ignores language.

Ratio = median ‖prediction − demonstration‖ ÷ the distance between two *unrelated*
demonstrations. Below ~0.7 means predictions track the specific trajectory
(`scripts/check_competence.py`, needs no forward passes when a run already exists).

Screening all eight public checkpoints with a compatible schema (`results/checkpoint_screen.json`):

| checkpoint | ratio | verdict |
|---|---|---|
| `k1000dai/smolvla_libero_scratch_80k` | 0.632 | **competent** |
| `k1000dai/smolvla_libero_finetune` | 0.654 | **competent** |
| `k1000dai/smolvla-libero-pick-up-…-20k` | 0.808 | not competent |
| `bicmol/smolvla-libero` | 1.673 | worse than baseline |
| `AustineJohnBreaker/smolvla_stratch_libero_spatial` | 1.676 | worse than baseline |
| `xainyuxxx/my-smolvla-libero` | 1.732 | worse than baseline |
| `msv6/smolvla_meta_libero` | 1.868 | worse than baseline |
| `jadechoghari/smolvla-libero-ckpts` | — | fails to load (config schema drift) |

**Only 2 of 8 are usable, and both come from the same uploader.** They differ in training
regime (one finetunes `smolvla_base`, the other trains from scratch), so agreement between
them is worth something — but they plausibly share a data pipeline, so this is a **weak
replication**, not two independent sources. No independent competent SmolVLA checkpoint
appears to be public. See *Limitations*.

The four failures are not our preprocessing. Re-running two across all four image
orientations leaves them near 1.8 regardless (`msv6` 1.78–2.06, `bicmol` 1.82 flat to three
decimals — meaning the image barely affects its output). They are broken on LIBERO-Goal.

---

## Condition 1 — neutral states: null

Minimal contrastive pairs: one observation, two instructions differing in exactly one
referent, both arms run with identical flow-matching noise.

```
observation:    demo_18, frame 0          <- byte-identical across arms
instruction A:  put the bowl on the plate
instruction B:  put the bowl on the stove
```

States drawn **pre-grasp**, so nothing has been manipulated and both instructions remain
achievable. 400 counterbalanced pairs. Readout is *directional IFR*: the state comes from a
demonstration of one task, so commanding the other should push the prediction away from that
demonstrated trajectory (chance 0.5).

| | destination (n=320) | object (n=80) | all |
|---|---|---|---|
| Directional IFR | 0.725 [0.675, 0.775] | 0.825 [0.738, 0.900] | 0.745 [0.703, 0.788] |

**The policy follows the instruction.** Same-instruction control passes at exactly 0.0
(identical inputs → bit-identical actions, so no nondeterminism inflates anything).

*Why null:* pre-grasp states are chosen so both instructions stay achievable — which also
guarantees nothing in the image contradicts either one. This measured instruction correctness
on neutral states, not grounding under conflict.

## Condition 2 — visual conflict: also null

240 post-commitment pairs, destination swaps only. The arm is already carrying the object
toward the demonstrated goal, so the observation carries a prior for it and commanding the
other destination puts vision and language in opposition. Restricted to destination swaps by
construction: with the object already held, an object swap is a different task (put this
down, pick that up) rather than a redirection.

| | finetune | scratch_80k |
|---|---|---|
| Directional IFR | 0.946 [0.917, 0.971] | 0.929 [0.896, 0.958] |

**Grounding got *better*, not worse.** The policy redirects mid-trajectory when told to.

### The apparent decline at high conflict is a metric artefact

Both checkpoints dip in the most-committed bin (finetune 0.950→0.983→0.900; scratch
0.975→0.983→0.783), which looks like the predicted effect. It is not. Sensitivity collapses
in the same bin (scratch 8.48 → 3.79), and when the instruction barely moves the action its
*direction* is estimated from almost no signal, so the score decays toward chance
mechanically. Splitting that bin by sensitivity:

| within high-conflict bin | low sensitivity | high sensitivity |
|---|---|---|
| finetune | 0.842 [0.711, 0.947] | 1.000 [1.000, 1.000] |
| scratch_80k | 0.684 [0.526, 0.816] | 0.949 [0.872, 1.000] |

corr(sensitivity, followed) = +0.250 and +0.399. Where the instruction *can* influence the
action late in the trajectory, grounding is essentially perfect.

**This generalizes and is a live risk for M2: a site where the instruction cannot matter looks
identical to a site where it is ignored. Any use of the directional readout must condition on
sensitivity.**

The two regimes are also not directly comparable — pre-grasp, both instructions demand the
same first move, so the readout sits near chance by construction. Only within-regime
comparisons are fair.

---

## Condition 3 — compositional: the effect (`results/comp_*`)

Both nulls shared one flaw: **every instruction was one of the 10 tasks the policy was trained
on.** A policy can pass that by recognising which memorised sentence it heard, without ever
treating "bowl" and "rack" as recombinable meanings. The design gave it no way to fail.

Since all ten scenes contain the same seven objects, we can command pairings that are
physically executable but were never demonstrated:

```
trained:  bowl -> plate, stove, cabinet        bottle -> rack, cabinet
NOVEL:    bowl -> rack                         bottle -> plate, stove
```

**Reference-free readout.** A novel command has no demonstration, so the demo-trajectory
reference used above does not exist. Instead each destination gets an *anchor* — the mean
final end-effector position of the demos ending there (sd 0.03–0.05 m, separations 0.19–0.57 m,
and the cabinet anchor agrees within 0.07 m across two different objects, so it tracks the
destination rather than the task). We score which anchor direction the predicted net
displacement matches. Cosine, so the controller's action scaling is irrelevant.

**Control:** trained compositions must beat chance or the readout is broken. They do —
0.737 [0.686, 0.785] and 0.699 [0.647, 0.750] against chance 0.25.

### Primary result — within destination, 480 trials per checkpoint

The raw trained-vs-novel aggregate is confounded and is *not* the headline: the policy has an
object-independent destination prior (it heads for the cabinet 45–49% of the time regardless)
and per-destination accuracy spans 0.19–0.95, so the two sets' different destination mixes
contribute to any aggregate difference. Holding the destination word fixed removes it — same
anchor, same target direction, only the object and trained/novel status change.

| destination | finetune trained → novel | scratch_80k trained → novel |
|---|---|---|
| the plate | 0.694 → 0.229 · **+0.465 [+0.305, +0.618]** | 0.597 → 0.083 · **+0.514 [+0.375, +0.646]** |
| the stove | 0.889 → 0.312 · **+0.576 [+0.431, +0.722]** | 0.806 → 0.500 · **+0.306 [+0.139, +0.472]** |
| the rack | 0.188 → 0.000 · **+0.188 [+0.083, +0.292]** | 0.229 → 0.014 · **+0.215 [+0.097, +0.340]** |
| **pooled** | **+0.410 [+0.329, +0.491]** | **+0.345 [+0.264, +0.431]** |

Every destination, both checkpoints, significant.

### Fixed-state control: the effect survives at roughly half the size (`results/fs_*`)

The result above still confounded the object word with the **source state** — for "the stove",
trained meant bowl→stove from bowl demos while novel meant bottle→stove from bottle demos.
The control removes it: every state is scored under **both** object words, so the observation
is byte-identical across the comparison and the state cancels exactly. One command is then
necessarily counterfactual (the named object is not the one in the gripper), which is the
point — it isolates the object token from the scene. Because both arms now share states, the
comparison is **paired**.

| destination | finetune (n=100 states) | scratch_80k (n=100 states) |
|---|---|---|
| the plate | 0.550 → 0.450 · **+0.100 [+0.030, +0.180]** | 0.520 → 0.340 · **+0.180 [+0.100, +0.270]** |
| the stove | 0.890 → 0.560 · **+0.330 [+0.240, +0.420]** | 0.810 → 0.670 · **+0.140 [+0.060, +0.220]** |
| the rack | 0.330 → 0.010 · **+0.320 [+0.230, +0.410]** | 0.310 → 0.050 · **+0.260 [+0.170, +0.350]** |
| **pooled paired** | **+0.250 [+0.197, +0.300]** | **+0.193 [+0.143, +0.243]** |

**All six cells positive and significant.** But the gaps are roughly **half** the uncontrolled
estimates (+0.41 / +0.35 → +0.25 / +0.19), so a substantial part of the original effect *was*
the source state rather than the object word. The uncontrolled numbers above should be read as
an upper bound; **these are the ones to quote.**

Novel-command accuracy is also much higher here (0.34 vs 0.16), because the pooled state set
means momentum sometimes happens to agree with a novel command. That inflation applies
identically to both arms — they share states — so the paired gap is unaffected.

### The mechanism is substitution, not confusion

**93–96% of novel-command errors go to a destination that object *was* trained with**
(187/198 and 180/194 under the fixed-state control; 136/142 and 135/139 without it). The
policy does not wander or freeze — it confidently executes a memorised pairing, and the named
destination loses by a clear margin rather than narrowly.

Object word grounded, destination word unbound outside memorised pairs. That is the binding
signature, and it is what M3's transplant test is designed to adjudicate: is the correct
binding absent from the stream feeding the action expert (encoding failure), or present but
unread (readout failure)?

---

## Limitations

1. **Effect size is about half the first estimate.** The fixed-state control (above) shrank
   the pooled gap from +0.41/+0.35 to +0.25/+0.19. All six cells stayed positive and
   significant, but the uncontrolled numbers were inflated by the source state and should be
   treated as an upper bound. Quote the paired figures.
2. **Weak replication.** Both competent checkpoints come from one uploader. OpenVLA, with
   official LIBERO checkpoints, would be a genuinely independent check.
3. **One object pairing.** LIBERO-Goal supports exactly one object contrast
   (`bowl`↔`wine bottle`), so results generalize over *states*, not *referents*. LIBERO-Object
   would broaden this.
4. **Offline, not closed-loop.** "Correct" means heading toward the named destination, not
   task success.
5. **Competence is adequate, not comfortable** (0.61–0.65), on unofficial checkpoints.

## Lessons recorded

Ten defects surfaced during M0. The dangerous ones were never the crashes — they were the
**four that produced clean, significant, plausible numbers**:

1. **Counterbalancing** — 100% of states came from the A-side task, turning the readout into a
   test of instruction asymmetry. Produced a striking destination-vs-object dissociation that
   was pure artefact, and was only exposed when a second checkpoint returned the *opposite*
   strongly-significant result.
2. **Image orientation** — LIBERO stores frames under MuJoCo's OpenGL convention. The policy
   saw upside-down scenes and scored no better than predicting a random trajectory, while
   still yielding confident IFR numbers.
3. **Directional reference** — compared a 50-step prediction against a single action repeated
   50 times, scoring a policy that stalls as highly as one that follows through.
4. **Destination prior** in the compositional readout — caught *before* reporting, by checking
   the distribution of chosen destinations rather than only the accuracy.

Each was caught by a check that **could disagree with the result**: a second checkpoint, a
balance audit, a competence baseline, an output-distribution check. None was caught by reading
code that looked correct, or docstrings that asserted the very properties that were missing.

**Consequence for M2:** a layer × component heatmap offers far fewer opportunities to visibly
contradict itself than a scalar did. The competence gate and the replication checkpoint must
run *before* any map is interpreted, the permutation null over shuffled positions matters more
than the headline effect, and every site-level score must condition on sensitivity.

## Next

**M2 is unblocked** — there is now an effect to localize, with a matched control. The
infrastructure applies unchanged: 130 tap points over
`{vlm, expert} × L0–L15 × {resid_pre, attn_out, mlp_out, resid_post}`, patching verified by
self-patch identity plus perturbed negative controls and an equivalence test pinning the
instrumented forward bitwise against stock LeRobot, prefix positions separable into
`[image][language][state]`, and bootstrap/permutation/BH-FDR in `src/eval/stats.py`.

Order: fixed-state control (done) → M2 sweep on the novel-vs-trained
contrast, per-family and position-resolved (needs a GPU; 130 sites × trials × 2 arms is not
viable at the ~5–10 s/trial seen on CPU) → M3 transplant targeting the novel condition, where
binding demonstrably fails and the trained condition supplies a working comparison.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[vla,dev]"
.venv/bin/python scripts/download_data.py --all
.venv/bin/python scripts/screen_checkpoints.py                       # competence first
.venv/bin/python scripts/build_pairs.py --suite libero_goal --n 400
.venv/bin/python scripts/reproduce_ifr.py   --checkpoint k1000dai/smolvla_libero_finetune
.venv/bin/python scripts/run_composition.py --checkpoint k1000dai/smolvla_libero_finetune
```

Windows paths and troubleshooting are in the README. Tests cover patching correctness, the
statistics, and the stimulus validity gates.
