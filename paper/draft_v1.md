# Paper A — draft v2


> ## ⚠️ Superseded by the review re-analysis (2026-08-22)
>
> Two claims below are **corrected** in `paper/main.tex` and
> `results/reanalysis/metrics.json`. This document is kept for provenance; cite the paper.
>
> **The "93–96% substitution" figure is inflated.** Its chance floor is **0.83**: for
> `bowl → rack` the bowl was trained on the other three destinations, so *every* possible
> wrong answer is a trained one by construction and that cell reads 1.00 whatever the policy
> does. Excess over chance is **+0.111 [0.073, 0.144]** and **+0.098 [0.056, 0.134]**.
>
> **"Substitution" overstates it.** On novel commands the chosen destination still tracks the
> one named — the bottle told "the plate" goes there on 0.45 of trials, told "the stove" on
> 0.56 — which a fixed object-conditioned prior cannot produce. The supported claim is
> *partial compositional generalization with a trained-destination bias*, not a memorised
> lookup, and not an identified binding mechanism.
>
> All intervals are now **demonstration-clustered**; unclustered ones were too narrow.

**Target:** RoboPAD short (4 pp) primary · VLM4RWD (8 pp) · Interpretability-for-Discovery (5 pp).
Budgets below are for 4 pages, with expansion notes for 8.

**Every number traces to a committed run in `results/`.** Unrun work is `[TODO]`, never a
plausible placeholder.

**What changed from v1:** M2 localization and M3 transplant were run and are **null / invalid**.
They are now *in* the paper as a methods contribution rather than promised as future work.
The claim is narrower and better supported: we localize *when* grounding fails, not *where*.

---

## Working title

*Instruction grounding in VLAs fails compositionally, not globally*

Alternatives:
- *Memorised pairings, not grounded words*
- *When does a robot policy stop listening? Not when you'd expect*

---

## Abstract (~190 words)

> Vision-language-action models are widely reported to under-use their language input. We ask
> *when* this happens. Using minimal contrastive pairs on LIBERO-Goal — identical observations,
> instructions differing in exactly one referent — we find that a LIBERO-finetuned SmolVLA
> follows instructions well in the conditions such failures are usually attributed to: 75% on
> neutral states, and 93–95% when caught mid-trajectory and commanded to redirect, where the
> visual context actively opposes the instruction. Grounding fails only when the instruction
> requires *composition*. Commanding a pairing of familiar words never demonstrated in training
> ("put the bowl on the rack") degrades accuracy by 0.19–0.25 under a paired control that holds
> the observation fixed and varies only the object word, replicated across both publicly
> available competent checkpoints. Errors are structured: 93–96% send the arm to a destination
> that object *was* trained with, indicating substitution of a memorised pairing rather than
> failure to attend. A 130-site causal patching sweep did **not** localize this failure; we show
> its apparent structure is reproduced by a control in which no failure exists, and report the
> artifact rather than the map. A linear probe finds the instruction encoded in the
> vision-language backbone, but its expert-level results do not replicate across checkpoints, so
> we report no readout-versus-encoding verdict. We release the stimulus generator, a checkpoint
> competence gate, and the controls that caught five false positives in our own pipeline.

*Notes.* Cut the last sentence for a 150-word version. Leading with the two negative conditions
is deliberate: it is what makes "compositional" a specific claim rather than generic VLA-bashing,
and it preempts *isn't your policy just weak?*

---

## 1. Introduction (0.5 pp / 1 pp)

**P1 — setup.** VLAs map image + instruction to actions. A growing behavioural literature
reports they under-use the language and default to visual priors [BeTTER; ICBench; LIBERO-CF;
RoboSemanticBench]. This matters because the instruction is the entire control interface.

**P2 — the gap.** That literature establishes *that* failures occur, mostly in aggregate. It
says much less about *when* — under which linguistic demands grounding holds and under which it
breaks. Without that, mechanistic work has no well-posed target: you cannot localize a failure
you cannot reliably elicit.

**P3 — approach.** Minimal contrastive pairs: identical observations, instructions differing in
exactly one referent, so any change in the predicted action is attributable to the swapped word.
Three regimes of increasing linguistic demand.

**P4 — findings.** Grounding holds in the two regimes prior work would predict failure,
including direct visual-motor conflict, and breaks only under composition. The error pattern
identifies the mechanism as substitution of a memorised pairing. A causal patching sweep does not
localize it, and we show why the sweep's apparent structure is an artifact.

**Contributions**
1. A controlled localization of *when* VLA grounding fails: compositionally, not globally. Two
   null regimes are evidence, not discarded attempts.
2. A released, gate-validated contrastive stimulus generator, plus a reference-free directional
   readout that makes undemonstrated commands scoreable.
3. A competence gate showing only 2 of 8 public LIBERO SmolVLA checkpoints are usable at all.
4. A negative localization result *with the control that establishes it as negative* — and four
   measurement artifacts that produced clean, significant, false results in our own pipeline.

---

## 2. Related work (2–3 paragraphs / 1 pp)

**Behavioural failures in VLAs.** [BeTTER; ICBench; LIBERO-CF/CAG; RoboSemanticBench] report
instruction under-use. *Missing:* aggregate rates, not the linguistic operation that fails.

**Compositional generalisation.** [SCAN; gSCAN; CoGnition]. *Missing:* studied in language-only
or navigation settings, rarely in continuous-control policies whose output is a trajectory.

**Binding.** [binding IDs 2310.17191; VLM binding 2505.22200] predicts exactly the
object↔attribute failure we observe. *Missing:* characterised mechanistically on the LM side;
the behavioural signature in an action policy has not been.

**VLA interpretability.** [CWRU "Not All Features"; Häon steering; VLA-Trace; event-grounded
SAEs]. *Missing:* these localize features; they need a condition that reliably elicits the
failure, which is what we supply. Our negative sweep also speaks to how easily such methods
produce artifacts without matched controls.

*Expansion to 8pp:* add reproducibility-of-released-checkpoints work; our 2-of-8 competence
result speaks to it directly.

---

## 3. Method (1 pp + figure / 2 pp)

### 3.1 Minimal contrastive pairs
One observation, two instructions differing in exactly one referent. Gates, each with a
regression test: single-referent edit (word-level diff, one edit block, merged across filler
words, preposition-free span, verb unchanged); shared scene; both instructions achievable;
counterbalanced source tasks. Two gates exist because naive versions admitted invalid pairs —
worth one sentence each, as they are reusable.

### 3.2 Three regimes
- **Neutral** — pre-grasp; both instructions achievable, no visual prior favours either.
- **Conflict** — post-grasp; arm already carrying the object toward the demonstrated goal, so
  the observation opposes the alternative command. Destination swaps only.
- **Compositional** — trained vs never-demonstrated pairings. All ten LIBERO-Goal scenes share
  the same seven objects, so novel pairings are executable.

### 3.3 Readouts
- **Directional IFR** (regimes 1–2): does commanding the non-demonstrated instruction push the
  prediction away from the demonstrated trajectory? Chance 0.5.
- **Destination-direction accuracy** (regime 3): novel commands have no demonstration, so we
  score which *destination anchor* the predicted net displacement points toward. Anchors are the
  mean final end-effector position of demos ending there (sd 0.03–0.05 m, separation 0.19–0.57 m;
  the cabinet anchor agrees within 0.07 m across two objects, so it tracks destination not task).

### 3.4 Controls
Competence gate; pinned flow-matching noise (identical inputs → bit-identical actions, control
passes at exactly 0.0); trained-pairing control for the compositional readout; fixed-state
control holding the observation byte-identical while varying only the object word.

**Figure 1** — one contrastive pair: observation, two instructions, two predicted trajectories
over the destination anchors. Carries the method visually. `[TODO: make]`

---

## 4. Experiments (1 pp / 2–3 pp)

**Setup.** SmolVLA-450M, LIBERO-Goal, offline on stored demonstration states. Two competent
checkpoints. Paired bootstrap, 10k resamples.

**Table 1 — checkpoint screen.** 2 of 8 competent; four score *worse than predicting an
unrelated trajectory*. Two of those four were re-run at every image orientation and stay
near 1.8, so their scores are not our preprocessing. A result, not housekeeping.

**Table 2 — three regimes.**

| regime | readout | result |
|---|---|---|
| neutral (n=400) | directional IFR | 0.745 [0.703, 0.788] — follows |
| conflict (n=240) | directional IFR | 0.946 / 0.929 — follows |
| compositional (n=800, paired) | destination accuracy | gap **+0.250 [+0.197, +0.300]** / **+0.193 [+0.143, +0.243]** |

**Table 3 — compositional, per destination.** Six cells, all positive and significant
(plate +0.100/+0.180, stove +0.330/+0.140, rack +0.320/+0.260).

**The error analysis is the mechanism claim.** 93–96% of novel-command errors go to a
destination that object *was* trained with. Confident and wrong in a structured way.

### 4.5 The localization sweep is null, and we can show why  ← NEW, and load-bearing

Patching all 130 sites (`{vlm,expert} × L0–15 × {resid_pre, attn_out, mlp_out, resid_post}`)
from a working run into a failing one produced an apparently clean picture: VLM influence
decaying with depth, expert influence rising to 1.0.

**It is an artifact of causal proximity.** Patching an early site propagates through everything
downstream; patching a late one changes almost nothing else. Running the identical sweep on a
*control* contrast — both instructions trained, no failure to recover — reproduces the same
profile. Novel minus control: **mean −0.056, median −0.033**, and **0 of 130 sites survive
Benjamini-Hochberg in either run**.

**Figure 2** — novel and control profiles overlaid, plus their difference. The figure *is* the
argument. `[TODO: make]`

We also report that patching the final expert residual gives recovery 1.000 with zero variance:
a useful positive control that the machinery works end-to-end, and a trap for anyone who reads
it as localization.

### 4.6 Probing: the instruction is encoded, but the expert-level claim does not replicate

A linear probe trained on **trained** pairings and tested on **novel** ones, with a
label-shuffled control at every site and a **state-grouped split**, shows the named
destination is trivially decodable in the VLM (1.00 at several attention outputs, shuffled
controls at chance, on both checkpoints). That establishes the instruction is encoded and
survives the vision-language backbone. It is also close to definitional, since the
destination word is present in the input tokens.

**The claim we cannot make is about the action expert.** On one checkpoint the destination
appears strongly decodable inside the expert (37 sites above 0.5, peaking at 0.985 at
`expert.L7.attn_out`), which would be a clean readout failure. On the second competent
checkpoint it is not: only 2 expert sites exceed 0.5, and **29 sit below 0.10** — far
*below* the 0.25 chance level, meaning the probe is reliably *wrong* rather than
uninformative.

| expert tower, acc on novel pairings | finetune | scratch\_80k |
|---|---|---|
| sites > 0.5 (clean control) | 37 | 2 |
| sites < 0.10 (systematically wrong) | 3 | 29 |

We therefore **do not claim a readout failure.** An earlier draft did, on the first
checkpoint alone; the replication removed it.

**The below-chance pattern is a lead, not a result.** Systematic sub-chance decoding is not
noise: with four balanced classes it requires a consistent mapping to a *wrong* class. One
hypothesis fits the behavioural finding directly — if the expert represents the destination
the policy will *actually* move toward (the memorised substitute) rather than the one named,
then a probe trained where named and actual coincide will predict the substitute and be
reliably wrong on novel pairings. That is testable by decoding the *executed* destination
instead of the named one, which we have not run. If it held, the mechanism would be a
substitution at the representational level rather than a readout failure. We flag it as the
obvious next experiment rather than asserting it.

*8pp version adds:* the binding-transplant attempt and why its verdict is not reportable — the
difference-of-means direction was computed between instructions differing in the **object**, so
it encoded object identity rather than binding, and injecting it degraded behaviour monotonically
with strength (−1.4 to −5.8) while a norm-matched random control stayed at ~0. A worked example
of a plausible-looking intervention that measures the wrong thing.

---

## 5. Conclusion (1 para / 0.5 pp)

Grounding in this VLA does not fail globally — it fails at composition. Under direct
visual-motor conflict the policy still redirects on command; under an undemonstrated pairing of
familiar words it substitutes a memorised destination. Causal patching did not localize this,
and the control shows the sweep's apparent structure is proximity, not mechanism.

**Limitations.** Effect sizes are moderate and halved under the strictest control; both competent
checkpoints come from one uploader; LIBERO-Goal supports one object contrast, so results
generalise over states rather than referents; evaluation is offline.

**Next.** The probe shows the destination is available where the action is computed, so the
open question narrows: *why* is it not used? A success-vs-failure contrast on the same
instruction would isolate binding without the object confound that invalidated our transplant,
and position-resolved patching would separate language- from vision-token contributions.
`[TODO — no results]`

---

## Workshop fit

| venue | pp | fit | note |
|---|---|---|---|
| **RoboPAD (short)** | 4 | **strong** | short track explicitly solicits "preliminary findings, positions, benchmarks, and **negative results**" — this paper's exact shape |
| **VLM4RWD** | 8 | **strong** | scope names visual grounding, compositional reasoning, benchmarks for grounding evaluation, embodied AI |
| Interpretability for Discovery | 5 | **weak** | welcomes "failure cases and negative results", so §4.5 qualifies as a methods contribution — but the interpretability payload is currently a null. Stronger if the probe experiment lands before Aug 29. |

All three are **non-archival**, so simultaneous submission is permitted and none blocks a later
archival version. Verify each venue's dual-submission line before relying on it.

**Recommendation:** submit to RoboPAD and VLM4RWD regardless. Add I4D only if the probe result
arrives in time — otherwise it is a paper about interpretability that reports no interpretability
finding, which reviewers there will notice.
