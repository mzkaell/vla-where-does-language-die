# Draft v1 — "Paper A" (behavioural + methods)

**Target:** RoboPAD short (4 pp) / VLM4RWD (8 pp). Section budgets below are for the 4-page
version, with expansion notes for 8.

**Every number here is from a committed run in `results/`.** Placeholders for unrun work are
marked `[TODO]` and must never be filled with plausible values.

---

## Working title

*Instruction grounding in VLAs fails compositionally, not globally*

Alternatives:
- *When does language stop mattering to a robot policy? Not when you'd think*
- *Memorised pairings, not grounded words: compositional failure in vision-language-action models*

---

## Abstract (target 150–200 words)

> Vision-language-action models (VLAs) are widely reported to under-use their language input,
> acting on visual priors instead of the instruction. We ask when this actually happens. Using
> minimal contrastive pairs on LIBERO-Goal — identical observations, instructions differing in
> exactly one referent — we find that a LIBERO-finetuned SmolVLA follows instructions well
> under the conditions such failures are usually attributed to: 75% on neutral states, and 93–95%
> when caught mid-trajectory and told to redirect, where the visual context actively opposes the
> command. The failure appears only when the instruction requires *composition*. Commanding a
> pairing of familiar words that was never demonstrated ("put the bowl on the rack") degrades
> accuracy by 0.19–0.25 relative to a trained pairing under a paired control that holds the
> observation fixed and varies only the object word, replicated across both publicly available
> competent checkpoints. Errors are not random: 93–96% send the arm to a destination that object
> *was* trained with. The policy substitutes a memorised pairing rather than binding the named
> destination to the named object. We release the stimulus generator, a competence gate that
> screens unusable checkpoints, and document four measurement artefacts that each produced
> clean, significant, and wrong results.

*Notes.* ~200 words; trim the pitfalls clause for a 150-word version. Leads with the negative
result because it is what makes the positive one specific.

---

## 1. Introduction (0.5 pp short / 1 pp long)

**Paragraph 1 — setup.** VLAs map an image and a language instruction to actions. A growing
behavioural literature reports that they under-use the language, defaulting to visual priors
[cite BeTTER, ICBench, LIBERO-CF, RoboSemanticBench]. This matters because instruction
following is the entire interface: a policy that ignores the words is not steerable.

**Paragraph 2 — the gap.** That literature establishes *that* failures occur, largely in
aggregate. It says much less about *when* — under which linguistic demands grounding holds and
under which it breaks. Without that, mechanistic work has no well-posed target: you cannot
localize a failure you cannot reliably elicit.

**Paragraph 3 — what we did.** We probe with minimal contrastive pairs: identical observations,
instructions differing in exactly one referent, so any change in the predicted action is
attributable to the swapped word. We test three regimes of increasing linguistic demand.

**Paragraph 4 — the finding.** Grounding holds in the two regimes where prior work would
predict failure, including direct visual-motor conflict, and breaks only when the instruction
requires composing familiar words into an undemonstrated pairing. The error pattern identifies
the mechanism: substitution of a memorised pairing, i.e. an object↔destination *binding*
failure rather than a failure to attend to language.

**Contributions** (bulleted):
1. A controlled localization of *when* VLA grounding fails: compositionally, not globally.
   Two null regimes are part of the evidence, not discarded attempts.
2. A released, gate-validated contrastive stimulus generator for LIBERO-Goal, plus the
   reference-free directional readout that makes undemonstrated commands scoreable.
3. A competence gate showing only 2 of 8 public LIBERO SmolVLA checkpoints are usable at all —
   with implications for how this literature is evaluated.
4. Four measurement artefacts that produced clean, significant, false results in our own
   pipeline, each caught only by a check capable of contradicting the headline.

*Expansion to 8 pp:* add a paragraph on why LIBERO-Goal specifically (fixed scene, shared
object set) and a forward pointer to the localization section.

---

## 2. Related work (2–3 paragraphs short / 1 pp long)

**Behavioural failures in VLAs.** [BeTTER; ICBench/linguistic blindness; LIBERO-CF/CAG;
RoboSemanticBench] report instruction under-use. *What is missing:* these characterise
aggregate rates; they do not isolate the linguistic operation that fails, so the results do not
tell you which manipulations reliably elicit the failure.

**Compositional generalisation.** [cite standard compositionality work — SCAN, CoGnition,
gSCAN for grounded settings]. *What is missing:* largely studied in language-only or
navigation settings; less examined in continuous-control policies where the readout is a
trajectory rather than a token sequence.

**Binding.** [binding IDs 2310.17191; VLM binding 2505.22200]. Predicts exactly the
object↔attribute failure mode we observe behaviourally. *What is missing:* the binding
literature is mechanistic and mostly language-model-side; the behavioural signature in an
action policy has not been characterised.

**Interpretability of VLAs.** [CWRU "Not All Features"; Häon steering; VLA-Trace;
event-grounded SAEs]. *What is missing:* these localize features; our contribution upstream of
them is a behavioural condition that reliably elicits the failure, which such methods need as a
target.

*Expansion to 8 pp:* add evaluation-methodology work on reproducibility of released
checkpoints, which our competence-gate finding speaks to directly.

---

## 3. Method (1 pp + figure short / 2 pp long)

### 3.1 Minimal contrastive pairs
One observation, two instructions differing in exactly one referent. Validity gates, each with
a regression test: single-referent edit (word-level diff, exactly one edit block, merged across
filler words, preposition-free span, verb unchanged); shared scene; both instructions
achievable from the state; counterbalanced source tasks.

*State the gates as gates.* Two of them exist because naive versions admitted invalid pairs —
worth one sentence each, since they are reusable.

### 3.2 Three regimes
- **Neutral** — pre-grasp states; both instructions achievable, no visual prior favours either.
- **Conflict** — post-grasp states; arm already carrying the object toward the demonstrated
  goal, so the observation opposes the alternative command. Destination swaps only (with the
  object held, an object swap is a different task, not a redirection).
- **Compositional** — trained vs never-demonstrated pairings of familiar words. All ten
  LIBERO-Goal scenes contain the same seven objects, so novel pairings are executable.

### 3.3 Readouts
- **Directional IFR** (regimes 1–2): does commanding the non-demonstrated instruction push the
  predicted chunk away from the demonstrated trajectory? Chance 0.5.
- **Destination-direction accuracy** (regime 3): novel commands have no demonstration, so we
  score which *destination anchor* the predicted net displacement points toward. Anchors are the
  mean final end-effector position of demos ending at each destination (sd 0.03–0.05 m,
  separation 0.19–0.57 m; the cabinet anchor agrees within 0.07 m across two different objects,
  so it tracks the destination not the task). Cosine, so controller action-scaling is irrelevant.

### 3.4 Controls
- **Competence gate.** ‖prediction − demonstration‖ ÷ distance between two *unrelated*
  demonstrations. A chance-level grounding score means "does not ground language" only if the
  policy can do the task; otherwise it means the metric is reading noise.
- **Determinism.** Flow-matching noise pinned; identical inputs give bit-identical actions
  (control passes at exactly 0.0).
- **Trained-pairing control** for the compositional readout: must beat chance or the readout is
  broken.
- **Fixed-state control.** Every state scored under both object words, so the observation is
  byte-identical and the comparison is paired.

**Figure 1** — one contrastive pair (observation + two instructions + two predicted
trajectories overlaid on destination anchors). Carries the whole method visually. `[TODO: make]`

*Expansion to 8 pp:* full gate pseudocode, anchor-validation table.

---

## 4. Experiments (1 pp short / 2–3 pp long)

**Setup.** SmolVLA-450M, LIBERO-Goal, offline on stored demonstration states. Two competent
checkpoints (`k1000dai/smolvla_libero_finetune`, `…_scratch_80k`). Paired bootstrap, 10k
resamples.

**Table 1 — checkpoint screen.** Only 2 of 8 public checkpoints are competent; four are worse
than predicting an unrelated trajectory, at every image orientation. *This is a result, not
housekeeping* — it bears on how reproducible this literature is.

**Table 2 — the three regimes.**

| regime | readout | result |
|---|---|---|
| neutral (n=400) | directional IFR | 0.745 [0.703, 0.788] — follows |
| conflict (n=240) | directional IFR | 0.946 / 0.929 — follows |
| compositional (n=800, paired) | destination accuracy | gap **+0.250 [+0.197, +0.300]** / **+0.193 [+0.143, +0.243]** |

**Table 3 — compositional, per destination** (six cells, all positive and significant):

| destination | finetune | scratch_80k |
|---|---|---|
| the plate | +0.100 [+0.030, +0.180] | +0.180 [+0.100, +0.270] |
| the stove | +0.330 [+0.240, +0.420] | +0.140 [+0.060, +0.220] |
| the rack | +0.320 [+0.230, +0.410] | +0.260 [+0.170, +0.350] |

**The error analysis is the mechanism claim.** 93–96% of novel-command errors go to a
destination that object *was* trained with. The policy is confident and wrong in a structured
way — substitution, not confusion.

**Negative controls we report because they nearly fooled us.** The apparent decline at high
visual conflict is a metric artefact: sensitivity collapses in that bin, and where the
instruction cannot move the action its direction is estimated from noise. Conditioning on
sensitivity removes it. Generalises: *a site where language cannot matter looks identical to a
site where it is ignored.*

*Expansion to 8 pp:* per-object breakdown; conflict-strength binning; the four artefacts as a
short subsection with before/after numbers.

---

## 5. Conclusion (1 para short / 0.5 pp long)

Grounding in this VLA does not fail globally — it fails at composition. Under direct
visual-motor conflict the policy still redirects on command; under an undemonstrated pairing of
familiar words it substitutes a memorised destination. That localises the behavioural target
for mechanistic work: not "where does the model stop attending to language" but "where does the
object↔destination binding fail to form or fail to be read."

**Limitations,** stated not buried: effect sizes are moderate and shrank by half under the
strictest control; both competent checkpoints come from one uploader, so replication is weak;
LIBERO-Goal supports one object contrast, so results generalise over states rather than
referents; evaluation is offline.

**Next.** Causal localization over layer × position × component on the trained-vs-novel
contrast, which has a matched control built in, and a binding-transplant test to separate
encoding from readout failure. `[TODO — requires GPU; no results yet]`

---

## Open questions for you

1. **Do we expect GPU access before ~Aug 20?** This is the fork. Yes → 8-page VLM4RWD with a
   localization section. No → 4-page RoboPAD, behavioural + methods, and it stands on its own.
2. **Author list and affiliations**, and whether Algoverse has a preferred acknowledgement.
3. **NeurIPS 2026 template** — yes please, upload it; I will format directly rather than
   converting late. RoboPAD's template requirement was not stated on its page and needs
   checking against its CFP.
4. **Is anyone else in the group working on adjacent claims?** Two of these workshops are
   non-archival, so overlapping submissions are allowed, but we should not collide internally.
