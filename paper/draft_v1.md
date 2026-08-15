# Paper A — draft v2

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
> artifact rather than the map. A linear probe instead recovers the named destination from the
> action expert at 0.95–0.99 on exactly the pairings whose actions go elsewhere, against a
> label-shuffled control at chance, indicating a readout rather than an encoding failure. We release the stimulus
> generator, a checkpoint competence gate, and the controls that caught five false positives in
> our own pipeline.

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
unrelated trajectory*, at every image orientation. A result, not housekeeping.

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

### 4.6 Probing: the destination reaches the expert and is not used  ← THE MECHANISM

Patching could not localize the failure, so we asked the question a different way. A linear
probe is trained on **trained** pairings and tested on **novel** ones, with a label-shuffled
control at every site and a **state-grouped split** (all examples from one observation go
entirely to train or to test).

Reading the destination out of the action expert's residual stream, for the very pairings
whose actions go elsewhere (n=150 states, 1200 examples, chance 0.25):

| expert layer | 0–2 | 3–4 | 5–6 | **7–8** | 9–12 | 15 |
|---|---|---|---|---|---|---|
| acc (novel) | 0.27–0.35 | 0.67 | 0.24–0.25 | **0.95–0.96** | 0.75–0.79 | 0.65 |
| shuffled | 0.27–0.32 | 0.28–0.30 | 0.36 | 0.24–0.29 | 0.27–0.32 | 0.33 |

Best clean site: `expert.L7.attn_out`, **0.985** novel against a shuffled control of 0.293.
124 of 130 sites have controls within 0.10 of chance.

**Verdict: readout failure**, replicated on both competent checkpoints. The named destination
is absent from the expert's earliest layers, becomes strongly decodable by layer 7 — consistent
with cross-attention pulling it in from the VLM — and remains decodable to the output, while
the action goes somewhere else entirely. The information is present and unused, not missing.

This is the opposite of what the transplant reported before its confound was found, which is
precisely why the standing rule asks for two techniques with different failure modes. Probing
is correlational and so immune to the causal-proximity artifact that defeated the sweep.

**Scope of the claim, and two measurement notes.**
*(i)* The probe decodes the named *destination*, which is present in the input tokens; high
accuracy in the VLM is therefore expected and is not the finding. The findings are that it
survives into the **expert** and that decodability is near-identical for trained and novel
pairings despite behaviour differing sharply. We have **not** shown the object↔destination
*binding* is represented — only that the destination the action should follow is available
where the action is computed.
*(ii)* Activations are mean-pooled over all positions, so language signal is diluted in the
residual stream (dominated by ~1000 image tokens) and concentrated in attention outputs. This
is why `attn_out` sites read higher than `resid_post` at the same layer, and it is a reason to
prefer position-resolved probing in follow-up work.
*(iii)* An earlier version of this probe reported perfect held-out accuracy at **every** site,
including one whose activation we verified is byte-identical across instructions. The cause was
an example-level rather than state-level split. We report it because the failure is invisible
without a site that provably cannot carry the signal.

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
| Interpretability for Discovery | 5 | **conditional** | welcomes "failure cases and negative results", so §4.5 qualifies as a methods contribution — but the interpretability payload is currently a null. Stronger if the probe experiment lands before Aug 29. |

All three are **non-archival**, so simultaneous submission is permitted and none blocks a later
archival version. Verify each venue's dual-submission line before relying on it.

**Recommendation:** submit to RoboPAD and VLM4RWD regardless. Add I4D only if the probe result
arrives in time — otherwise it is a paper about interpretability that reports no interpretability
finding, which reviewers there will notice.
