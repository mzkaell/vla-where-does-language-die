# Project briefing — talking script

*~6 minutes spoken. Every number here traces to a committed run in `results/`.
Square brackets are 95% bootstrap CIs.*

---

## 1. The question (30 s)

Vision-language-action models take a camera image plus a language instruction and output
robot actions. There's a growing literature saying they under-use the language — they act on
what the scene looks like and ignore the words.

Our project was supposed to find *where inside the network* that happens. But before
localizing a failure you have to reliably produce one, so the first job was reproducing the
behavioural effect on our own model and stimuli.

**Setup:** SmolVLA-450M on LIBERO-Goal, evaluated offline on stored demonstration states.
The method throughout is **minimal contrastive pairs** — the same observation, two
instructions differing in exactly one word. Because the pixels are byte-identical, any change
in the predicted action is attributable to that word.

---

## 2. What we found (2 min)

### The headline: grounding fails at *composition*, not in general

We tested three conditions of increasing difficulty.

| condition | what it does | result |
|---|---|---|
| **Neutral** | robot hasn't touched anything; swap one word | **0.745** [0.703, 0.788], n=400 — follows |
| **Conflict** | robot already carrying the object toward goal A; command goal B | **0.946** / **0.929**, n=240 — follows |
| **Compositional** | a word combination never demonstrated in training | **fails** |

The first two are the conditions this failure is usually attributed to — and the model
handles them. Under direct visual-motor conflict it *redirects mid-trajectory on command*,
94% of the time.

It breaks only when the instruction requires composing familiar words into a pairing it never
saw. It learned bowls go to plates and stoves; bottles go to racks. Ask for **"put the bowl
on the rack"** and it fails.

Under our strictest control — identical observation, only the object word changes, paired —
accuracy drops from **0.698 → 0.340** and **0.678 → 0.353** on the two checkpoints.
That's a gap of **+0.250** [0.197, 0.300] and **+0.193** [0.143, 0.243], significant in all
six destination × checkpoint cells.

### The mechanism is substitution, not confusion

**93–96% of the errors send the arm to a destination that object *was* trained with.** It
doesn't wander or freeze — it confidently executes a memorised pairing. That's an
object↔destination *binding* failure, not a failure to attend to language.

### A methodological result: only 2 of 8 public checkpoints work at all

We built a competence gate before measuring anything. Four public LIBERO SmolVLA checkpoints
score *worse than predicting an unrelated trajectory* — at every image orientation. One won't
load. That bears on how reproducible this literature is.

---

## 3. What we could NOT show (1.5 min) — say this plainly

### The localization sweep is null

We patched all 130 sites and got a beautiful layer profile. Then we ran the same sweep on a
**control** where both instructions were trained, so there was no failure to recover — and
the same profile appeared. It was measuring distance-from-output, not mechanism.
**0 of 130 sites survive multiple-comparison correction.**

### The probe does not replicate

A linear probe shows the instruction is encoded and survives the vision-language backbone.
But inside the action expert the two checkpoints disagree completely: one has 37 sites
decoding above 0.5, the other has 2 — with 29 sites *below* 0.10 against a chance level of
0.25, meaning reliably wrong rather than uninformative. **We assert no readout-vs-encoding
verdict.**

### Five retractions, and why that's the contribution

Five results were withdrawn during this project: a one-sided stimulus set, 180°-rotated input
images, a broken reference metric, a leaking probe split, and the transplant confound. **Four
of the five produced clean, significant, entirely plausible numbers.** Every one was caught
by a check capable of *disagreeing* — a second checkpoint, a matched control, a shuffled
baseline, a site that provably cannot carry signal. None was caught by reading code.

That's the most transferable thing here, and it's a section of the paper.

---

## 4. Where the paper is (1 min)

**Paper A: behavioural finding + methodology.** Written, in LaTeX, with two generated figures.

- Target: **RoboPAD** short track, 4 pp (Aug 29) and **VLM4RWD**, 8 pp (Aug 30). Both
  non-archival, so we can submit to both.
- RoboPAD's short track explicitly solicits negative results, which is this paper's shape.
- Interpretability-for-Discovery is a weak fit now — the interpretability payload is a null.

**Remaining before submission:** build the PDF and confirm the page count, verify six
citation fields, author list.

**Not claimed:** any mechanistic localization. We say where grounding fails behaviourally,
not where it fails inside the network.

---

## 5. What's next / asks (1 min)

**Coordination.** Another team member has a parallel branch with a full M2 sweep, M3
transplants, and — importantly — **language-token-position patching**, which is better than
what we had and avoids a confound that invalidated our transplant. His runs are missing the
two controls we learned the hard way: no proximity control on the sweep, no random-direction
control on the transplant. His transplant flips sign with dose (−0.87, −0.80, +0.59), which
is not a dose-response curve.

**His targeting plus our controls is the one experiment worth running** — roughly 2 GPU-hours,
and it could produce a valid mechanism result.

**Asks:**
1. A view on whether to submit Paper A as-is or hold for the merged M3.
2. Whether a genuinely independent checkpoint exists — both of ours come from one uploader,
   which is our weakest claim.
3. GPU access for ~4 hours if we go after the mechanism.
