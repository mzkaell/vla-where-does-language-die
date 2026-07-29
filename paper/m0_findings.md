# M0 findings: a referent-type dissociation in SmolVLA's instruction grounding

*Interim write-up. Assumes familiarity with VLAs and activation patching, but not with this
project. All numbers come from `results/m0_pilot_k1000dai/`; the resolved run config is
committed next to them.*

---

## The question

VLAs are known to under-use their language input — they often act on visual priors and
ignore the instruction. That much is established behaviorally. This project asks the
mechanistic follow-up: **where inside the network does instruction identity stop causally
influencing the action?** M0 is the prerequisite step: establish, on our own stimuli and
model, that there is a grounding failure worth localizing at all.

## Setup

**Model.** SmolVLA-450M (`lerobot/smolvla_base` architecture), specifically the
LIBERO-finetuned checkpoint `k1000dai/smolvla_libero_finetune`. It is a flow-matching
policy: a SmolVLM-2 backbone (SigLIP vision + SmolLM2) encodes image + instruction once
into a KV cache, then a 16-layer action expert attends to that cache once per denoising
step (10 steps) to emit a 50-step action chunk. 8-dim state in, 7-dim actions out.

Two properties make it a good subject. It is small enough to patch exhaustively on a
laptop CPU, and the VLM→expert boundary — the place our central hypothesis says grounding
fails — is a literal object in the forward pass (`past_key_values`) rather than an
abstraction we have to define.

**Benchmark.** LIBERO-Goal. Chosen because all ten of its tasks share one kitchen scene
with a fixed object set, and only the commanded goal varies. Vision therefore
underdetermines the target and the instruction *must* be read. LIBERO-Object and -Spatial
change the scene along with the task, which would confound a word swap with a visual
difference.

**Stimuli.** 200 minimal contrastive pairs, built offline from the released demonstration
HDF5s. A pair is one visual state plus two instructions differing in exactly one referent:

```
photo:          demo_18, frame 0            <- byte-identical across both arms
instruction A:  put the bowl on the plate
instruction B:  put the bowl on the stove
```

Because the observation is identical, any difference in the predicted action is
attributable to the swapped span. Two families:

| Family | n | Example swap |
|---|---|---|
| destination | 160 | `put the bowl on the PLATE` → `... on the STOVE` |
| object | 40 | `put the BOWL on the cabinet` → `put the WINE BOTTLE on the cabinet` |

States are drawn only from **pre-grasp** timesteps (before the first gripper-close
command), so nothing has been manipulated yet and both instructions remain achievable —
the operational form of "both actions individually valid." Minimality is enforced by a
word-level diff; the gate and its two non-obvious failure modes are documented in
[`src/data/README.md`](../src/data/README.md).

**Readouts.** Two, deliberately, because they fail differently:

- **Instruction sensitivity** — `‖a_A − a_B‖` on identical pixels. Asks only whether the
  instruction changes the action at all. A language-ignoring policy scores exactly 0, so
  it cannot be faked. But it is direction-blind: reacting *incorrectly* still scores high.
- **Directional IFR** — the state comes from a demonstration of one of the two tasks, so
  that demo's action is the visually-consistent behavior. Commanding the *other*
  instruction should push the predicted action away from it. This has direction, but
  depends on the demo action being a fair reference (see Threats).

Both arms are run with **identical flow-matching noise**, so the instruction is the only
difference between them. Aggregation is a paired bootstrap over pairs (10k resamples).

---

## Results

| Readout | All (n=200) | Destination (n=160) | Object (n=40) |
|---|---|---|---|
| Instruction sensitivity | 7.98 [7.61, 8.35] | 8.29 [7.85, 8.73] | 6.72 [6.30, 7.21] |
| Directional IFR (chance 0.5) | 0.885 [0.84, 0.93] | **0.994 [0.98, 1.00]** | **0.45 [0.30, 0.60]** |

Aggregate IFR is above chance at p ≈ 1e-4 — but the aggregate is dominated by the 160
destination pairs and **should not be quoted alone.** The finding is the split.

**The policy tracks a swapped destination almost perfectly (99.4%) and is at chance on a
swapped object (45%, CI spanning 0.5) — while still changing its action substantially in
the object condition (sensitivity 6.72, comfortably non-zero).**

## Interpretation

The object-swap condition is not a case of the model ignoring language. Sensitivity is
high there: swapping `bowl` → `wine bottle` measurably moves the action. What is absent is
*correspondence* between the new word and the behavior. The model registers a lexical
change and then acts in a way unrelated to the new referent.

That is a binding failure rather than an attention failure, and it is precisely the
phenomenon the project set out to localize. It also arrives in a more tractable form than
expected: rather than a single failure rate to explain, we have a **within-model
dissociation**. Same network, same scene, same measurement, same pre-grasp states — one
class of referent succeeds and another fails.

This matters methodologically. A localization sweep over a single failing condition has to
distinguish "circuitry that carries instruction information" from "circuitry that carries
*this* instruction information correctly," which is hard. With a dissociation, M2 gets a
built-in control: trace the destination pathway (which works) against the object pathway
(which does not) through the same 130 tap points, and look for where they diverge.

The direction of the split is also consistent with the binding-ID literature: destination
words may be recoverable as a relatively direct goal-location signal, whereas selecting
which of several visible objects a noun refers to requires an object↔word binding — the
operation CLAUDE.md §3 predicts the action expert fails to read.

---

## Threats to validity

Stated plainly, because they gate what can be claimed.

**1. The directional reference is weak.** The readout compares a 50-step predicted action
chunk against a *single* demonstration action broadcast across all 50 steps. Both families
are scored identically, so the dissociation is unlikely to be an artifact of it, but the
absolute values (0.994, 0.45) should not be reported until the reference is the
demonstration's true next-50-step chunk. **This is the top-priority fix.**

**2. The object arm is thin.** n=40, from a single instruction pairing
(`bowl` ↔ `wine bottle`). Its interval spans chance, which supports "not distinguishable
from chance" but not "is chance."

**3. Single checkpoint of unofficial provenance.** No official LIBERO-finetuned SmolVLA
exists. This is a community upload of unknown training quality. Under the project's
standing rule — no headline claim on a single technique — the dissociation must reproduce
on a second checkpoint before it counts.

**4. Offline, not closed-loop.** These are action predictions on fixed stored states, not
rollouts. That is deliberate (exactly reproducible, cheap, and CLAUDE.md §8 reserves
closed-loop for the final shortlist), but it means "does the right thing" is a statement
about predicted actions, not task success.

**5. No same-instruction control has been run.** Running instruction A against itself
should yield exactly 0 divergence under fixed noise. It is a cheap check that would prove
no nondeterminism is leaking into the measurement.

---

## What still needs to be run

### Immediate — validity, before any claim ships (~1–2 hours, CPU)

1. **Fix the directional reference** to the demonstration's true next-50-step action chunk,
   then re-run all 200 pairs. Expect absolute values to move; the dissociation should not.
2. **Add the same-instruction control.** Must be exactly 0.
3. **Expand the object arm.** Download the four remaining LIBERO-Goal tasks and rebuild;
   target n ≥ 100 object pairs so its interval can actually exclude chance.
4. **Replicate on a second checkpoint** (`msv6/smolvla_meta_libero` has the same schema).
   If the dissociation does not reproduce, it is a property of one upload, not of SmolVLA.

### M2 — causal localization (needs a GPU)

The infrastructure is built and tested: 130 tap points across
`{vlm, expert} × L0–L15 × {resid_pre, attn_out, mlp_out, resid_post}`, with
patching correctness verified by self-patch identity plus perturbed negative controls.

5. **Activation patching over (layer × position × component)**, run *separately for the two
   families*. The primary contrast is no longer "where does instruction info live" but
   **where does the object pathway diverge from the destination pathway** — the
   dissociation supplies the control condition.
6. **Position-resolved patching.** The prefix is laid out as
   `[image tokens][language tokens][state]`, so language-token positions can be patched
   independently of vision positions. This is what separates "the instruction was never
   encoded" from "it was encoded but not read."
7. **Significance**: permutation null over shuffled positions, Benjamini-Hochberg FDR
   across sites (implemented and tested in `src/eval/stats.py`). Report the fraction of the
   clean-vs-failing gap the identified sites explain, plus localization sharpness.
8. **Path patching** at head level for the shortlist, to get a circuit rather than a
   heatmap.

Cost note: 130 sites × 200 pairs × 2 arms is infeasible on CPU at the observed ~11 s/pair.
This is where the remote GPU decision becomes real.

### M3 — encoding vs readout

9. **Split language-token vs expert-input patching** to localize the failure to one side of
   the VLM→expert interface.
10. **Binding transplant**: extract the object↔word binding direction (difference-of-means
    between the two runs at the binding site) and inject it into the failing run's expert
    input. Verdict is *readout failure* if recovery ≥50% of the clean-minus-failing gap,
    *encoding failure* if not.

The object-swap condition is the natural target for this, since that is where binding
demonstrably fails while the destination condition provides a working comparison.

### Cross-checks required by the standing rule

11. Corroborate any headline localization with a **second technique with different failure
    modes** — SAE-feature patching, or knockout against patching. Patching can push
    activations off-distribution; knockout can miss information that reroutes.

---

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[vla,dev]"
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/build_pairs.py --suite libero_goal --n 200
.venv/bin/python scripts/reproduce_ifr.py --checkpoint k1000dai/smolvla_libero_finetune
```

~30 min for the M0 run on 8 CPU cores. See the README for the Windows paths and
troubleshooting. 55 tests cover patching correctness, the statistics, and the stimulus
validity gates.
