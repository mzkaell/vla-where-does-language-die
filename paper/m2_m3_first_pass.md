# M2 + M3 first pass — 2026-08-14

First real localization and transplant data (`loc_full_mps`, `m3_*`). One
checkpoint (`k1000dai/smolvla_libero_finetune`), 3 contrasts, offline readout.
Interpretation, then the caveats that bound it.

## What the data says

1. **The instruction's causal influence dies gradually inside the VLM.**
   Full-stream patching recovers 0.73 of the gap at vlm.L0, ~0.55 through the
   middle, 0.16 by L13, 0.00 at L15. No bottleneck: the top 10% of sites carry
   30% of the effect (the "narrow" criterion needs 70%).

2. **The transplantable binding lives in the language-token block.** At vlm.L4,
   injecting only the language positions' delta recovers +0.594 vs +0.617 for
   the whole prefix — the image and state positions add ~0.02.

3. **It decays with depth in reinjectable form too.** Lang-block transplant at
   alpha=1: +0.741 at L0 (CI fully above the 0.5 rule), +0.594 at L4
   (straddles), +0.372 at L12 (CI fully below). Mirrors the M2 profile.

4. **Partial doses actively hurt.** alpha 0.25/0.5 at L4 recover −0.87/−0.80 —
   worse than no patch at all. The stream between the two instructions' states
   is not an interpolation path; behaviourally it reads like sitting between
   two memorized attractors. This is the most unexpected number of the day.

## Against the central hypothesis, as stated

CLAUDE.md §3 predicts a readout failure at the VLM→expert interface: the
backbone forms the binding, the expert fails to read it. First-pass evidence
points the other way: by late VLM the binding is no longer present in a form
whose reinjection helps (L12 transplant fails its own rule), which is the
signature of **progressive encoding loss inside the VLM**, not of intact
information the expert ignores. The expert-side monotonic rise to 1.0 is
structural (late sites sit downstream of everything) and supports neither side.

## Caveats that bound all of the above

- **Nothing clears the position-shuffled null after BH** in M2 (min p 0.051 at
  n=78). The graded profile is consistent and large, but per-site significance
  claims are not yet available. More trials would settle it.
- **Tap dilution confound**: the expert reads the VLM through per-layer KV, so
  patching resid_post(i) reaches only KVs i+1..15. Part of the depth decay is
  mechanical. A KV-targeted patch would deconfound; not built yet.
- **L0 "readout" verdict is a sanity anchor, not a discovery** — injecting the
  lang delta at L0 is close to swapping the instruction embedding.
- One checkpoint, 3 contrasts, offline cosine readout, MPS backend (resid sites
  agree with CPU within 0.014; attn/mlp up to 0.21 — the profile above is a
  resid-site story, which is the agreeing regime).
- Replication on the second checkpoint is running; nothing here is replicated
  yet.

## What this buys the paper

The reframed claim ("first component-level causal account of compositional
binding failure in VLAs") now has its two load-bearing figures: the layer ×
component map with its graded VLM decay, and the transplant dose/depth curves
localizing the binding to the language block and timing its disappearance.
The encoding-vs-readout verdict, stated honestly: *encoding-side, progressive,
with the interface exonerated at first pass.*
