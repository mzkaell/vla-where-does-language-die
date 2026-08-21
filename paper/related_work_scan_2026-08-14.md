# Concurrent-work scan, 2026-08-14

Web scan for concurrent/overlapping work, Feb–Aug 2026, followed by an
independent second-pass verification that re-fetched the two load-bearing
sources and corrected two characterizations (marked below). Things that could
not be verified are listed at the bottom. Bottom line first.

## Verdict

**The novelty claim needs a reframe.** "First inside-the-model causal account of
instruction grounding in VLAs" is no longer defensible: arXiv 2603.19233
(Multimodal Intelligence Workshop @ ICLR 2026) already runs causal activation
injection on SmolVLA and OpenVLA-OFT and already states a probe-based
encoded-but-not-used result (on pi-0.5). The behavioural compositional finding
is also independently established (three times, once on SmolVLA specifically).

**What remains open, and no paper occupies it:** a component-level causal
localization of **where compositional object–destination binding fails**, with
causal (not probe-based) evidence for the encoding-vs-readout distinction —
i.e. exactly M2 + M3. 2603.19233 names this as its own future work, so the
window is real but closing.

Recommended reframe: *"first component-level causal account of compositional
binding failure in VLAs"*, citing 2603.19233 and 2602.24143 as concurrent work.

## Direct competitors

- **2603.19233 — "Not All Features Are Created Equal" (Multimodal Intelligence
  Workshop @ ICLR 2026, March).** *Two corrections from the verification pass:
  it is a workshop paper, not ICLR main; and its headline is NOT "VLAs are
  vision-only" — it is that language sensitivity depends on task structure
  (language ignored when vision uniquely specifies the task, essential in
  multi-goal scenes; libero_goal collapses 94%→10% under wrong prompts).* Six
  models incl. OpenVLA-OFT and SmolVLA; activation injection, counterfactual
  prompting, per-token SAEs, linear probes. The layer-17 probe decoding the
  prompt at 99.3% while behaviour ignores it is real but **pi-0.5-specific**.
  Verified gaps stand: no layer/component localization of where *language*
  influence is lost (its layer interventions act on visual/full-pathway
  activations), and its limitations section names compositional instructions
  as future work. Cite as concurrent work with the task-structure conditional
  intact; the probe result is correlational where ours is causal.
- **2602.24143 — "Robust Skills, Brittle Grounding" (Feb 27, preprint, no
  venue).** Held-out object–location pairings on **SmolVLA** and pi-0.5;
  success reflects "object–location correlations that do not transfer beyond
  the training distribution" (verified quote; the paper phrases it as "can
  mask", not an absolute claim). Purely behavioural, no internals — verified,
  the authors call it "deliberately diagnostic rather than prescriptive". This
  scoops the M0 behavioural headline on our own model. Paper A must cite it and lead with what it does
  not have: the vision-override null (C2), the error-substitution signature
  (93–96%), the strict pixel-identical control, and the causal follow-through.
- **2607.21582 — "Scale Up Strategically" (NVIDIA/Northeastern, July).**
  Re-pairs instruction factors across six models; finds memorized factor
  associations. Behavioural + data-collection mitigation. Same citation
  treatment as above.

## Adjacent

- 2505.03500 v5 (May 2026): "spatial overfitting"; libero-ood; intervenes on a
  pooled text latent — mitigation-grade internals, no localization.
- 2604.09824 ProGAL-VLA: frames the problem as binding; mitigation
  architecture, no diagnosis.
- 2603.06001 "linguistic blindness": attention-weight analysis, correlational.
- 2605.00321 Embodied Interpretability (ICML 2026): causal masking of visual
  regions, not language pathways.
- 2604.09364 and 2606.28273: activation patching on VLMs for vision-language
  conflict — notably the framing our C2 null argues against.
- Binding theory (useful citations): 2602.24264, 2606.03976, 2605.31503,
  2411.00238.

## Artifact status (checked directly)

- **LIBERO-CF** (2602.17659, "When Vision Overrides Language"): code released
  at github.com/yuffish/LIBERO-CF — five counterfactual suites, needs only
  standard LIBERO. Its title is the exact framing our C2 null pushes back on;
  its CAG mitigation is a baseline to beat.
- **BeTTER** (2604.18000): paper real, **no code/data found** — dependency risk.
- **RoboSemanticBench** (2606.02277): repo exists (ZGC-EmbodyAI) but partial —
  only RSB-Math-4/10 on HF, more "released gradually."

## Independent-checkpoint option (confirmed)

Official OpenVLA LIBERO checkpoints exist on huggingface.co/openvla:
`-libero-spatial`, `-libero-object`, `-libero-goal`, `-libero-10`. This is the
cheapest purchase of external validity available (both competent SmolVLA
checkpoints come from one uploader).

## Not verified

- OpenReview reviews for 2603.19233 (Cloudflare-blocked).
- X/Twitter chatter (no direct search access; nothing surfaced via web).

## Appendix: known structural debt (review pass, 2026-08-14)

Two review findings are deferred deliberately, not forgotten. (1) run_localization
and run_transplant share a ~55-line per-trial scaffold by copy; their comparability
is enforced only by the copies staying in sync, and the right fix is one shared
trial-builder helper. Deferred because the first real M2 sweep and M3 run are in
flight on the current code; refactor before the next sweep, not during this one.
(2) The resolved-config writer exists in four copies across scripts/. Same timing
argument. (3) CLAUDE.md §10's target CLI takes --pairs; both runners instead use
the three hardcoded CONTRASTS — a team decision to revisit, since the M3 verdict is
currently measured on 3 contrasts, not the M1 stimulus set.
