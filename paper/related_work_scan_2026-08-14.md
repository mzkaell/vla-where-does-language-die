# Concurrent-work scan, 2026-08-14

Web scan for concurrent/overlapping work, Feb–Aug 2026. Every claim below was
checked against the arXiv page or repo it cites; the two things that could not
be verified are listed at the bottom. Bottom line first.

## Verdict

**The novelty claim needs a reframe.** "First inside-the-model causal account of
instruction grounding in VLAs" is no longer defensible: arXiv 2603.19233 (ICLR
2026) already runs causal activation injection on SmolVLA and OpenVLA-OFT and
already states a probe-based encoded-but-not-used result. The behavioural
compositional finding is also independently established (three times, once on
SmolVLA specifically).

**What remains open, and no paper occupies it:** a component-level causal
localization of **where compositional object–destination binding fails**, with
causal (not probe-based) evidence for the encoding-vs-readout distinction —
i.e. exactly M2 + M3. 2603.19233 names this as its own future work, so the
window is real but closing.

Recommended reframe: *"first component-level causal account of compositional
binding failure in VLAs"*, citing 2603.19233 and 2602.24143 as concurrent work.

## Direct competitors

- **2603.19233 — "Not All Features Are Created Equal" (ICLR 2026, March).**
  The big one. Six models incl. OpenVLA-OFT and SmolVLA; activation injection,
  counterfactual prompting, per-token SAEs, linear probes, 394k rollouts.
  Headline: fine-tuned VLAs are "effectively vision-only policies with
  vestigial language pathways"; a layer-17 probe decodes the prompt at 99.3%
  while behaviour ignores it. Verified gaps: no systematic layer/component
  localization of where language influence is lost, and the paper itself lists
  "compositional instructions" as a limitation. Cite as concurrent work; the
  probe result is correlational where ours is causal.
- **2602.24143 — "Robust Skills, Brittle Grounding" (Feb 27).** Held-out
  object–location pairings on **SmolVLA** and pi-0.5; concludes benchmark
  success reflects "object-location correlations rather than genuine language
  grounding." Purely behavioural, no internals. This scoops the M0 behavioural
  headline on our own model. Paper A must cite it and lead with what it does
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
