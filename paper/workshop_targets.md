# Workshop targets — NeurIPS 2026

Researched 2026-08-02. **Every relevant deadline is ~27 days out.** All three are
**non-archival**, so submitting to more than one is normally permitted and none blocks a
later archival version — verify each workshop's dual-submission line before relying on it.

| workshop | deadline | limit | format | fit |
|---|---|---|---|---|
| **[VLM4RWD](https://vlm4rwd.github.io/)** — VLMs for Real-World Deployment | **Aug 30** | **8 pp** excl. refs/appendix | NeurIPS 2026 | **best** |
| **[RoboPAD](https://robotpad2026.github.io/)** — robot FMs: adaptation, reasoning, evaluation | **Aug 29** | **4 pp** short / 9 pp long | (check CFP) | **best for short** |
| [Interpretability for Discovery](https://interpretability4discovery.github.io/) | Aug 29 | 5 pp (6 camera-ready) | NeurIPS 2026 workshop template | good *if* M2 lands |

Missed this cycle: ICML 2026 Mech Interp (was May 8, held July 10), CoRL 2026 main track
(May 28). CoRL workshops are Nov 11–12 in Austin; individual CFPs were not published at time
of writing and are worth re-checking — that is the natural fallback if NeurIPS slips.

## Why these three

**VLM4RWD** is the closest match on topic. Its scope list explicitly names *visual grounding*,
*compositional reasoning*, *benchmarks for grounding evaluation*, *embodied AI*, and
*interpretability* — which is this paper's exact intersection. 8 pages is enough for the full
story including a localization section if M2 lands.

**RoboPAD** is the best home for the version we can write *today*. Its short track explicitly
solicits "preliminary findings, positions, benchmarks, and **negative results**" — and our two
null conditions are load-bearing evidence, not filler. 4 pages forces the tight version.

**Interpretability for Discovery** only makes sense with M2 in hand. Right now our
interpretability contribution is verified infrastructure, not results, and a workshop framed
around *using interpretability to discover things* would reasonably expect discoveries.

## The strategic question

**We do not have M2 or M3.** They need a GPU. So there are two different papers:

**Paper A — behavioural + methods (writable now).** Claim: VLA instruction-grounding failure
is *compositional*, not general. Evidence: the compositional gap under the fixed-state control,
replicated on both competent checkpoints, plus two nulls showing the failure does not appear
under neutral or visually-conflicting conditions. Deliverables: released stimulus generator, a
competence gate, and the measurement pitfalls. This is a complete, honest 4–5 page paper.

**Paper B — the full localization story.** Paper A plus the layer × component causal map and
the encoding-vs-readout verdict. Needs GPU time *and* analysis time inside 27 days.

Recommendation: **write Paper A now** and target RoboPAD-short (4 pp). If a GPU appears in the
next ~10 days, the same text expands into VLM4RWD (8 pp) with a localization section. If it
does not, submit A to both — VLM4RWD accepts "extended abstracts, position papers, datasets,
benchmarks, and emerging research ideas," so a behavioural + benchmark paper is in scope there
too.

Writing Paper A first is not wasted work under either branch: it is the front half of Paper B.

## Update 2026-08-14 — two premises above are now dead

**The GPU premise is dead.** MPS matches CPU within 0.02 on the M2 smoke and runs 24× faster
(see `results/_loc_smoke_mps`); the full 130-site, n=40 sweep is a ~3 h laptop job. Paper B's
"needs GPU time" branch collapses into "needs analysis time." M3 machinery is implemented and
smoke-tested (`scripts/run_transplant.py`); it needs M2's site name and hours, not hardware.

**The novelty premise moved.** The concurrent-work scan
(`related_work_scan_2026-08-14.md`) found the behavioural compositional finding independently
established three times (2602.24143 on SmolVLA itself, a preprint), and 2603.19233 (an ICLR
2026 *workshop* paper, verified — not ICLR main) already running causal activation injection
on our two models. Paper A can no longer lead with "VLAs
fail compositionally" as a discovery; it leads with what the others lack — the vision-override
null, the 93–96% substitution signature, the pixel-identical control, the competence gate —
and cites 2602.24143 as concurrent. Paper B's claim narrows to the still-unoccupied
intersection: *component-level causal localization of the binding failure, with a causal
encoding-vs-readout verdict.* 2603.19233 names exactly this as future work, so speed matters.

Net effect: the A-vs-B decision tilts toward B. A alone is now a replication-plus-controls
paper; B is the paper nobody else has.
