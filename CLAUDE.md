CLAUDE.md — Where Does Language Die?

Persistent context for this repo. Read this first every session.

1. Project

Causal, component-level localization of instruction-grounding failure in Vision-Language-Action models (VLAs). VLAs map a camera image + a language instruction to robot actions, but they frequently ignore the instruction and act on visual priors. Prior work shows that this happens (behaviorally); we find where inside the model the instruction stops influencing the action, and why.

Audience / venue: a NeurIPS or ICML workshop (mechanistic interpretability or robot-learning track). This means a focused, clean, reproducible result in ~4–8 pages, not a full-length paper. Optimize for one decisive finding + released code/stimuli, not breadth.
Team / timeline: Algoverse research group, ~8–10 weeks, single-GPU budget.
Guiding standard: if the three research questions below are answered, the work is complete and novel.
2. Research questions
RQ1 (Localization). At which layer, token position, and component (attention vs MLP; language-token vs vision-token positions; VLM backbone vs action expert) does instruction identity stop causally influencing the action?
RQ2 (Mechanism). Is the failure an encoding failure (instruction semantics never written into the stream feeding the action head) or a readout failure (present but not read)? Decided by the binding-transplant test.
RQ3 (stretch, only if RQ1–2 land). Does the mechanistic signature predict which mitigation restores grounding?
3. Central hypothesis

Readout-dominated failure localized at the VLM → action-expert interface: the backbone forms a correct object↔word binding that the action expert fails to read when vision conflicts with the instruction. Transplanting the correct binding into the expert's input should recover grounding.

4. Definition of done (workshop MVP)

Ship these, in order. Do not start a later item before the earlier one is solid.

 M0 Repro: measure the instruction-following-under-contradiction gap on SmolVLA in LIBERO (establish the behavioral effect we will localize).
 M1 Contrastive-pair generator + a released stimulus set (same scene, one swapped referent; both actions individually valid).
 M2 Causal localization map (layer × component) via activation/path patching, with a permutation null and Benjamini-Hochberg FDR correction.
 M3 Binding-transplant result → an encoding-vs-readout verdict with quantified recovery.
 M4 3–4 publication-quality figures + a 4–8 page workshop draft.

Stretch (post-MVP): OpenVLA action-lens replication (RQ1 cross-architecture); diagnostic→mitigation (RQ3).

5. Models
SmolVLA-450M — PRIMARY. Flow-matching policy on the SmolVLM-2 backbone (SigLIP vision + SmolLM2). HF: lerobot/smolvla_base, via LeRobot. Small enough for full-model patching on a single consumer GPU. Actions are continuous chunks → no action-token logits → use offline action divergence as the patching readout and a linear readout probe as the action-lens surrogate.
OpenVLA-7B — SECONDARY (stretch). Autoregressive; DINOv2+SigLIP vision, Llama-2 7B, 7-DoF actions discretized to 256 bins. The discrete action vocab gives a clean action lens (project the residual stream through the LM head restricted to the action-bin token ids). Run with LoRA + 4-bit on a 40GB A100.

Keep all model-specific logic behind a common interface (src/models/base.py): load(), forward_with_cache(inputs, sites), patch(run, site, value), predict_action(inputs). SmolVLA vs OpenVLA differences must not leak into src/interp/ or src/eval/.

6. Environments & data
LIBERO (sim manipulation; Lifelong-Robot-Learning/LIBERO, MuJoCo/robosuite). LIBERO-Goal is primary: scene and objects are fixed across tasks and only the commanded goal changes, so vision underdetermines the target and the instruction must be read. LIBERO-Spatial / -Object are robustness checks.
Contrastive pairs are generated from LIBERO. If a released counterfactual set (LIBERO-CF) is available, use it; otherwise reconstruct pairs by swapping the referent under a fixed layout (the same recipe). Confirm availability before depending on it.
SIMPLER (simpler-env/SimplerEnv) is a secondary real-to-sim check.
We train nothing: all analysis is forward passes with hooks on frozen public checkpoints.
7. Method — toolkit and the standing rule
Activation patching / causal mediation (primary, RQ1): run on instruction A, patch in an activation recorded on instruction B at a target site, measure the change in the action.
Path patching (RQ1, head level): patch a component only along its path to a chosen receiver → head-level circuit, not a heatmap.
Action lens (RQ1/RQ3, OpenVLA only): decode per-layer action distribution via the action-bin unembedding. SmolVLA has no unembedding → use a trained linear readout probe on the expert's conditioning vector.
SAE-feature patching (cross-check): patch an interpretable feature (SAEs via dictionary_learning) rather than a raw direction.
Binding transplant (RQ2, the decisive test): extract the object↔word binding direction (difference-of-means between A/B runs at the binding site, or a probe direction) and inject it into the failing run's expert input.
Linear probes: correlational companion only.

Standing rule (enforce in reviews): every headline claim needs at least two techniques with different failure modes agreeing. Knockout can miss information that reroutes; patching can push activations off-distribution.

8. Readouts, metrics, statistics
Offline action divergence = primary readout (L2 / cosine change in the predicted continuous action chunk on fixed states). Aggregate over thousands of paired states to resolve effects below closed-loop noise.
Closed-loop success rate = confirm the final shortlist only (it is slow and noisy).
IFR (instruction-following rate under contradiction) = the behavioral gap the map must explain.
Causal indirect effect: significant only above the 95th percentile of a position-shuffled null.
Logit-difference recovery (OpenVLA): ≥0.8 at the causal site, <0.2 at random sites.
Localization sharpness: "narrow" if ≥70% of total effect sits in ≤10% of sites.
Binding-transplant recovery: verdict is readout if recovery ≥50% of the clean-minus-failing gap.
Stats: paired bootstrap 95% CIs (10k resamples), permutation nulls, Benjamini-Hochberg FDR across the many candidate sites. Report effect sizes, not just significance.
9. Repo layout
where-does-language-die/
  CLAUDE.md                # this file
  README.md
  pyproject.toml           # pinned deps
  configs/                 # yaml: models, tasks, experiments (logged with every run)
  src/
    models/                # base.py + smolvla.py, openvla.py (behind common interface)
    data/                  # contrastive/counterfactual pair generation from LIBERO
    interp/                # patching, path_patching, action_lens, sae, binding_transplant
    eval/                  # ifr, metrics, stats (bootstrap, permutation, fdr)
    viz/                   # heatmaps, redirection plots
  scripts/                 # thin CLI entrypoints (call src/)
  experiments/             # runners; each writes its resolved config + results
  results/                 # cached activations + metrics (LARGE dirs gitignored)
  figures/
  paper/                   # workshop draft + final figures
  tests/                   # unit tests for interp + metrics
10. Commands (target CLI — build these as you go)
make setup                       # env, deps, download SmolVLA, LIBERO assets
python scripts/smoke_test.py     # load SmolVLA, run 1 LIBERO-Goal episode headless
python scripts/build_pairs.py    --suite libero_goal --n 200
python scripts/reproduce_ifr.py  --model smolvla --suite libero_goal      # M0
python scripts/run_localization.py --model smolvla --pairs <path>         # M2
python scripts/run_transplant.py   --model smolvla --pairs <path>         # M3
python scripts/make_figures.py                                            # M4
pytest -q                        # interp + metrics tests must pass
11. Conventions
Determinism: set and log seeds; every run dumps its fully-resolved config to results/<run_id>/config.yaml.
Cache aggressively: activations are expensive — cache to disk keyed by (model, input hash, site) and reuse.
Small-first: validate any new analysis on a handful of tasks/episodes and a couple of layers before scaling to the full sweep.
No fabricated results — ever. If an experiment isn't run, the number does not exist. Placeholders must be obviously marked TODO/NaN, never plausible fake values.
Typed core + tests for src/interp/ and src/eval/ (patching correctness and metric math are where silent bugs hide).
12. Gotchas & constraints
Flow-matching (SmolVLA) has no clean unembedding → anchor action-lens claims on OpenVLA; use the probe surrogate for SmolVLA and say so.
LIBERO/MuJoCo rendering is finicky headless (EGL/OSMesa); get smoke_test.py green before anything else.
OpenVLA-7B needs LoRA + 4-bit on a 40GB A100; SmolVLA is primary precisely to keep the core single-GPU.
Closed-loop eval is slow and noisy → drive fine-grained maps with offline action divergence; reserve closed-loop for the shortlist.
Ask before launching any run > ~1 GPU-hour or before downloading >10GB.
Verify nnsight (or raw forward-hook) compatibility with the exact SmolVLA / OpenVLA model classes early; do not assume TransformerLens supports them.
The field moves fast — before finalizing novelty claims, re-scan for concurrent causal-localization-of-VLA-grounding work.
13. Out of scope (do not build without explicit approval)
Real-robot / hardware execution.
Training or fine-tuning VLAs from scratch.
Novel steering or mitigation algorithms (we localize; RQ3 only matches known fixes to failure types).
Multi-GPU / cluster jobs.
14. Key references (for grounding, not to re-derive)
Behavioral targets: BeTTER (2604.18000), ICBench/linguistic blindness (2603.06001), LIBERO-CF/CAG (2602.17659), RoboSemanticBench (2606.02277).
VLA interp: CWRU "Not All Features" (2603.19233), Häon steering (2509.00328), VLA-Trace (2605.30117), event-grounded SAEs (2605.17204).
Methods: ROME causal tracing (2202.05262), IOI path patching (2211.00593), binding IDs (2310.17191; VLM 2505.22200), logit lens (nostalgebraist 2020).
Models/envs: SmolVLA (2506.01844), OpenVLA (2406.09246), LIBERO (2306.03310), SIMPLER (2405.05941).