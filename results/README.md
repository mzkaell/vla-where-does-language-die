# Results index

Every directory holds the **fully-resolved config** that produced it alongside its metrics, so
any number here is traceable to the exact run. Large artefacts (cached activations, per-trial
dumps) are gitignored; configs and metrics are tracked.

Interpretation lives in [`../paper/m0_findings.md`](../paper/m0_findings.md) — read that
before drawing conclusions from any single file here.

| run | experiment | status |
|---|---|---|
| `checkpoint_screen.json` | Competence screen over all 8 public LIBERO SmolVLA checkpoints | **current** |
| `m0_v2_k1000dai` | Condition 1, neutral states, 400 counterbalanced pairs | **current** — null (policy follows instruction) |
| `m0_v2_msv6` | Same, second checkpoint | **excluded** — checkpoint fails the competence gate |
| `conflict_finetune` | Condition 2, visual conflict (post-grasp), 240 pairs | **current** — null (grounding improves) |
| `conflict_scratch80k` | Same, second checkpoint | **current** — null, replicates |
| `comp_finetune` | Condition 3, compositional, 480 trials | **superseded** by `fs_*` (confounded by source state) |
| `comp_scratch80k` | Same, second checkpoint | **superseded** |
| `fs_finetune` | Condition 3 with the **fixed-state control**, paired | **headline** |
| `fs_scratch80k` | Same, second checkpoint | **headline — replication** |
| `loc_full_mps` | **M2 full sweep**, 130 sites, n=40/contrast, MPS | **current** — graded VLM decay 0.73→0.00; diffuse; nothing clears BH |
| `m3_L4_lang` / `m3_L4_all` | M3 transplant at vlm.L4, lang-only vs whole prefix | **current** — lang block carries ~all signal; partial doses hurt |
| `m3_L0_lang` / `m3_L12_lang` | M3 lang-block transplant, depth pair | **current** — 0.74 at L0 → 0.37 at L12; decays with depth |
| `_loc_smoke` | M2 smoke, CPU, first 8 sites, n=3 | **smoke only** — sizes the sweep, answers nothing |
| `_loc_smoke_mps` | Same smoke on MPS | **smoke only** — same ranking; resid sites within 0.02, mlp up to 0.21; 24× faster |

## Which numbers to quote

Use `fs_*`. The `comp_*` runs measure the same effect without holding the source state fixed
and give roughly **double** the gap; they are an upper bound, not the result. Both are kept
because the difference between them is itself informative about how much a plausible-looking
uncontrolled measurement can inflate.

## Not present yet

An M2 replication on the second checkpoint is running (resid sites, CPU); OpenVLA
cross-architecture and closed-loop confirmation do not exist. Historical note kept
for the record: this section previously said M2/M3 needed a GPU. The GPU assumption in earlier
versions of this note is dead: MPS runs 24× faster than CPU, so the full 130-site, n=40 sweep
costs ~3 h on an M-series laptop. Agreement, stated precisely (an earlier version of this
note claimed a blanket "within 0.02", which the committed metrics contradict): site ranking
is identical; the four resid sites — the ones carrying the recovery signal — agree within
0.014; the small/noisy attn_out and mlp_out values differ by 0.03–0.21 and the null mean by
0.15. Cross-backend agreement on noisy negative sites is NOT established, which is a caveat
any MPS-run map inherits for those components. (The old "27 CPU-hours" figure was also 2.5×
optimistic against the measured CPU rate of ~16 s/forward.)
The smoke runs are sizing data only — 8 early-VLM sites, n=6 usable trials, nothing
significant after BH. **No full-sweep M2 numbers exist in this repo yet.**

## Retracted

Two earlier result sets were deleted after controls contradicted them: a destination-vs-object
dissociation that came from a one-sided stimulus set, and a set of runs computed on
180°-rotated images. Both causes and their diagnostic signatures are written up in the findings
doc. They are reproducible from git history if anyone wants to see the failure modes directly.
