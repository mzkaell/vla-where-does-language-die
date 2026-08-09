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

## Which numbers to quote

Use `fs_*`. The `comp_*` runs measure the same effect without holding the source state fixed
and give roughly **double** the gap; they are an upper bound, not the result. Both are kept
because the difference between them is itself informative about how much a plausible-looking
uncontrolled measurement can inflate.

## Not present yet

`loc_*` (M2 causal localization) and the M3 binding transplant. The sweep is implemented and
smoke-tested (`scripts/run_localization.py`) but needs a GPU — ~144 forward passes per trial
across 130 sites, roughly 27 CPU-hours versus ~1 GPU-hour. **No M2 numbers exist anywhere in
this repo**; anything you see referring to localization is a plan, not a result.

## Retracted

Two earlier result sets were deleted after controls contradicted them: a destination-vs-object
dissociation that came from a one-sided stimulus set, and a set of runs computed on
180°-rotated images. Both causes and their diagnostic signatures are written up in the findings
doc. They are reproducible from git history if anyone wants to see the failure modes directly.
